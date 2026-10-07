from flask import Flask, request, jsonify, render_template, session, redirect, url_for
import pandas as pd
import numpy as np
import pickle
import requests
import io
import os
import json
import uuid
import re
import functools
import threading
import yfinance as yf
from dotenv import load_dotenv
from google import genai
from openai import OpenAI


load_dotenv()

app = Flask(__name__)
HISTORY_FILE = 'history.json'
FEATURE_ORDER = ['Close', 'SMA_20', 'SMA_50', 'Volume', 'Daily_Return', 'RSI']
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "prophet_dev_secret_change_me")

# Reentrant lock guards every read-modify-write of the history file so
# concurrent requests can't clobber each other or write a partial file.
_history_lock = threading.RLock()


# ═══════════════════════════════════════════════════════════
# MODEL LOAD
# ═══════════════════════════════════════════════════════════
try:
    with open('xgboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"[WARN] Failed to load XGBoost model: {e}")
    model = None


# ═══════════════════════════════════════════════════════════
# LLM CLIENTS
# ═══════════════════════════════════════════════════════════
gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

groq_client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

deepseek_client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

openrouter_client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)


# ═══════════════════════════════════════════════════════════
# HISTORY HELPERS — lock-protected + atomic writes
# ═══════════════════════════════════════════════════════════
def load_history():
    with _history_lock:
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, 'r') as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else []
            except Exception:
                return []
        return []


def save_history(data):
    with _history_lock:
        # Write to a temp file, then atomically swap — never leaves a
        # half-written history.json on disk if the process dies mid-write.
        tmp = HISTORY_FILE + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=4)
        os.replace(tmp, HISTORY_FILE)


# ═══════════════════════════════════════════════════════════
# AUTH DECORATOR
# ═══════════════════════════════════════════════════════════
def login_required(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if 'user' not in session:
            if request.path.startswith('/api/') or request.is_json:
                return jsonify({'error': 'Unauthorized'}), 401
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrapper


# ═══════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def detect_direction(text):
    """Server-side fallback verdict classifier."""
    if not text:
        return 'hold'
    t = text.lower()
    up = len(re.findall(r'\b(bullish|buy|up|upside|increase|long|positive|rise|rising|gain|outperform|entry)\b', t))
    down = len(re.findall(r'\b(bearish|sell|down|downside|decrease|short|negative|fall|falling|drop|loss|underperform|corrected)\b', t))
    if up > down:
        return 'bull'
    if down > up:
        return 'bear'
    return 'hold'


def parse_llm_json(text):
    """Pull a JSON object out of an LLM response."""
    if not text:
        return None
    m = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


def compute_feature_contributions(df):
    """
    Per-feature contribution estimates for a single-row DataFrame.
    Prefers SHAP (if installed); falls back to importance × normalized deviation.
    """
    names = list(df.columns)
    row = df.iloc[0]

    # Preferred: real SHAP values from the tree
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(df)
        arr = np.array(vals).reshape(-1)
        max_abs = float(np.max(np.abs(arr))) or 1.0
        return [
            {
                'name': n,
                'value': float(row[n]),
                'contribution': float(v) / max_abs,
            }
            for n, v in zip(names, arr)
        ]
    except Exception:
        pass

    # Fallback: importance × normalized deviation from neutral baseline
    importances = np.array(
        getattr(model, 'feature_importances_', np.ones(len(names)) / len(names))
    )

    devs = []
    for n in names:
        v = float(row[n])
        if n == 'RSI':
            devs.append((v - 50) / 30)
        elif n == 'Volume':
            devs.append((np.log10(max(v, 1)) - 7) / 3)
        elif n == 'Daily_Return':
            devs.append(v / 0.03)
        elif n in ('SMA_20', 'SMA_50'):
            devs.append((float(row['Close']) - v) / max(v, 1e-6) * 5)
        else:
            devs.append(0.0)

    contribs = importances * np.array(devs)
    max_abs = float(np.max(np.abs(contribs))) or 1.0
    return [
        {
            'name': n,
            'value': float(row[n]),
            'contribution': float(c) / max_abs,
        }
        for n, c in zip(names, contribs)
    ]


# ═══════════════════════════════════════════════════════════
# PAGES
# ═══════════════════════════════════════════════════════════
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        session['user'] = request.form.get('email')
        return redirect(url_for('home'))
    return render_template('login.html')


@app.route('/')
@login_required
def home():
    return render_template('index.html')


@app.route('/history')
@login_required
def history():
    return render_template('history.html')


# ═══════════════════════════════════════════════════════════
# HISTORY API
# ═══════════════════════════════════════════════════════════
@app.route('/api/history', methods=['GET', 'POST'])
@login_required
def api_history():
    if request.method == 'POST':
        new_entry = request.get_json(silent=True) or {}
        new_entry['id'] = str(uuid.uuid4())
        new_entry.setdefault('time', pd.Timestamp.utcnow().isoformat() + 'Z')

        # Hold the lock around read-modify-write
        with _history_lock:
            history_data = load_history()
            history_data.insert(0, new_entry)
            save_history(history_data)

        return jsonify({'success': True, 'id': new_entry['id']})

    return jsonify(load_history())


@app.route('/api/history/<entry_id>', methods=['DELETE'])
@login_required
def delete_history_entry(entry_id):
    with _history_lock:
        history_data = load_history()
        before = len(history_data)
        history_data = [e for e in history_data if e.get('id') != entry_id]
        if len(history_data) == before:
            return jsonify({'error': 'Entry not found'}), 404
        save_history(history_data)

    return jsonify({'success': True})


@app.route('/api/history/bulk-delete', methods=['POST'])
@login_required
def bulk_delete_history():
    payload = request.get_json(silent=True) or {}
    ids = payload.get('ids')
    if not isinstance(ids, list) or not ids:
        return jsonify({'error': 'ids array required'}), 400

    id_set = set(str(i) for i in ids)

    with _history_lock:
        history_data = load_history()
        before = len(history_data)
        history_data = [e for e in history_data if e.get('id') not in id_set]
        removed = before - len(history_data)
        save_history(history_data)

    return jsonify({'success': True, 'removed': removed, 'total': len(history_data)})


@app.route('/api/history/<entry_id>', methods=['PATCH'])
@login_required
def update_history_entry(entry_id):
    patch = request.get_json(silent=True) or {}

    with _history_lock:
        history_data = load_history()
        for entry in history_data:
            if entry.get('id') == entry_id:
                if 'ticker' in patch:
                    entry['ticker'] = str(patch['ticker']).upper()
                if 'note' in patch:
                    entry['note'] = patch['note']
                save_history(history_data)
                return jsonify({'success': True, 'entry': entry})

    return jsonify({'error': 'Entry not found'}), 404


# ═══════════════════════════════════════════════════════════
# TICKERS
# ═══════════════════════════════════════════════════════════
@app.route('/tickers', methods=['GET'])
def get_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; ProphetBot/1.0)'}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        tables = pd.read_html(io.StringIO(r.text))
        tickers = tables[0]['Symbol'].tolist()
        return jsonify({'tickers': tickers})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════
# CHART DATA
# ═══════════════════════════════════════════════════════════
@app.route('/chart-data', methods=['POST'])
@login_required
def chart_data():
    try:
        ticker = (request.json or {}).get('ticker', '').upper().strip()
        if not ticker:
            return jsonify({'error': 'ticker required'}), 400

        hist = yf.Ticker(ticker).history(period='6mo')
        if hist.empty:
            return jsonify({'error': f'No data for {ticker}'}), 404

        data = []
        for idx, row in hist.iterrows():
            data.append({
                'time': idx.strftime('%Y-%m-%d'),
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
                'volume': float(row['Volume']) if 'Volume' in row and not pd.isna(row['Volume']) else 0,
            })
        return jsonify({'data': data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════
# NEWS
# ═══════════════════════════════════════════════════════════
@app.route('/news', methods=['POST'])
@login_required
def get_news():
    import urllib.request
    import xml.etree.ElementTree as ET
    from urllib.parse import quote

    try:
        ticker = (request.json or {}).get('ticker', '').upper().strip()
        if not ticker:
            return jsonify({'error': 'ticker required'}), 400

        query = quote(f'{ticker} stock')
        url = f'https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        xml_data = urllib.request.urlopen(req, timeout=10).read()
        root = ET.fromstring(xml_data)

        news = []
        for item in root.findall('.//channel/item')[:5]:
            source_node = item.find('source')
            news.append({
                'title': item.find('title').text,
                'link': item.find('link').text,
                'publisher': source_node.text if source_node is not None else 'Google News',
            })
        return jsonify({'news': news})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════
# XGBOOST PREDICTION
# ═══════════════════════════════════════════════════════════
@app.route('/predict/xgboost', methods=['POST'])
@login_required
def predict_xgboost():
    try:
        ticker = (request.get_json() or {}).get('ticker', '').upper().strip()
        if not ticker:
            return jsonify({'error': 'ticker required'}), 400
        if model is None:
            return jsonify({'error': 'XGBoost model not loaded on server.'}), 500

        hist = yf.Ticker(ticker).history(period='3mo')
        if hist.empty:
            return jsonify({'error': f'No data available for {ticker}.'}), 404

        hist['SMA_20'] = hist['Close'].rolling(20).mean()
        hist['SMA_50'] = hist['Close'].rolling(50).mean()
        hist['Daily_Return'] = hist['Close'].pct_change()
        hist['RSI'] = calculate_rsi(hist['Close'])
        hist = hist.dropna()
        if hist.empty:
            return jsonify({'error': 'Not enough data to calculate indicators.'}), 422

        latest = hist.iloc[-1]
        df = pd.DataFrame([{
            'Close': float(latest['Close']),
            'SMA_20': float(latest['SMA_20']),
            'SMA_50': float(latest['SMA_50']),
            'Volume': float(latest['Volume']),
            'Daily_Return': float(latest['Daily_Return']),
            'RSI': float(latest['RSI']),
        }])[FEATURE_ORDER]

        pred = int(model.predict(df)[0])

        # Real confidence from predict_proba when available
        try:
            proba = model.predict_proba(df)[0]
            confidence = int(round(float(max(proba)) * 100))
        except Exception:
            confidence = 65

        verdict = 'bull' if pred == 1 else 'bear'
        direction = 'Up (Bullish)' if pred == 1 else 'Down (Bearish)'
        text = f'XGBoost Technical Analysis: The trend is predicted to go {direction}.'

        features = compute_feature_contributions(df)

        return jsonify({
            'prediction': text,
            'verdict': verdict,
            'confidence': confidence,
            'features': features,
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════
# AI PREDICTIONS
# ═══════════════════════════════════════════════════════════
SYSTEM_PROMPT = (
    "You are a professional stock market analyst. Respond with ONLY a valid JSON object, "
    "no markdown, no commentary, matching this exact schema: "
    '{"analysis": "<3-4 sentence analysis>", "verdict": "bull|bear|hold", '
    '"confidence": <integer 0-100>}. '
    "The 'verdict' must be your directional call. "
    "'confidence' must reflect your certainty (0 = none, 100 = absolute)."
)


def _build_user_prompt(ticker, price_info):
    return (
        f'Analyze the current outlook for {ticker}. {price_info} '
        'Mention these exact prices in your analysis. '
        'Return ONLY the JSON object described in the system prompt.'
    )


def _openai_style_call(client, model_name, messages):
    """Try JSON mode first; fall back to plain if the provider rejects it."""
    try:
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
            response_format={'type': 'json_object'},
        ).choices[0].message.content
    except Exception:
        return client.chat.completions.create(
            model=model_name,
            messages=messages,
        ).choices[0].message.content


def _call_llm(ai_model, ticker, price_info):
    user_prompt = _build_user_prompt(ticker, price_info)
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]

    if ai_model == 'groq':
        return _openai_style_call(groq_client, 'qwen/qwen3.8-27b', messages)

    if ai_model == 'deepseek':
        return _openai_style_call(deepseek_client, 'deepseek-chat', messages)

    if ai_model == 'openrouter':
        return _openai_style_call(openrouter_client, 'qwen/qwen3.8-27b:free', messages)

    if ai_model == 'gemini':
        try:
            resp = gemini_client.models.generate_content(
                model='gemini-3.8-flash',
                contents=f'{SYSTEM_PROMPT}\n\n{user_prompt}',
                config={'response_mime_type': 'application/json'},
            )
            return resp.text
        except Exception:
            resp = gemini_client.models.generate_content(
                model='gemini-3.8-flash',
                contents=f'{SYSTEM_PROMPT}\n\n{user_prompt}',
            )
            return resp.text

    raise ValueError(f'Unknown model: {ai_model}')


@app.route('/predict/ai', methods=['POST'])
@login_required
def predict_ai():
    data = request.get_json() or {}
    ticker = (data.get('ticker') or '').upper().strip()
    ai_model = data.get('model')

    if not ticker:
        return jsonify({'error': 'ticker required'}), 400
    if not ai_model:
        return jsonify({'error': 'model required'}), 400

    # Price context
    try:
        hist = yf.Ticker(ticker).history(period='1mo')
        if not hist.empty:
            current_price = round(float(hist['Close'].iloc[-1]), 2)
            past_price = round(float(hist['Close'].iloc[0]), 2)
            price_info = (
                f'The current price is ${current_price}. '
                f'One month ago, it was ${past_price}.'
            )
        else:
            price_info = ''
    except Exception:
        price_info = ''

    try:
        raw = _call_llm(ai_model, ticker, price_info)
    except Exception as e:
        return jsonify({'error': f'API Error: {e}'}), 502

    # Structured output, with fallbacks
    parsed = parse_llm_json(raw) or {}
    analysis = parsed.get('analysis') or raw
    verdict = parsed.get('verdict')

    if verdict not in ('bull', 'bear', 'hold'):
        verdict = detect_direction(analysis)

    try:
        confidence = int(parsed.get('confidence'))
        confidence = max(0, min(100, confidence))
    except (TypeError, ValueError):
        confidence = None

    return jsonify({
        'prediction': analysis,
        'verdict': verdict,
        'confidence': confidence,
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)