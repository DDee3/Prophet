import functools
import io
import json
import os
import pickle
import re
import threading
import uuid
import xml.etree.ElementTree as ET

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
import numpy as np
from openai import OpenAI
import pandas as pd
import requests
import yfinance as yf

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "prophet_dev_secret_change_me")

HISTORY_FILE = 'history.json'
USERS_FILE = 'users.json'
FEATURE_ORDER = ['Close', 'SMA_20', 'SMA_50', 'Volume', 'Daily_Return', 'RSI']

_lock = threading.RLock()

try:
    with open('xgboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
except (FileNotFoundError, pickle.PickleError, Exception) as e:
    print(f"[WARN] Failed to load XGBoost model: {e}")
    model = None

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get(
    "OLLAMA_MODEL",
    "hf.co/HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive:Q4_K_M",
)

# Headers required for ngrok tunnels to bypass HTML interstitial warnings
_NGROK_HEADERS = {
    "ngrok-skip-browser-warning": "true",
    "User-Agent": "Prophet-Flask-App",
}

ollama_client = OpenAI(
    api_key="ollama",
    base_url=f"{OLLAMA_HOST}/v1",
    default_headers=_NGROK_HEADERS,
)

GROQ_MODELS = [
    os.environ.get("GROQ_MODEL", "qwen/qwen3.6-27b"),
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]
GROQ_MODEL = GROQ_MODELS[0]
groq_client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY", "missing"),
    base_url="https://api.groq.com/openai/v1",
)


def _load_json(path, default):
    with _lock:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    return data if isinstance(data, type(default)) else default
            except (json.JSONDecodeError, OSError):
                return default
        return default


def _save_json(path, data):
    with _lock:
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)


def load_history():
    return _load_json(HISTORY_FILE, [])


def save_history(data):
    _save_json(HISTORY_FILE, data)


def load_users():
    return _load_json(USERS_FILE, {})


def save_users(data):
    _save_json(USERS_FILE, data)


def _now_iso():
    return pd.Timestamp.utcnow().isoformat() + 'Z'


def get_or_create_user(email):
    email = email.strip().lower()
    with _lock:
        users = load_users()
        if email not in users:
            users[email] = {
                'email': email,
                'name': email.split('@')[0].capitalize(),
                'joined': _now_iso(),
                'last_seen': _now_iso(),
                'prefs': {
                    'theme': 'dark',
                    'chartType': 'candles',
                    'indicators': [],
                    'hiddenModels': [],
                },
                'watchlist': [],
                'portfolio': [],
            }
        else:
            users[email]['last_seen'] = _now_iso()
        save_users(users)
        return users[email]


def login_required(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        if 'user' not in session:
            if request.path.startswith('/api/') or request.is_json:
                return jsonify({'error': 'Unauthorized'}), 401
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return wrapper


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def detect_direction(text):
    if not text:
        return 'hold'
    text_lower = text.lower()
    bullish = len(re.findall(r'\b(bullish|buy|up|upside|increase|long|positive|rise|rising|gain|outperform|entry)\b', text_lower))
    bearish = len(re.findall(r'\b(bearish|sell|down|downside|decrease|short|negative|fall|falling|drop|loss|underperform|corrected)\b', text_lower))
    if bullish > bearish:
        return 'bull'
    if bearish > bullish:
        return 'bear'
    return 'hold'


def parse_llm_json(text):
    if not text:
        return None
    patterns = [r'```(?:json)?\s*(\{.*?\})\s*```', r'\{.*\}']
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1 if match.lastindex else 0))
            except json.JSONDecodeError:
                continue
    return None


def compute_feature_contributions(df):
    names = list(df.columns)
    row = df.iloc[0]

    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = np.array(explainer.shap_values(df)).reshape(-1)
        max_abs = float(np.max(np.abs(shap_values))) or 1.0
        return [
            {'name': n, 'value': float(row[n]), 'contribution': float(v) / max_abs}
            for n, v in zip(names, shap_values)
        ]
    except Exception:
        pass

    importances = np.array(getattr(model, 'feature_importances_', np.ones(len(names)) / len(names)))
    deviations = []
    for n in names:
        val = float(row[n])
        if n == 'RSI':
            deviations.append((val - 50) / 30)
        elif n == 'Volume':
            deviations.append((np.log10(max(val, 1)) - 7) / 3)
        elif n == 'Daily_Return':
            deviations.append(val / 0.03)
        elif n in ('SMA_20', 'SMA_50'):
            deviations.append((float(row['Close']) - val) / max(val, 1e-6) * 5)
        else:
            deviations.append(0.0)

    contributions = importances * np.array(deviations)
    max_abs = float(np.max(np.abs(contributions))) or 1.0
    return [
        {'name': n, 'value': float(row[n]), 'contribution': float(c) / max_abs}
        for n, c in zip(names, contributions)
    ]


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = (request.form.get('email') or '').strip().lower()
        if email:
            session['user'] = email
            get_or_create_user(email)
            return redirect(url_for('home'))
    return render_template('login.html')


@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))


@app.route('/')
@login_required
def home():
    return render_template('index.html')


@app.route('/history')
@login_required
def history():
    return render_template('history.html')


@app.route('/profile')
@login_required
def profile():
    return render_template('profile.html')


@app.route('/data/<entry_id>')
@login_required
def data_page(entry_id):
    email = session['user']
    with _lock:
        history_list = load_history()

    entry = next(
        (e for e in history_list
         if str(e.get('id')) == str(entry_id)
         and e.get('user_email') == email),
        None
    )
    if not entry:
        return render_template('data.html', entry={'error': 'not found'}), 404

    return render_template('data.html', entry=entry)


@app.route('/api/profile', methods=['GET', 'PATCH'])
@login_required
def api_profile():
    email = session['user']

    if request.method == 'PATCH':
        patch = request.get_json(silent=True) or {}
        with _lock:
            users = load_users()
            user = users.get(email)
            if not user:
                return jsonify({'error': 'User not found'}), 404
            if 'name' in patch:
                user['name'] = str(patch['name'])[:50].strip() or user['name']
            if 'prefs' in patch and isinstance(patch['prefs'], dict):
                user.setdefault('prefs', {}).update(patch['prefs'])
            save_users(users)
            return jsonify({'success': True, 'user': user})

    return jsonify(get_or_create_user(email))


@app.route('/api/watchlist', methods=['GET', 'POST'])
@login_required
def api_watchlist():
    email = session['user']

    if request.method == 'POST':
        payload = request.get_json(silent=True) or {}
        ticker = str(payload.get('ticker', '')).upper().strip()
        if not ticker:
            return jsonify({'error': 'ticker required'}), 400
        with _lock:
            users = load_users()
            user = users.setdefault(email, get_or_create_user(email))
            watchlist = user.setdefault('watchlist', [])
            if ticker not in watchlist:
                watchlist.append(ticker)
                save_users(users)
            return jsonify({'success': True, 'watchlist': watchlist, 'added': ticker})

    user = get_or_create_user(email)
    return jsonify({'watchlist': user.get('watchlist', [])})


@app.route('/api/watchlist/<ticker>', methods=['DELETE'])
@login_required
def api_watchlist_delete(ticker):
    email = session['user']
    ticker = ticker.upper().strip()
    with _lock:
        users = load_users()
        user = users.get(email)
        if not user:
            return jsonify({'error': 'User not found'}), 404
        user['watchlist'] = [t for t in user.get('watchlist', []) if t != ticker]
        save_users(users)
        return jsonify({'success': True, 'watchlist': user['watchlist']})


@app.route('/api/history', methods=['GET', 'POST'])
@login_required
def api_history():
    email = session['user']

    if request.method == 'POST':
        new_entry = request.get_json(silent=True) or {}
        new_entry['id'] = str(uuid.uuid4())
        new_entry['user_email'] = email
        new_entry.setdefault('time', _now_iso())
        with _lock:
            history_data = load_history()
            history_data.insert(0, new_entry)
            save_history(history_data)
        return jsonify({'success': True, 'id': new_entry['id']})

    with _lock:
        history_data = load_history()
    user_history = [e for e in history_data if e.get('user_email') == email]
    return jsonify(user_history)


@app.route('/api/history/<entry_id>', methods=['DELETE'])
@login_required
def delete_history_entry(entry_id):
    email = session['user']
    with _lock:
        history_data = load_history()
        before_count = len(history_data)
        history_data = [
            e for e in history_data
            if not (e.get('id') == entry_id and e.get('user_email') == email)
        ]
        if len(history_data) == before_count:
            return jsonify({'error': 'Entry not found'}), 404
        save_history(history_data)
    return jsonify({'success': True})


@app.route('/api/history/bulk-delete', methods=['POST'])
@login_required
def bulk_delete_history():
    email = session['user']
    payload = request.get_json(silent=True) or {}
    ids = payload.get('ids')
    if not isinstance(ids, list) or not ids:
        return jsonify({'error': 'ids array required'}), 400

    id_set = set(str(i) for i in ids)
    with _lock:
        history_data = load_history()
        before_count = len(history_data)
        history_data = [
            e for e in history_data
            if not (e.get('id') in id_set and e.get('user_email') == email)
        ]
        removed = before_count - len(history_data)
        save_history(history_data)
    return jsonify({'success': True, 'removed': removed, 'total': len(history_data)})


@app.route('/api/history/<entry_id>', methods=['PATCH'])
@login_required
def update_history_entry(entry_id):
    email = session['user']
    patch = request.get_json(silent=True) or {}
    with _lock:
        history_data = load_history()
        for entry in history_data:
            if entry.get('id') == entry_id and entry.get('user_email') == email:
                if 'ticker' in patch:
                    entry['ticker'] = str(patch['ticker']).upper()
                if 'note' in patch:
                    entry['note'] = patch['note']
                save_history(history_data)
                return jsonify({'success': True, 'entry': entry})
    return jsonify({'error': 'Entry not found'}), 404


@app.route('/tickers', methods=['GET'])
def get_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; ProphetBot/1.0)'}
    try:
        r = requests.get(url, headers=headers, timeout=10)
        tables = pd.read_html(io.StringIO(r.text))
        tickers = tables[0]['Symbol'].tolist()
        return jsonify({'tickers': [t.replace('.', '-') for t in tickers]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/chart-data', methods=['POST'])
@login_required
def chart_data():
    ticker = (request.get_json(silent=True) or {}).get('ticker', '').upper().strip()
    if not ticker:
        return jsonify({'error': 'ticker required'}), 400

    try:
        hist = yf.Ticker(ticker).history(period='6mo')
        if hist.empty:
            return jsonify({'error': f'No data for {ticker}'}), 404

        data = [
            {
                'time': idx.strftime('%Y-%m-%d'),
                'open': float(row['Open']),
                'high': float(row['High']),
                'low': float(row['Low']),
                'close': float(row['Close']),
                'volume': float(row['Volume']) if 'Volume' in row and not pd.isna(row['Volume']) else 0,
            }
            for idx, row in hist.iterrows()
        ]
        return jsonify({'data': data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/news', methods=['POST'])
@login_required
def get_news():
    ticker = (request.get_json(silent=True) or {}).get('ticker', '').upper().strip()
    if not ticker:
        return jsonify({'error': 'ticker required'}), 400

    try:
        resp = requests.get(
            'https://news.google.com/rss/search',
            params={'q': f'{ticker} stock', 'hl': 'en-US', 'gl': 'US', 'ceid': 'US:en'},
            headers={'User-Agent': 'Mozilla/5.0'},
            timeout=10,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.content)

        news = []
        for item in root.findall('.//channel/item')[:5]:
            source_node = item.find('source')
            news.append({
                'title': item.findtext('title', ''),
                'link': item.findtext('link', ''),
                'publisher': source_node.text if source_node is not None else 'Google News',
            })
        return jsonify({'news': news})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict/xgboost', methods=['POST'])
@login_required
def predict_xgboost():
    ticker = (request.get_json(silent=True) or {}).get('ticker', '').upper().strip()
    if not ticker:
        return jsonify({'error': 'ticker required'}), 400
    if model is None:
        return jsonify({'error': 'XGBoost model not loaded on server.'}), 500

    try:
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
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(df)[0]
            confidence = int(round(float(max(proba)) * 100))
        else:
            confidence = 65

        verdict = 'bull' if pred == 1 else 'bear'
        direction = 'Up (Bullish)' if pred == 1 else 'Down (Bearish)'

        return jsonify({
            'prediction': f'XGBoost Technical Analysis: The trend is predicted to go {direction}.',
            'verdict': verdict,
            'confidence': confidence,
            'features': compute_feature_contributions(df),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


SYSTEM_PROMPT = (
    "You are a stock market analyst. Reply with ONLY a valid JSON object. "
    "Do not include markdown, code fences, or any text outside the JSON.\n"
    "\n"
    "The JSON must have exactly these keys:\n"
    '  "headline":   one short sentence summarizing your call (max 12 words),\n'
    '  "verdict":    one of the exact strings "bull", "bear", or "hold",\n'
    '  "confidence": an integer between 0 and 100,\n'
    '  "key_points": array of 2-4 short bullet reasons (each max 8 words),\n'
    '  "risks":      array of 0-2 short risk flags (each max 8 words),\n'
    '  "analysis":   a 2-3 sentence full analysis (for detail view).\n'
    "\n"
    "Example of a valid response:\n"
    '{"headline": "Uptrend intact, watch resistance", "verdict": "bull", "confidence": 72, '
    '"key_points": ["5% gain over past month", "Broke above $225 resistance", "Strong AI chip demand"], '
    '"risks": ["Valuation elevated"], '
    '"analysis": "NVDA has risen about 5% over the past month, indicating positive momentum and breaking recent resistance around $225. The gain, coupled with strong demand for AI chips, suggests further upside, though valuation remains elevated."}\n'
    "\n"
    "Output MUST start with { and end with }."
)


def _build_user_prompt(ticker, price_info):
    return (
        f'Analyze the current outlook for {ticker}. {price_info} '
        'Respond with ONLY the JSON object.'
    )


def _openai_style_call(client, model_name, messages, json_mode=True, timeout=90, reasoning_effort=None):
    kwargs = {
        "model": model_name,
        "messages": messages,
        "timeout": timeout,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort

    try:
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content
    except Exception:
        if json_mode:
            kwargs.pop("response_format", None)
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        raise


def _call_llm(ai_model, ticker, price_info):
    user_prompt = _build_user_prompt(ticker, price_info)
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]

    ollama_call = ("Ollama", ollama_client, OLLAMA_MODEL, False, "none", 120)
    groq_calls = [(f"Groq ({m})", groq_client, m, True, None, 45) for m in GROQ_MODELS]

    calls = groq_calls + [ollama_call] if ai_model == 'groq' else [ollama_call] + groq_calls

    last_error = None
    for name, client, model_name, json_mode, reasoning_effort, timeout in calls:
        try:
            result = _openai_style_call(
                client, model_name, messages,
                json_mode=json_mode,
                reasoning_effort=reasoning_effort,
                timeout=timeout,
            )
            if result and str(result).strip():
                return result
            last_error = RuntimeError(f"{name} returned empty content")
        except Exception as e:
            last_error = e

    raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")


@app.route('/predict/ai', methods=['POST'])
@login_required
def predict_ai():
    data = request.get_json(silent=True) or {}
    ticker = (data.get('ticker') or '').upper().strip()
    ai_model = data.get('model') or 'ollama'

    if not ticker:
        return jsonify({'error': 'ticker required'}), 400

    price_info = ''
    try:
        hist = yf.Ticker(ticker).history(period='1mo')
        if not hist.empty:
            current_price = round(float(hist['Close'].iloc[-1]), 2)
            past_price = round(float(hist['Close'].iloc[0]), 2)
            price_info = f'The current price is ${current_price}. One month ago, it was ${past_price}.'
    except Exception:
        pass

    try:
        raw = _call_llm(ai_model, ticker, price_info)
    except Exception as e:
        return jsonify({'error': f'API Error: {e}'}), 502

    parsed = parse_llm_json(raw) or {}
    analysis = parsed.get('analysis') or raw
    verdict = parsed.get('verdict')

    if verdict not in ('bull', 'bear', 'hold'):
        verdict = detect_direction(analysis)

    confidence = parsed.get('confidence')
    if isinstance(confidence, (int, float)):
        confidence = max(0, min(100, int(confidence)))
    else:
        confidence = None

    key_points = parsed.get('key_points')
    risks = parsed.get('risks')

    return jsonify({
        'prediction': analysis,
        'verdict': verdict,
        'confidence': confidence,
        'headline': parsed.get('headline') or None,
        'key_points': key_points if isinstance(key_points, list) else None,
        'risks': risks if isinstance(risks, list) else None,
    })


@app.route('/health/llm', methods=['GET'])
def health_llm():
    results = {}

    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", headers=_NGROK_HEADERS, timeout=5)
        if r.status_code == 200:
            try:
                models = [m.get('name', '') for m in r.json().get('models', [])]
                has_model = any(OLLAMA_MODEL in m for m in models)
                results['ollama'] = {
                    'ok': has_model,
                    'host': OLLAMA_HOST,
                    'model_configured': OLLAMA_MODEL,
                    'models_available': models,
                    'note': 'model present' if has_model else 'model NOT pulled on the server',
                }
            except ValueError as e:
                results['ollama'] = {'ok': False, 'host': OLLAMA_HOST, 'error': f'Bad JSON response: {e}'}
        else:
            results['ollama'] = {'ok': False, 'host': OLLAMA_HOST, 'status': r.status_code}
    except Exception as e:
        results['ollama'] = {'ok': False, 'host': OLLAMA_HOST, 'error': str(e)}

    key = os.environ.get('GROQ_API_KEY', '')
    if not key or key == 'missing':
        results['groq'] = {'ok': False, 'error': 'GROQ_API_KEY not set'}
    else:
        groq_status = {'ok': False, 'models': {}}
        for model_name in GROQ_MODELS:
            try:
                resp = groq_client.chat.completions.create(
                    model=model_name,
                    messages=[{'role': 'user', 'content': 'Reply with only: pong'}],
                    max_completion_tokens=20,
                    timeout=15,
                )
                text = (resp.choices[0].message.content or '').strip()
                groq_status['models'][model_name] = {
                    'ok': bool(text),
                    'reply': text[:40] if text else '(empty)',
                }
                if text:
                    groq_status['ok'] = True
            except Exception as e:
                groq_status['models'][model_name] = {
                    'ok': False,
                    'error': f'{type(e).__name__}: {str(e)[:120]}',
                }
        results['groq'] = groq_status

    return jsonify(results)


if __name__ == '__main__':
    print("─" * 60)
    print(f"  Ollama host  : {OLLAMA_HOST}")
    print(f"  Ollama model : {OLLAMA_MODEL}")
    print(f"  Groq primary : {GROQ_MODEL}")
    print(f"  Groq backups : {', '.join(GROQ_MODELS[1:])}")
    print("─" * 60)
    app.run(debug=True, port=5000)