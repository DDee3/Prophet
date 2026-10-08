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
from openai import OpenAI


load_dotenv()

app = Flask(__name__)
HISTORY_FILE = 'history.json'
USERS_FILE = 'users.json'
FEATURE_ORDER = ['Close', 'SMA_20', 'SMA_50', 'Volume', 'Daily_Return', 'RSI']
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "prophet_dev_secret_change_me")

_lock = threading.RLock()


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
# ── Primary: Ollama (local or Colab via ngrok) ──
OLLAMA_HOST  = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get(
    "OLLAMA_MODEL",
    "hf.co/HauhauCS/Qwen3.5-4B-Uncensored-HauhauCS-Aggressive:Q4_K_M",
)

# Headers required for free ngrok tunnels — otherwise ngrok returns an HTML
# warning page instead of JSON, and the OpenAI SDK chokes on it.
_NGROK_HEADERS = {
    "ngrok-skip-browser-warning": "true",
    "User-Agent": "Prophet-Flask-App",
}

ollama_client = OpenAI(
    api_key="ollama",                       # any non-empty string works
    base_url=f"{OLLAMA_HOST}/v1",
    default_headers=_NGROK_HEADERS,
)

# ── Fallback: Groq (cloud) ──
# NOTE: llama-3.3-70b-versatile was deprecated by Groq on 2026-08-16.
# Current recommendation: openai/gpt-oss-120b or qwen/qwen3.6-27b
GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
groq_client = OpenAI(
    api_key=os.environ.get("GROQ_API_KEY", "missing"),
    base_url="https://api.groq.com/openai/v1",
)


# ═══════════════════════════════════════════════════════════
# JSON STORAGE HELPERS (atomic, locked)
# ═══════════════════════════════════════════════════════════
def _load_json(path, default):
    with _lock:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                    return data if isinstance(data, type(default)) else default
            except Exception:
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


# ═══════════════════════════════════════════════════════════
# USER MANAGEMENT
# ═══════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════
# AUTH
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
    """Extract JSON from LLM output, tolerant of markdown fences and stray text."""
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
    names = list(df.columns)
    row = df.iloc[0]

    try:
        import shap
        explainer = shap.TreeExplainer(model)
        vals = explainer.shap_values(df)
        arr = np.array(vals).reshape(-1)
        max_abs = float(np.max(np.abs(arr))) or 1.0
        return [
            {'name': n, 'value': float(row[n]), 'contribution': float(v) / max_abs}
            for n, v in zip(names, arr)
        ]
    except Exception:
        pass

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
        {'name': n, 'value': float(row[n]), 'contribution': float(c) / max_abs}
        for n, c in zip(names, contribs)
    ]


# ═══════════════════════════════════════════════════════════
# AUTH ROUTES
# ═══════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════
# PAGES
# ═══════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════
# PROFILE API
# ═══════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════
# WATCHLIST API
# ═══════════════════════════════════════════════════════════
@app.route('/api/watchlist', methods=['GET', 'POST'])
@login_required
def api_watchlist():
    email = session['user']
    get_or_create_user(email)

    if request.method == 'POST':
        payload = request.get_json(silent=True) or {}
        ticker = str(payload.get('ticker', '')).upper().strip()
        if not ticker:
            return jsonify({'error': 'ticker required'}), 400
        with _lock:
            users = load_users()
            user = users.get(email, {})
            watchlist = user.get('watchlist', [])
            if ticker not in watchlist:
                watchlist.append(ticker)
                user['watchlist'] = watchlist
                users[email] = user
                save_users(users)
            return jsonify({'success': True, 'watchlist': watchlist, 'added': ticker})

    with _lock:
        users = load_users()
        watchlist = users.get(email, {}).get('watchlist', [])
    return jsonify({'watchlist': watchlist})


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


# ═══════════════════════════════════════════════════════════
# HISTORY API (per-user)
# ═══════════════════════════════════════════════════════════
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
        all_history = load_history()
    filtered = [e for e in all_history if e.get('user_email') == email]
    return jsonify(filtered)


@app.route('/api/history/<entry_id>', methods=['DELETE'])
@login_required
def delete_history_entry(entry_id):
    email = session['user']
    with _lock:
        history_data = load_history()
        before = len(history_data)
        history_data = [
            e for e in history_data
            if not (e.get('id') == entry_id and e.get('user_email') == email)
        ]
        if len(history_data) == before:
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
        before = len(history_data)
        history_data = [
            e for e in history_data
            if not (e.get('id') in id_set and e.get('user_email') == email)
        ]
        removed = before - len(history_data)
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
        return jsonify({'tickers': [t.replace('.', '-') for t in tickers]})
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


def _openai_style_call(client, model_name, messages, json_mode=True,
                       timeout=90, reasoning_effort=None):
    """
    Call an OpenAI-compatible chat completions endpoint.

    - json_mode: tries response_format={"type":"json_object"}, retries without it on error
    - reasoning_effort: pass "none" to disable thinking on models that support it
      (Qwen 3.5 via Ollama accepts this; without it, responses take 15s+ and burn tokens)
    """
    kwargs = {
        "model": model_name,
        "messages": messages,
        "timeout": timeout,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if reasoning_effort:
        # Send via extra_body so the OpenAI SDK doesn't strip it
        kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}

    try:
        resp = client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content
    except Exception as e:
        if json_mode:
            print(f"[LLM] JSON mode failed for {model_name}: {e}. Retrying without it...")
            kwargs.pop("response_format", None)
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        raise


def _call_llm(ai_model, ticker, price_info):
    """
    Get a prediction from an LLM with automatic failover.

    Primary:  Ollama (local or Colab via ngrok) — free, private, no rate limits
    Fallback: Groq (cloud)                      — free tier, 14,400 req/day
    """
    user_prompt = _build_user_prompt(ticker, price_info)
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_prompt},
    ]

    # Order of preference — first success wins
    engines = [
        (
            "Ollama",
            lambda: _openai_style_call(
                ollama_client, OLLAMA_MODEL, messages,
                json_mode=False,           # Ollama's OpenAI shim can be picky about response_format
                reasoning_effort="none",   # <-- the important fix: kills the "thinking" phase
                timeout=120,
            ),
        ),
        (
            "Groq",
            lambda: _openai_style_call(
                groq_client, GROQ_MODEL, messages,
                json_mode=True,
                timeout=45,
            ),
        ),
    ]

    last_exception = None
    for name, call in engines:
        try:
            print(f"[LLM] Trying {name}...")
            result = call()
            if result:
                print(f"[LLM] ✓ {name} responded.")
                return result
            print(f"[LLM] {name} returned empty response.")
        except Exception as e:
            print(f"[LLM] ✗ {name} failed: {type(e).__name__}: {e}")
            last_exception = e

    raise RuntimeError(f"All LLM providers failed. Last error: {last_exception}")


@app.route('/predict/ai', methods=['POST'])
@login_required
def predict_ai():
    data = request.get_json() or {}
    ticker = (data.get('ticker') or '').upper().strip()
    ai_model = data.get('model') or 'ollama'

    if not ticker:
        return jsonify({'error': 'ticker required'}), 400

    # ── Price context for the prompt ──
    try:
        hist = yf.Ticker(ticker).history(period='1mo')
        if not hist.empty:
            current_price = round(float(hist['Close'].iloc[-1]), 2)
            past_price    = round(float(hist['Close'].iloc[0]), 2)
            price_info = (
                f'The current price is ${current_price}. '
                f'One month ago, it was ${past_price}.'
            )
        else:
            price_info = ''
    except Exception:
        price_info = ''

    # ── Ask the LLM chain ──
    try:
        raw = _call_llm(ai_model, ticker, price_info)
    except Exception as e:
        return jsonify({'error': f'API Error: {e}'}), 502

    # ── Parse structured response (with fallbacks) ──
    parsed    = parse_llm_json(raw) or {}
    analysis  = parsed.get('analysis') or raw
    verdict   = parsed.get('verdict')

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
        'headline':   parsed.get('headline') or None,
        'key_points': parsed.get('key_points') if isinstance(parsed.get('key_points'), list) else None,
        'risks':      parsed.get('risks') if isinstance(parsed.get('risks'), list) else None,
    })


# ═══════════════════════════════════════════════════════════
# HEALTH CHECK
# ═══════════════════════════════════════════════════════════
@app.route('/health/llm', methods=['GET'])
def health_llm():
    """
    Probe both LLM providers with real requests so you know what's actually working.
    Ollama probe includes the ngrok header; otherwise it would always fail behind a tunnel.
    """
    results = {}

    # ── Ollama: check the server is reachable AND has our model ──
    try:
        r = requests.get(
            f"{OLLAMA_HOST}/api/tags",
            headers=_NGROK_HEADERS,
            timeout=5,
        )
        if r.status_code == 200:
            try:
                tags = r.json()
                models = [m.get('name', '') for m in tags.get('models', [])]
                has_model = any(OLLAMA_MODEL in m for m in models)
                results['ollama'] = {
                    'ok': has_model,
                    'host': OLLAMA_HOST,
                    'model_configured': OLLAMA_MODEL,
                    'models_available': models,
                    'note': 'model present' if has_model else 'model NOT pulled on the server',
                }
            except Exception as e:
                results['ollama'] = {
                    'ok': False, 'host': OLLAMA_HOST,
                    'error': f'Bad JSON response: {e}',
                    'raw': r.text[:200],
                }
        else:
            results['ollama'] = {
                'ok': False, 'host': OLLAMA_HOST,
                'status': r.status_code,
                'raw': r.text[:200],
            }
    except Exception as e:
        results['ollama'] = {'ok': False, 'host': OLLAMA_HOST, 'error': str(e)}

    # ── Groq: real 1-token test call ──
    key = os.environ.get('GROQ_API_KEY', '')
    if not key or key == 'missing':
        results['groq'] = {'ok': False, 'model_configured': GROQ_MODEL, 'error': 'GROQ_API_KEY not set'}
    else:
        try:
            resp = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{'role': 'user', 'content': 'ping'}],
                max_tokens=3,
                timeout=10,
            )
            results['groq'] = {
                'ok': True,
                'model_configured': GROQ_MODEL,
                'reply': (resp.choices[0].message.content or '').strip()[:40],
            }
        except Exception as e:
            results['groq'] = {
                'ok': False,
                'model_configured': GROQ_MODEL,
                'error': f'{type(e).__name__}: {e}',
            }

    return jsonify(results)


if __name__ == '__main__':
    print("─" * 60)
    print(f"  Ollama host  : {OLLAMA_HOST}")
    print(f"  Ollama model : {OLLAMA_MODEL}")
    print(f"  Groq model   : {GROQ_MODEL}  (fallback)")
    print("─" * 60)
    app.run(debug=True, port=5000)