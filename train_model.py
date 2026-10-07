import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import time
import requests
import io
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ═══════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════
PERIOD = '5y'
MIN_ROWS = 200
FORWARD_DAYS = 1
OUTPUT_MODEL = 'xgboost_model.pkl'
OUTPUT_DATA = 'sp500_processed_data.csv'

USER_AGENT = (
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
    'AppleWebKit/537.36 (KHTML, like Gecko) '
    'Chrome/120.0.0.0 Safari/537.36'
)

# Fallback list (top ~110 liquid tickers) used if Wikipedia is unreachable
FALLBACK_TICKERS = [
    'AAPL','MSFT','NVDA','AMZN','GOOGL','META','TSLA','BRK-B','UNH','XOM',
    'LLY','JPM','JNJ','V','PG','MA','AVGO','HD','MRK','COST','ABBV','CVX',
    'PEP','ADBE','KO','WMT','CRM','BAC','MCD','TMO','CSCO','ACN','ABT','LIN',
    'AMD','ORCL','NFLX','DIS','WFC','TXN','PM','DHR','INTU','CAT','VZ','NEE',
    'QCOM','IBM','GE','AMGN','NOW','UNP','UBER','SPGI','PFE','RTX','LOW','HON',
    'COP','BA','ELV','BKNG','ISRG','T','SBUX','PLD','MDT','GS','BLK','LMT',
    'AXP','ADP','SYK','TJX','MMC','DE','GILD','VRTX','ADI','MDLZ','REGN','CI',
    'CB','SCHW','MO','ZTS','SO','PGR','BSX','ETN','BMY','C','MU','CVS','INTC',
    'AMAT','LRCX','PANW','KLAC','SNPS','CDNS','ANET','MELI','PYPL','COIN',
]

FEATURE_COLS = ['Close', 'SMA_20', 'SMA_50', 'Volume', 'Daily_Return', 'RSI']


# ═══════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def make_features(df):
    """Same feature engineering as the live /predict/xgboost endpoint."""
    df = df.copy()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['Daily_Return'] = df['Close'].pct_change()
    df['RSI'] = calculate_rsi(df['Close'])
    # Target: will close be higher FORWARD_DAYS from now?
    df['Target'] = (df['Close'].shift(-FORWARD_DAYS) > df['Close']).astype(int)
    return df.dropna()


# ═══════════════════════════════════════════════════════════
# TICKER LIST
# ═══════════════════════════════════════════════════════════
def fetch_all_tickers():
    print('Fetching S&P 500 ticker list...')
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': USER_AGENT}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        r.raise_for_status()
        tables = pd.read_html(io.StringIO(r.text))
        tickers = tables[0]['Symbol'].tolist()
        # Wikipedia sometimes uses dots (BRK.B) — Yahoo uses dashes (BRK-B)
        tickers = [t.replace('.', '-') for t in tickers]
        print(f'  Got {len(tickers)} tickers from Wikipedia.')
        return tickers
    except Exception as e:
        print(f'  Wikipedia fetch failed ({e}).')
        print(f'  Falling back to hardcoded list of {len(FALLBACK_TICKERS)} tickers.')
        return FALLBACK_TICKERS


# ═══════════════════════════════════════════════════════════
# DATA DOWNLOAD
# ═══════════════════════════════════════════════════════════
def download_and_process(tickers):
    frames = []
    failed = []
    total = len(tickers)
    t_start = time.time()

    for i, t in enumerate(tickers, 1):
        try:
            df = yf.Ticker(t).history(period=PERIOD)
            if df is None or len(df) < MIN_ROWS:
                failed.append((t, 'too short'))
                continue

            df = make_features(df)
            if len(df) < MIN_ROWS:
                failed.append((t, 'too short after dropna'))
                continue

            df['ticker'] = t
            frames.append(df[FEATURE_COLS + ['Target']])

            if i % 25 == 0:
                rows_so_far = sum(len(f) for f in frames)
                elapsed = time.time() - t_start
                eta = (elapsed / i) * (total - i)
                print(f'  [{i}/{total}] {t:6s} ok — {rows_so_far:>7,} rows total '
                      f'({elapsed:.0f}s elapsed, ~{eta:.0f}s left)')

            time.sleep(0.05)   # be nice to Yahoo

        except Exception as e:
            failed.append((t, str(e)[:60]))

    print(f'\nDownloaded {len(frames)} tickers. Skipped {len(failed)}.')
    if failed:
        print('Skipped examples:', failed[:10])

    if not frames:
        raise RuntimeError('No tickers downloaded successfully. Check your network.')

    return pd.concat(frames, ignore_index=True)


# ═══════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════
def train(combined):
    X = combined[FEATURE_COLS]
    y = combined['Target']

    print(f'\nDataset: {len(X):,} rows × {len(FEATURE_COLS)} features')
    print(f'Class balance: {y.mean():.3f} up / {1 - y.mean():.3f} down')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f'Train: {len(X_train):,}   Test: {len(X_test):,}')

    # Handle class imbalance: weight the minority class
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1,
    )

    print('\nTraining XGBoost...')
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f'Training finished in {time.time() - t0:.1f}s')

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f'\nTest accuracy: {acc:.4f}')
    print('\nClassification report:')
    print(classification_report(y_test, preds, target_names=['Down', 'Up']))

    print('Confusion matrix:')
    print(confusion_matrix(y_test, preds))

    return model


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    t0 = time.time()

    tickers = fetch_all_tickers()
    print(f'\nTraining on {len(tickers)} tickers...')

    combined = download_and_process(tickers)
    combined.to_csv(OUTPUT_DATA, index=False)
    print(f'Saved dataset → {OUTPUT_DATA}')

    model = train(combined)

    with open(OUTPUT_MODEL, 'wb') as f:
        pickle.dump(model, f)
    print(f'Saved model → {OUTPUT_MODEL}')

    print(f'\nTotal elapsed: {time.time() - t0:.0f}s')
    print('Restart Flask to pick up the new model.')