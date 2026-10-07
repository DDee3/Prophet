import pandas as pd
import requests
import io

url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
header = {
  "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

r = requests.get(url, headers=header)
tables = pd.read_html(io.StringIO(r.text))
tickers = tables[0]['Symbol'].tolist()

print(f"Total Tickers: {len(tickers)}")
print(f"First 10 Tickers: {tickers[:10]}")