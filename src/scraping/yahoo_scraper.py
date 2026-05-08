import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "SPY"]
OUTPUT_PATH = "data/raw/yahoo_prices.csv"

def scrape_yahoo(tickers=TICKERS, period_days=90):
    os.makedirs("data/raw", exist_ok=True)
    end_date = datetime.today()
    start_date = end_date - timedelta(days=period_days)
    all_data = []

    for ticker in tickers:
        print(f"  Fetching {ticker}...")
        try:
            df = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"),
                           end=end_date.strftime("%Y-%m-%d"), progress=False, auto_adjust=True)
            if df.empty:
                continue
            df = df.reset_index()
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df["ticker"] = ticker
            df = df.rename(columns={"Date": "timestamp"})
            df = df[["timestamp","ticker","Open","High","Low","Close","Volume"]]
            df.columns = ["timestamp","ticker","open","high","low","close","volume"]
            all_data.append(df)
        except Exception as e:
            print(f"  [ERROR] {ticker}: {e}")

    if not all_data:
        print("[ERROR] No data collected.")
        return

    final_df = pd.concat(all_data, ignore_index=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"[OK] Yahoo data saved -> {OUTPUT_PATH} ({len(final_df)} rows)")

if __name__ == "__main__":
    scrape_yahoo()