"""
Reddit scraper — uses Kaggle dataset or synthetic fallback.
Reddit API bans ML training use. Place Kaggle CSV at:
  data/raw/kaggle_reddit.csv
Download: https://www.kaggle.com/datasets/unanimad/reddit-rwallstreetbets
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os, random

OUTPUT_PATH   = "data/raw/reddit_posts.csv"
KAGGLE_SOURCE = "data/raw/kaggle_reddit.csv"

_TEMPLATES = [
    "Bullish on {t} after strong earnings report",
    "{t} looks ready to break out — loading up calls",
    "Just bought more {t}, fundamentals are solid",
    "{t} crashing, sell before it gets worse",
    "Bearish on {t} — macro headwinds are real",
    "Dumped all my {t} shares, this rally is fake",
    "What does everyone think about {t} earnings?",
    "Anyone following {t} closely? Thoughts?",
    "{t} trading sideways — waiting for a catalyst",
]
_TICKERS = ["AAPL","TSLA","GME","SPY","AMZN","MSFT","NVDA","AMC","PLTR","AMD"]

def _generate_synthetic(n=500):
    random.seed(42); np.random.seed(42)
    base = datetime.utcnow()
    rows = []
    for _ in range(n):
        t = random.choice(_TICKERS)
        rows.append({
            "timestamp":    (base - timedelta(hours=random.randint(1,2160))).strftime("%Y-%m-%d %H:%M:%S"),
            "subreddit":    random.choice(["wallstreetbets","stocks","investing"]),
            "title":        random.choice(_TEMPLATES).format(t=t),
            "score":        int(np.random.exponential(150)),
            "num_comments": int(np.random.exponential(40)),
        })
    return pd.DataFrame(rows)

def scrape_reddit():
    os.makedirs("data/raw", exist_ok=True)

    if os.path.exists(KAGGLE_SOURCE):
        print(f"  Loading Kaggle dataset from {KAGGLE_SOURCE}...")
        df = pd.read_csv(KAGGLE_SOURCE)
        df.columns = [c.lower().strip() for c in df.columns]
        source_tag = "kaggle"
    else:
        print("  [INFO] No Kaggle file found. Using synthetic data.")
        print("         Download real data: https://www.kaggle.com/datasets/unanimad/reddit-rwallstreetbets")
        print("         Save as: data/raw/kaggle_reddit.csv")
        df = _generate_synthetic(500)
        source_tag = "synthetic"

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[OK] Reddit posts saved -> {OUTPUT_PATH} ({len(df)} rows, source={source_tag})")

if __name__ == "__main__":
    scrape_reddit()