"""
Twitter scraper — uses snscrape (no API key) or synthetic fallback.
Install: pip install snscrape
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os, random, subprocess, json

OUTPUT_PATH = "data/raw/twitter_posts.csv"
QUERIES = ["stock market lang:en", "NYSE lang:en", "investing finance lang:en"]

_TEMPLATES = [
    "Loaded up on {t} calls, this thing is going to moon 🚀",
    "{t} earnings beat — time to buy the dip",
    "Selling {t} before earnings, too risky",
    "Short {t} — chart is broken",
    "Watching {t} closely this week #stocks",
    "Anyone have a price target on {t}?",
]
_TICKERS = ["AAPL","TSLA","NVDA","AMZN","MSFT","SPY","META","AMD","GOOGL"]

def _synthetic(n=500):
    random.seed(99); np.random.seed(99)
    base = datetime.utcnow()
    rows = []
    for _ in range(n):
        t = random.choice(_TICKERS)
        rows.append({
            "timestamp": (base - timedelta(minutes=random.randint(1,43200))).strftime("%Y-%m-%d %H:%M:%S"),
            "text":      random.choice(_TEMPLATES).format(t=t),
            "likes":     int(np.random.exponential(50)),
            "retweets":  int(np.random.exponential(15)),
            "username":  f"trader_{random.randint(100,9999)}",
        })
    return pd.DataFrame(rows)

def scrape_twitter():
    os.makedirs("data/raw", exist_ok=True)
    all_rows = []

    for q in QUERIES:
        try:
            result = subprocess.run(
                ["snscrape","--jsonl","--max-results","100","twitter-search", q],
                capture_output=True, text=True, timeout=60)
            for line in result.stdout.strip().splitlines():
                obj = json.loads(line)
                all_rows.append({
                    "timestamp": obj.get("date",""),
                    "text":      obj.get("rawContent", obj.get("content","")),
                    "likes":     obj.get("likeCount",0),
                    "retweets":  obj.get("retweetCount",0),
                    "username":  obj.get("user",{}).get("username",""),
                })
        except Exception:
            pass

    if all_rows:
        df = pd.DataFrame(all_rows)
        source_tag = "snscrape"
    else:
        print("  [INFO] snscrape unavailable. Using synthetic Twitter data.")
        df = _synthetic(500)
        source_tag = "synthetic"

    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[OK] Twitter posts saved -> {OUTPUT_PATH} ({len(df)} rows, source={source_tag})")

if __name__ == "__main__":
    scrape_twitter()