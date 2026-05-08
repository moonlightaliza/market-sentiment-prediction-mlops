import feedparser
import pandas as pd
from datetime import datetime
import os

OUTPUT_PATH = "data/raw/news_headlines.csv"

RSS_FEEDS = {
    "reuters_business": "https://feeds.reuters.com/reuters/businessNews",
    "yahoo_finance":    "https://finance.yahoo.com/news/rssindex",
    "cnbc":             "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
    "marketwatch":      "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
}

def scrape_news():
    os.makedirs("data/raw", exist_ok=True)
    all_rows = []

    for name, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                published = datetime(*entry.published_parsed[:6]) if hasattr(entry, "published_parsed") and entry.published_parsed else datetime.utcnow()
                all_rows.append({
                    "timestamp": published,
                    "source":    name,
                    "title":     entry.get("title", "").strip(),
                    "summary":   entry.get("summary", "").strip(),
                    "url":       entry.get("link", ""),
                })
            print(f"  [OK] {name}: {len(feed.entries)} articles")
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")

    df = pd.DataFrame(all_rows).drop_duplicates(subset=["title"])
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"[OK] News saved -> {OUTPUT_PATH} ({len(df)} rows)")

if __name__ == "__main__":
    scrape_news()