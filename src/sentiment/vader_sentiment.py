import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

OUTPUT_DIR = "data/sentiment_labeled"

def get_label(compound):
    if compound >= 0.05:   return "positive"
    elif compound <= -0.05: return "negative"
    else:                   return "neutral"

def label_dataframe(df, text_col, analyzer):
    df = df.copy()
    df[text_col] = df[text_col].fillna("").astype(str)
    scores = df[text_col].apply(lambda t: analyzer.polarity_scores(t))
    df["vader_compound"] = scores.apply(lambda s: s["compound"])
    df["sentiment"]      = df["vader_compound"].apply(get_label)
    return df

def run_vader_sentiment():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    analyzer = SentimentIntensityAnalyzer()

    sources = [
        ("data/raw/reddit_posts.csv",   "title",  "reddit_sentiment.csv"),
        ("data/raw/twitter_posts.csv",  "text",   "twitter_sentiment.csv"),
        ("data/raw/news_headlines.csv", "title",  "news_sentiment.csv"),
    ]

    for in_path, text_col, out_file in sources:
        if not os.path.exists(in_path):
            print(f"[SKIP] {in_path} not found.")
            continue
        df = pd.read_csv(in_path)
        df = label_dataframe(df, text_col, analyzer)
        out_path = os.path.join(OUTPUT_DIR, out_file)
        df.to_csv(out_path, index=False)
        print(f"[OK] {out_path} ({len(df)} rows) {df['sentiment'].value_counts().to_dict()}")

if __name__ == "__main__":
    run_vader_sentiment()