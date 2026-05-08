import pandas as pd, os
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

OUTPUT_DIR = "data/sentiment_labeled"
MODEL_NAME = "ProsusAI/finbert"

def load_finbert():
    print("  Loading FinBERT (downloads ~440MB on first run)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    return pipeline("text-classification", model=model, tokenizer=tokenizer,
                    truncation=True, max_length=512, batch_size=32)

def run_finbert_sentiment():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    clf = load_finbert()

    sources = [
        ("data/raw/reddit_posts.csv",   "title",  "reddit_finbert.csv"),
        ("data/raw/twitter_posts.csv",  "text",   "twitter_finbert.csv"),
        ("data/raw/news_headlines.csv", "title",  "news_finbert.csv"),
    ]

    for in_path, text_col, out_file in sources:
        if not os.path.exists(in_path):
            print(f"[SKIP] {in_path} not found."); continue
        df = pd.read_csv(in_path)
        texts   = df[text_col].fillna("").astype(str).tolist()
        results = clf(texts)
        df["sentiment"]      = [r["label"].lower() for r in results]
        df["finbert_score"]  = [round(r["score"], 4) for r in results]
        out_path = os.path.join(OUTPUT_DIR, out_file)
        df.to_csv(out_path, index=False)
        print(f"[OK] {out_path} ({len(df)} rows)")

if __name__ == "__main__":
    run_finbert_sentiment()