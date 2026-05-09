import mlflow
import pandas as pd
import os

EXPERIMENT_NAME = "market-sentiment-analysis"
SENTIMENT_DIR   = "data/sentiment_labeled"

SOURCES = {
    "reddit":  "reddit_sentiment.csv",
    "twitter": "twitter_sentiment.csv",
    "news":    "news_sentiment.csv",
}

def compute_metrics(df, model_name, source):
    total = len(df)
    counts = df["sentiment"].value_counts()
    return {
        f"{model_name}_{source}_total_rows":     total,
        f"{model_name}_{source}_positive_count": int(counts.get("positive", 0)),
        f"{model_name}_{source}_negative_count": int(counts.get("negative", 0)),
        f"{model_name}_{source}_neutral_count":  int(counts.get("neutral",  0)),
        f"{model_name}_{source}_positive_pct":   round(counts.get("positive", 0) / total * 100, 2),
        f"{model_name}_{source}_negative_pct":   round(counts.get("negative", 0) / total * 100, 2),
        f"{model_name}_{source}_neutral_pct":    round(counts.get("neutral",  0) / total * 100, 2),
        f"{model_name}_{source}_avg_score":      round(df["vader_compound"].mean(), 4) if "vader_compound" in df.columns else round(df["finbert_score"].mean(), 4),
    }

def log_sentiment_run():
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="sentiment-labeling"):

        mlflow.set_tags({
            "stage":   "sentiment",
            "models":  "vader,finbert",
            "sources": "reddit,twitter,news",
        })

        all_metrics = {}

        # --- VADER ---
        for source, filename in SOURCES.items():
            path = os.path.join(SENTIMENT_DIR, filename)
            if not os.path.exists(path):
                print(f"[SKIP] {path} not found"); continue
            df = pd.read_csv(path)
            metrics = compute_metrics(df, "vader", source)
            all_metrics.update(metrics)
            mlflow.log_artifact(path, artifact_path=f"vader/{source}")
            print(f"[OK] VADER {source}: {metrics}")

        # --- FinBERT ---
        finbert_files = {
            "reddit":  "reddit_finbert.csv",
            "twitter": "twitter_finbert.csv",
            "news":    "news_finbert.csv",
        }
        for source, filename in finbert_files.items():
            path = os.path.join(SENTIMENT_DIR, filename)
            if not os.path.exists(path):
                print(f"[SKIP] {path} not found (FinBERT not run yet)"); continue
            df = pd.read_csv(path)
            metrics = compute_metrics(df, "finbert", source)
            all_metrics.update(metrics)
            mlflow.log_artifact(path, artifact_path=f"finbert/{source}")
            print(f"[OK] FinBERT {source}: {metrics}")

        mlflow.log_metrics(all_metrics)
        print("\n[MLflow] Run logged successfully.")

