"""
Market Sentiment Prediction Pipeline DAG
=========================================
Orchestrates the full pipeline:
  1. Data Ingestion (scraping)
  2. Sentiment Labeling (VADER)
  3. Time-Series Construction
  4. Model Training (RNN, LSTM, GRU)
  5. Model Evaluation
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.operators.python import PythonOperator

# ── Default arguments ────────────────────────────────────────────────────────
default_args = {
    "owner": "moonlightaliza",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

# ── Project root inside the container (matches docker-compose volume mount) ──
PROJECT_ROOT = "/opt/airflow/project"
SRC = f"{PROJECT_ROOT}/src"
PYTHON = "python"

# ── DAG definition ────────────────────────────────────────────────────────────
with DAG(
    dag_id="market_sentiment_pipeline",
    default_args=default_args,
    description="End-to-end market sentiment prediction pipeline",
    schedule="@daily",                     # runs once a day; change to None to trigger manually
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["market", "sentiment", "mlops"],
) as dag:

    # ── Stage 1: Data Ingestion ───────────────────────────────────────────────
    ingest_yahoo = BashOperator(
        task_id="ingest_yahoo",
        bash_command=f"{PYTHON} {SRC}/scraping/yahoo_scraper.py",
    )

    ingest_news = BashOperator(
        task_id="ingest_news",
        bash_command=f"{PYTHON} {SRC}/scraping/news_scraper.py",
    )

    ingest_reddit = BashOperator(
        task_id="ingest_reddit",
        bash_command=f"{PYTHON} {SRC}/scraping/reddit_scraper.py",
    )

    ingest_twitter = BashOperator(
        task_id="ingest_twitter",
        bash_command=f"{PYTHON} {SRC}/scraping/twitter_scraper.py",
    )

    # ── Stage 2: Sentiment Labeling ───────────────────────────────────────────
    # VADER is fast and rule-based; used as primary sentiment labeler
    sentiment_vader = BashOperator(
        task_id="sentiment_vader",
        bash_command=f"{PYTHON} {SRC}/sentiment/vader_sentiment.py",
    )

    # ── Stage 3: Time-Series Construction ────────────────────────────────────
    build_timeseries = BashOperator(
        task_id="build_timeseries",
        bash_command=f"{PYTHON} {SRC}/timeseries/timeseries_construction.py",
    )

    # ── Stage 4: Model Training ───────────────────────────────────────────────
    train_models = BashOperator(
        task_id="train_models",
        bash_command=f"{PYTHON} {SRC}/models/train.py",
    )

    # ── Stage 5: Evaluation ───────────────────────────────────────────────────
    evaluate_models = BashOperator(
        task_id="evaluate_models",
        bash_command=f"{PYTHON} {SRC}/models/evaluate.py",
    )

    # ── Pipeline dependencies ─────────────────────────────────────────────────
    # All scrapers run in parallel → sentiment → timeseries → train → evaluate
    [ingest_yahoo, ingest_news, ingest_reddit, ingest_twitter] >> sentiment_vader
    sentiment_vader >> build_timeseries
    build_timeseries >> train_models
    train_models >> evaluate_models