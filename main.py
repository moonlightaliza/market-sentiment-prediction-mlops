
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from scraping.yahoo_scraper    import scrape_yahoo
from scraping.reddit_scraper   import scrape_reddit
from scraping.twitter_scraper  import scrape_twitter
from scraping.news_scraper     import scrape_news
from sentiment.vader_sentiment import run_vader_sentiment

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "mlops/mlflow"))
from mlflow_tracker import log_sentiment_run

def main():
    print("\n=== STAGE 1: Scraping ===")
    scrape_yahoo()
    scrape_reddit()
    scrape_twitter()
    scrape_news()

    print("\n=== STAGE 2: Sentiment Labeling ===")
    run_vader_sentiment()

    print("\n=== STAGE 3: MLflow Tracking ===")
    log_sentiment_run()

    print("\n=== Pipeline complete ===")

if __name__ == "__main__":
    main()

