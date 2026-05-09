# src/models/config.py

CONFIG = {
    # Data
    "data_path": "data/processed/timeseries_dataset.csv",
    "feature_columns": [
        "open", "high", "low", "close", "volume",
        "avg_sentiment", "sentiment_count",
        "price_change_pct", "volume_change_pct",
        "sentiment_lag_1", "sentiment_lag_2", "sentiment_lag_3",
        "sentiment_ma_3", "volume_ma_3", "price_ma_3",
        "price_vs_ma", "hl_spread_pct"
    ],
    "target_column": "target",
    "sequence_length": 30,       # look back 30 × 15-min windows = 7.5 hours
    "test_size": 0.2,

    # Model
    "input_size": 17,            # matches feature_columns above
    "hidden_size": 64,
    "num_layers": 2,
    "output_size": 1,            # binary classification
    "dropout": 0.2,

    # Training
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001,

    # MLflow
    "experiment_name": "market-sentiment-prediction",

    # Paths
    "model_save_path": "src/models/saved_models/",
}