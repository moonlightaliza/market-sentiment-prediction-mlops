# src/models/config.py

CONFIG = {
    # Data
    "data_path": "data/processed/timeseries_dataset.csv",
    "feature_columns": [
        "price_change_pct",
        "volume_change_pct",
        "sentiment_lag_1",
        "sentiment_lag_2",
        "sentiment_lag_3",
        "sentiment_ma_3",
        "volume_ma_3",
        "price_ma_3",
        "price_vs_ma",
        "hl_spread_pct",
        "avg_sentiment",
        "sentiment_count",
        "open",
        "high",
        "low",
        "close",
        "volume"
    ],
    "target_column": "target",
    "sequence_length": 10,       # 348 samples is small, keep seq short
    "test_size": 0.2,

    # Model
    "input_size": 17,
    "hidden_size": 64,
    "num_layers": 2,
    "output_size": 1,
    "dropout": 0.2,

    # Training
    "epochs": 50,
    "batch_size": 16,            # small dataset so small batch
    "learning_rate": 0.001,

    # MLflow
    "experiment_name": "market-sentiment-prediction",

    # Paths
    "model_save_path": "src/models/saved_models/",
}