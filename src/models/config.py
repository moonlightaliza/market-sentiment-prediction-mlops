# src/models/config.py

CONFIG = {
    # Data
    "sequence_length": 30,       # how many days to look back
    "target_column": "close",    # what we're predicting
    "test_size": 0.2,

    # Model
    "input_size": 6,             # OHLCV + sentiment score
    "hidden_size": 64,
    "num_layers": 2,
    "output_size": 1,
    "dropout": 0.2,

    # Training
    "epochs": 50,
    "batch_size": 32,
    "learning_rate": 0.001,

    # MLflow
    "experiment_name": "market-sentiment-prediction",

    # Paths
    "data_path": "src/timeseries/processed_timeseries.csv",
    "model_save_path": "src/models/saved_models/",
}