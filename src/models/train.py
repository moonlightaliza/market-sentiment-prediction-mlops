# src/models/train.py

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import CONFIG
from rnn_model import RNNModel
from lstm_model import LSTMModel
from gru_model import GRUModel


# ════════════════════════════════════════════════════════════
# 1 — LOAD DATA
# ════════════════════════════════════════════════════════════

def load_data():
    print("📂 Loading dataset...")
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, CONFIG["data_path"])
    df = pd.read_csv(data_path)
    df = df.dropna()

    X = df[CONFIG["feature_columns"]].values
    y = df[CONFIG["target_column"]].values

    print(f"  ✅ Loaded {len(df):,} rows | {X.shape[1]} features")
    print(f"  UP: {int(y.sum())} | DOWN: {int(len(y)-y.sum())}")
    return X, y


# ════════════════════════════════════════════════════════════
# 2 — BUILD SEQUENCES
# ════════════════════════════════════════════════════════════

def build_sequences(X, y, seq_len):
    print(f"🔄 Building sequences (length={seq_len})...")
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])
    Xs = np.array(Xs)
    ys = np.array(ys)
    print(f"  ✅ Sequences shape: {Xs.shape}")
    return Xs, ys


# ════════════════════════════════════════════════════════════
# 3 — TRAIN ONE MODEL (with early stopping)
# ════════════════════════════════════════════════════════════

def train_model(model, model_name, train_loader, val_loader):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=CONFIG["learning_rate"]
    )

    best_val_loss    = float("inf")
    patience         = CONFIG["early_stopping_patience"]
    patience_counter = 0
    stopped_epoch    = CONFIG["epochs"]

    os.makedirs(CONFIG["model_save_path"], exist_ok=True)
    save_path = os.path.join(CONFIG["model_save_path"], f"{model_name}_best.pt")

    print(f"\n🚀 Training {model_name}...")

    epoch_bar = tqdm(range(CONFIG["epochs"]), desc=f"{model_name}", unit="epoch")

    for epoch in epoch_bar:

        # ── Train ────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch).squeeze()
            loss  = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── Validate ─────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        correct  = 0
        total    = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                preds     = model(X_batch).squeeze()
                loss      = criterion(preds, y_batch)
                val_loss += loss.item()
                predicted = (torch.sigmoid(preds) >= 0.5).float()
                correct  += (predicted == y_batch).sum().item()
                total    += y_batch.size(0)

        val_loss /= len(val_loader)
        val_acc   = correct / total * 100

        # ── Update progress bar ───────────────────────────────
        epoch_bar.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            "val_loss":   f"{val_loss:.4f}",
            "val_acc":    f"{val_acc:.1f}%",
            "patience":   f"{patience_counter}/{patience}"
        })

        # ── Save best & early stopping ────────────────────────
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                stopped_epoch = epoch + 1
                epoch_bar.close()
                print(f"  ⏹  Early stopping at epoch {stopped_epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    print(f"  ✅ Best val loss: {best_val_loss:.4f} → saved to {save_path}")
    return best_val_loss, stopped_epoch


# ════════════════════════════════════════════════════════════
# 4 — MAIN
# ════════════════════════════════════════════════════════════

def main():
    # ── Load & scale ─────────────────────────────────────────
    X, y = load_data()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # ── Build sequences ───────────────────────────────────────
    seq_len = CONFIG["sequence_length"]
    X_seq, y_seq = build_sequences(X, y, seq_len)

    # ── Train/val split (no shuffle — keep time order) ────────
    X_train, X_val, y_train, y_val = train_test_split(
        X_seq, y_seq, test_size=CONFIG["test_size"], shuffle=False
    )

    print(f"\n  Train samples: {len(X_train)} | Val samples: {len(X_val)}")

    # ── Convert to tensors ────────────────────────────────────
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_val   = torch.tensor(X_val,   dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_val   = torch.tensor(y_val,   dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=CONFIG["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=CONFIG["batch_size"], shuffle=False
    )

    # ── Train all three models ────────────────────────────────
    models = {
        "RNN":  RNNModel( CONFIG["input_size"], CONFIG["hidden_size"],
                          CONFIG["num_layers"], CONFIG["output_size"],
                          CONFIG["dropout"]),
        "LSTM": LSTMModel(CONFIG["input_size"], CONFIG["hidden_size"],
                          CONFIG["num_layers"], CONFIG["output_size"],
                          CONFIG["dropout"]),
        "GRU":  GRUModel( CONFIG["input_size"], CONFIG["hidden_size"],
                          CONFIG["num_layers"], CONFIG["output_size"],
                          CONFIG["dropout"]),
    }

    results = {}
    for name, model in models.items():
        best_loss, stopped_at = train_model(model, name, train_loader, val_loader)
        results[name] = {"loss": best_loss, "stopped_at": stopped_at}

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  TRAINING SUMMARY")
    print("=" * 55)
    best_model = min(results, key=lambda k: results[k]["loss"])
    for name, info in results.items():
        tag = " ← best" if name == best_model else ""
        print(f"  {name:<6}: val loss = {info['loss']:.4f} "
              f"| stopped at epoch {info['stopped_at']}{tag}")
    print(f"\n🏆 Best model: {best_model}")
    print(f"✅ All models saved to {CONFIG['model_save_path']}")


if __name__ == "__main__":
    main()