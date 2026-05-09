# src/models/evaluate.py

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix,
    roc_curve, auc
)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import CONFIG
from rnn_model import RNNModel
from lstm_model import LSTMModel
from gru_model import GRUModel


# ════════════════════════════════════════════════════════════
# 1 — LOAD & PREPARE DATA
# ════════════════════════════════════════════════════════════

def load_and_prepare():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, CONFIG["data_path"])
    df = pd.read_csv(data_path).dropna()

    X = df[CONFIG["feature_columns"]].values
    y = df[CONFIG["target_column"]].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    seq_len = CONFIG["sequence_length"]
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i + seq_len])
        ys.append(y[i + seq_len])

    X_seq = np.array(Xs)
    y_seq = np.array(ys)

    _, X_val, _, y_val = train_test_split(
        X_seq, y_seq, test_size=CONFIG["test_size"], shuffle=False
    )

    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=CONFIG["batch_size"], shuffle=False
    )
    return val_loader, y_val.numpy()


# ════════════════════════════════════════════════════════════
# 2 — EVALUATE ONE MODEL
# ════════════════════════════════════════════════════════════

def evaluate_model(model, model_name, val_loader, y_true):
    model_path = os.path.join(CONFIG["model_save_path"], f"{model_name}_best.pt")

    if not os.path.exists(model_path):
        print(f"  ⚠️  {model_name}: no saved model found at {model_path}")
        return None

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    all_preds  = []
    all_probs  = []

    with torch.no_grad():
        for X_batch, _ in val_loader:
            logits = model(X_batch).squeeze()
            probs  = torch.sigmoid(logits)
            preds  = (probs >= 0.5).float()
            all_preds.extend(preds.numpy())
            all_probs.extend(probs.numpy())

    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc  = accuracy_score(y_true,  y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true,    y_pred, zero_division=0)
    f1   = f1_score(y_true,        y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    print(f"\n📊 {model_name} Results:")
    print(f"   Accuracy  : {acc*100:.2f}%")
    print(f"   Precision : {prec*100:.2f}%")
    print(f"   Recall    : {rec*100:.2f}%")
    print(f"   F1 Score  : {f1*100:.2f}%")
    print(f"   ROC AUC   : {roc_auc:.4f}")
    print(f"   Confusion Matrix:\n{cm}")

    return {
        "model":   model_name,
        "accuracy": acc,
        "precision": prec,
        "recall":  rec,
        "f1":      f1,
        "roc_auc": roc_auc,
        "cm":      cm,
        "fpr":     fpr,
        "tpr":     tpr,
        "y_pred":  y_pred,
        "y_prob":  y_prob,
    }


# ════════════════════════════════════════════════════════════
# 3 — PLOT ALL RESULTS
# ════════════════════════════════════════════════════════════

def plot_results(results):
    os.makedirs(CONFIG["model_save_path"], exist_ok=True)
    save_path = os.path.join(CONFIG["model_save_path"], "evaluation_plots.png")

    names   = [r["model"]    for r in results]
    colors  = ["steelblue", "darkorange", "mediumseagreen"]

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle("Model Evaluation — RNN vs LSTM vs GRU",
                 fontsize=16, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Plot 1: Accuracy / Precision / Recall / F1 bar chart ─────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    metrics     = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(metrics))
    width = 0.25

    for i, r in enumerate(results):
        vals = [r[m] * 100 for m in metrics]
        bars = ax1.bar(x + i * width, vals, width, label=r["model"],
                       color=colors[i], alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=7.5)

    ax1.set_xticks(x + width)
    ax1.set_xticklabels(metric_labels)
    ax1.set_ylabel("Score (%)")
    ax1.set_title("Metrics Comparison")
    ax1.set_ylim(0, 115)
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # ── Plot 2: ROC AUC bar ───────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    auc_vals = [r["roc_auc"] for r in results]
    bars = ax2.bar(names, auc_vals, color=colors, alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, auc_vals):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax2.set_title("ROC AUC Score")
    ax2.set_ylabel("AUC")
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    # ── Plot 3-5: Confusion matrices ──────────────────────────────────────────
    for i, r in enumerate(results):
        ax = fig.add_subplot(gs[1, i])
        cm = r["cm"]
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.set_title(f"{r['model']} — Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["DOWN", "UP"])
        ax.set_yticklabels(["DOWN", "UP"])
        for row in range(2):
            for col in range(2):
                ax.text(col, row, str(cm[row, col]),
                        ha="center", va="center",
                        color="white" if cm[row, col] > cm.max() / 2 else "black",
                        fontsize=14, fontweight="bold")

    # ── Plot 6: ROC curves ────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, :2])
    for i, r in enumerate(results):
        ax6.plot(r["fpr"], r["tpr"], color=colors[i], linewidth=2,
                 label=f"{r['model']} (AUC = {r['roc_auc']:.3f})")
    ax6.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax6.set_xlabel("False Positive Rate")
    ax6.set_ylabel("True Positive Rate")
    ax6.set_title("ROC Curves")
    ax6.legend(loc="lower right")
    ax6.grid(linestyle="--", alpha=0.4)

    # ── Plot 7: F1 score comparison ───────────────────────────────────────────
    ax7 = fig.add_subplot(gs[2, 2])
    f1_vals = [r["f1"] * 100 for r in results]
    bars = ax7.bar(names, f1_vals, color=colors, alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, f1_vals):
        ax7.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    ax7.set_title("F1 Score Comparison")
    ax7.set_ylabel("F1 (%)")
    ax7.set_ylim(0, 115)
    ax7.grid(axis="y", linestyle="--", alpha=0.4)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n✅ Plots saved to {save_path}")


# ════════════════════════════════════════════════════════════
# 4 — MAIN
# ════════════════════════════════════════════════════════════

def main():
    print("🔍 Evaluating all models on validation set...\n")
    val_loader, y_true = load_and_prepare()

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

    results = []
    for name, model in models.items():
        result = evaluate_model(model, name, val_loader, y_true)
        if result:
            results.append(result)

    # ── Plots ─────────────────────────────────────────────────
    if results:
        plot_results(results)

    # ── Final comparison table ────────────────────────────────
    if results:
        print("\n" + "=" * 60)
        print("  FINAL COMPARISON")
        print("=" * 60)
        print(f"  {'Model':<8} {'Accuracy':>10} {'Precision':>10} "
              f"{'Recall':>8} {'F1':>8} {'AUC':>8}")
        print("  " + "-" * 56)
        best = max(results, key=lambda x: x["f1"])
        for r in results:
            tag = " ←" if r["model"] == best["model"] else ""
            print(f"  {r['model']:<8} "
                  f"{r['accuracy']*100:>9.2f}% "
                  f"{r['precision']*100:>9.2f}% "
                  f"{r['recall']*100:>8.2f}% "
                  f"{r['f1']*100:>7.2f}% "
                  f"{r['roc_auc']:>8.4f}{tag}")
        print(f"\n🏆 Best model by F1: {best['model']}")


if __name__ == "__main__":
    main()