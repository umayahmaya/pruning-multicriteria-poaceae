"""
visualize.py - Visualisasi Hasil Eksperimen

Modul ini menangani:
1. Plot confusion matrix
2. Plot kurva training (loss dan akurasi)
3. Plot perbandingan metrik antar skenario
4. Tabel ringkasan hasil
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.config import CFG


def plot_confusion_matrix(cm, title="Confusion Matrix", save_path=None):
    """
    Menampilkan confusion matrix dengan anotasi.

    Args:
        cm: numpy array confusion matrix
        title: Judul plot
        save_path: Path untuk menyimpan gambar (opsional)
    """
    fig, ax = plt.subplots(figsize=(12, 10))

    # Buat label singkat supaya muat
    short_labels = [name.replace("_", "\n") for name in CFG.CLASS_NAMES]

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=short_labels,
        yticklabels=short_labels,
        ax=ax
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Confusion matrix disimpan: {save_path}")

    plt.show()


def plot_training_history(history, title="Training History", save_path=None):
    """
    Menampilkan kurva loss dan akurasi per epoch.

    Args:
        history: dict dengan keys "train_loss", "train_acc", "val_loss", "val_acc"
        title: Judul plot
        save_path: Path untuk menyimpan
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Plot Loss
    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=3)
    ax1.plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=3)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss per Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot Accuracy
    ax2.plot(epochs, history["train_acc"], "b-o", label="Train Acc", markersize=3)
    ax2.plot(epochs, history["val_acc"], "r-o", label="Val Acc", markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Accuracy per Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Training history disimpan: {save_path}")

    plt.show()


def plot_pruning_comparison(results_dict, metric="accuracy", save_path=None, show=True):
    """
    Menampilkan perbandingan metrik antar skenario pruning.

    Args:
        results_dict: {
            "Baseline": {"accuracy": 0.89, ...},
            "L1 saja": {"10%": {"accuracy": 0.88}, "20%": {...}, ...},
            "BN saja": {...},
            "Entropi saja": {...},
            "Multi-Kriteria": {...},
        }
        metric: Metrik yang ditampilkan
        save_path: Path untuk menyimpan
        show: Tampilkan jendela plot (set False untuk terminal non-interaktif)
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    ratios = [f"{int(r*100)}%" for r in CFG.PRUNING_RATIOS]
    colors = {
        "L1 saja": "#1f77b4",
        "BN saja": "#ff7f0e",
        "Entropi saja": "#2ca02c",
        "Multi-Kriteria": "#d62728",
    }

    for scenario_name, scenario_data in results_dict.items():
        if scenario_name == "Baseline":
            baseline_val = scenario_data.get(metric, 0)
            ax.axhline(y=baseline_val, color="black", linestyle="--",
                       label=f"Baseline ({baseline_val:.4f})", alpha=0.7)
            continue

        values = []
        for ratio in CFG.PRUNING_RATIOS:
            ratio_key = f"{int(ratio*100)}%"
            if ratio_key in scenario_data:
                values.append(scenario_data[ratio_key].get(metric, 0))
            else:
                values.append(None)

        color = colors.get(scenario_name, "gray")
        ax.plot(ratios, values, "-o", label=scenario_name,
                color=color, markersize=6, linewidth=2)

    ax.set_xlabel("Rasio Pemangkasan", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(f"Perbandingan {metric.title()} Antar Skenario Pruning",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Perbandingan disimpan: {save_path}")

    if show:
        plt.show()


def print_results_table(results, scenario_name=""):
    """
    Mencetak tabel hasil evaluasi yang rapi.

    Args:
        results: dict dengan metrik evaluasi
        scenario_name: Nama skenario untuk header
    """
    print(f"\n{'='*60}")
    if scenario_name:
        print(f"  {scenario_name}")
        print(f"{'='*60}")

    print(f"  {'Metrik':<20} {'Nilai':<20}")
    print(f"  {'-'*40}")
    print(f"  {'Akurasi':<20} {results.get('accuracy', 0)*100:.2f}%")
    print(f"  {'Precision':<20} {results.get('precision', 0):.4f}")
    print(f"  {'Recall':<20} {results.get('recall', 0):.4f}")
    print(f"  {'F1-Score':<20} {results.get('f1_score', 0):.4f}")
    print(f"  {'Parameter':<20} {results.get('num_params', 0):,}")
    print(f"  {'Ukuran (MB)':<20} {results.get('model_size_mb', 0):.2f}")
    print(f"  {'FLOPs (M)':<20} {results.get('flops', 0)/1e6:.3f}")
    print(f"  {'Inferensi (ms)':<20} {results.get('inference_ms', 0):.2f}")
    print(f"{'='*60}")
