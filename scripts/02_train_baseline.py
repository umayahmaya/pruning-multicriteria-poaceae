"""
02_train_baseline.py
Pelatihan model baseline MobileNetV2 pretrained ImageNet pada 9 kelas Poaceae

Jalankan:
    python scripts/02_train_baseline.py
    python scripts/02_train_baseline.py --epochs 50 --lr 0.0005
"""

import sys
import os
import argparse
import random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CFG
from src.dataset import get_dataloaders
from src.model import (
    create_mobilenetv2, train_model, evaluate_model,
    count_parameters, load_checkpoint
)
from src.visualize import plot_confusion_matrix, plot_training_history, print_results_table


def set_seed(seed=CFG.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    parser = argparse.ArgumentParser(description="Training Baseline MobileNetV2")
    parser.add_argument("--epochs", type=int, default=CFG.BASELINE_EPOCHS)
    parser.add_argument("--lr", type=float, default=CFG.BASELINE_LR)
    parser.add_argument("--resume", action="store_true",
                        help="Lanjutkan dari checkpoint yang sudah ada")
    args = parser.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("TAHAP 2: PELATIHAN MODEL BASELINE")
    print("=" * 60)
    print(f"  Device    : {device}")
    print(f"  Epochs    : {args.epochs}")
    print(f"  LR        : {args.lr}")
    print(f"  Batch     : {CFG.BATCH_SIZE}")
    print(f"  Kelas     : {CFG.NUM_CLASSES}")

    # Muat DataLoader
    SPLIT_DIR = CFG.DATASET_DIR.parent / "dataset_split"
    dataloaders, dataset_sizes = get_dataloaders(SPLIT_DIR)

    # Buat atau muat model
    BASELINE_CKPT = CFG.CHECKPOINT_DIR / "baseline_9class.pth"

    if BASELINE_CKPT.exists() and args.resume:
        print(f"\n[INFO] Memuat checkpoint: {BASELINE_CKPT}")
        model, ckpt = load_checkpoint(BASELINE_CKPT, device)
        history = ckpt.get("history", {})
    else:
        print(f"\n[INFO] Membuat model baru...")
        model = create_mobilenetv2(num_classes=CFG.NUM_CLASSES, pretrained=True)
        print(f"  Parameter trainable: {count_parameters(model):,}")

        # Training
        model, history = train_model(
            model=model,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            num_epochs=args.epochs,
            lr=args.lr,
            device=device,
            checkpoint_path=BASELINE_CKPT,
            phase_name="Baseline 9 Kelas Poaceae"
        )

    # Evaluasi pada test set
    print("\n" + "=" * 60)
    print("EVALUASI MODEL BASELINE PADA TEST SET")
    print("=" * 60)

    metrics, preds, labels = evaluate_model(model, dataloaders["test"], device)
    print_results_table(metrics, "Baseline MobileNetV2 (9 Kelas)")

    # Simpan visualisasi
    if history:
        plot_training_history(
            history,
            title="Training History - Baseline MobileNetV2 (9 Kelas Poaceae)",
            save_path=CFG.OUTPUT_DIR / "training_baseline.png"
        )

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        title=f"Confusion Matrix Baseline (Acc: {metrics['accuracy']*100:.2f}%)",
        save_path=CFG.OUTPUT_DIR / "cm_baseline_9class.png"
    )

    print(f"\n[SELESAI] Baseline selesai. Lanjutkan ke 03_ablation_study.py")


if __name__ == "__main__":
    main()
