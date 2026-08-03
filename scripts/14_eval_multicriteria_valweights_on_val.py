"""
14_eval_multicriteria_valweights_on_val.py
Memuat 7 checkpoint multicriteria_{rasio}pct_30ep_valweights.pth (hasil
pruning multi-kriteria dengan bobot dari val, lihat
12_multicriteria_valweights.py) dan mengevaluasi masing-masing pada
dataloaders["val"] -- bukan test, supaya konsisten dengan sumber bobot yang
dipakai untuk memilih channel-nya.

Skrip ini TIDAK melatih ulang model apa pun.

Jalankan:
    venv/Scripts/python.exe scripts/14_eval_multicriteria_valweights_on_val.py
"""

import sys
import os
import json

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CFG
from src.dataset import get_dataloaders
from src.model import evaluate_model, load_checkpoint
from src.visualize import print_results_table

RESULTS_PATH = CFG.OUTPUT_DIR / "multicriteria_valweights_val_results.json"

RATIOS = [10, 20, 30, 40, 50, 60, 70]
CHECKPOINTS = [
    (f"{r}%", CFG.CHECKPOINT_DIR / f"multicriteria_{r}pct_30ep_valweights.pth")
    for r in RATIOS
]


def main():
    device = torch.device("cpu")

    print("=" * 60)
    print("EVALUASI MULTICRITERIA (bobot val) PADA DATALOADERS['val']")
    print("=" * 60)

    missing = [str(p) for _, p in CHECKPOINTS if not p.exists()]
    if missing:
        print("\n[ERROR] Checkpoint berikut tidak ditemukan:")
        for m in missing:
            print(f"  - {m}")
        print("Jalankan 12_multicriteria_valweights.py terlebih dahulu.")
        return

    SPLIT_DIR = CFG.DATASET_DIR.parent / "dataset_split"
    dataloaders, _ = get_dataloaders(SPLIT_DIR)

    results = {}
    for i, (ratio_key, ckpt_path) in enumerate(CHECKPOINTS, 1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(CHECKPOINTS)}] EVALUASI VAL: {ratio_key} | {ckpt_path.name}")
        print(f"{'=' * 60}")

        model, _ = load_checkpoint(ckpt_path, device)
        metrics, _, _ = evaluate_model(model, dataloaders["val"], device)
        print_results_table(metrics, f"Multi-Kriteria (val weights) {ratio_key}")

        results[ratio_key] = {
            "checkpoint": str(ckpt_path),
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
            "num_params": metrics["num_params"],
            "model_size_mb": metrics["model_size_mb"],
            "flops": metrics["flops"],
            "inference_ms": metrics["inference_ms"],
        }

        # Simpan progres setiap selesai satu checkpoint (untuk keamanan)
        with open(RESULTS_PATH, "w") as f:
            json.dump({
                "eval_split": "val",
                "results": results,
                "status": f"{i}/{len(CHECKPOINTS)} selesai",
            }, f, indent=2)

        del model

    # Cetak tabel ringkasan
    print(f"\n{'=' * 60}")
    print("RINGKASAN AKURASI VAL: MULTI-KRITERIA (bobot val)")
    print(f"{'=' * 60}")
    print(f"{'Rasio':<8} {'Akurasi':<10} {'F1':<8}")
    print(f"{'-' * 30}")
    for ratio_key, r in results.items():
        print(f"{ratio_key:<8} {r['accuracy']*100:>6.2f}%   {r['f1_score']:.4f}")
    print(f"{'=' * 60}")

    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "eval_split": "val",
            "results": results,
            "status": "SELESAI",
        }, f, indent=2)
    print(f"\n[INFO] Hasil disimpan: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
