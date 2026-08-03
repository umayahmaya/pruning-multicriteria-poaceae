"""
04_pruning_multicriteria.py
Pruning terstruktur berbasis skoring multi-kriteria pada tujuh rasio

Jalankan:
    python scripts/04_pruning_multicriteria.py --load_weights
    python scripts/04_pruning_multicriteria.py --load_weights --ratios 0.2 0.3 0.4
    python scripts/04_pruning_multicriteria.py --w1 0.35 --w2 0.34 --w3 0.31
"""

import sys
import os
import json
import argparse
import random
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CFG
from src.dataset import get_dataloaders
from src.model import train_model, evaluate_model, load_checkpoint
from src.pruning import (
    compute_l1_scores, compute_bn_scores, compute_entropy_scores,
    compute_importance_scores, get_pruning_mask, apply_pruning
)
from src.visualize import plot_confusion_matrix, print_results_table


def set_seed(seed=CFG.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Pruning Multi-Kriteria")
    parser.add_argument("--ratios", type=float, nargs="+",
                        default=CFG.PRUNING_RATIOS,
                        help="Rasio pruning yang diuji")
    parser.add_argument("--epochs", type=int, default=CFG.FINETUNE_EPOCHS,
                        help="Epoch fine-tuning (default: 10)")
    parser.add_argument("--w1", type=float, default=None, help="Bobot L1")
    parser.add_argument("--w2", type=float, default=None, help="Bobot BN")
    parser.add_argument("--w3", type=float, default=None, help="Bobot Entropi")
    parser.add_argument("--load_weights", action="store_true",
                        help="Muat bobot dari hasil ablation study")
    args = parser.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tentukan bobot -- sumber bobot wajib eksplisit, tidak ada fallback diam-diam
    if args.load_weights:
        ablation_path = CFG.OUTPUT_DIR / "ablation_results.json"
        if not ablation_path.exists():
            print(f"[ERROR] File ablation tidak ditemukan: {ablation_path}")
            print("Jalankan 03_ablation_study.py terlebih dahulu, atau tentukan "
                  "bobot manual dengan --w1 --w2 --w3.")
            return

        with open(ablation_path) as f:
            ablation_data = json.load(f)
        weights = ablation_data.get("optimal_weights") or {}
        if not all(k in weights for k in ("w1_l1", "w2_bn", "w3_entropy")):
            print(f"[ERROR] Kunci 'optimal_weights' tidak lengkap di {ablation_path}")
            print("Jalankan 03_ablation_study.py sampai selesai agar bobot optimal tersimpan.")
            return

        w1 = weights["w1_l1"]
        w2 = weights["w2_bn"]
        w3 = weights["w3_entropy"]
        print(f"[INFO] Bobot dimuat dari ablation study")
    elif args.w1 is not None and args.w2 is not None and args.w3 is not None:
        w1, w2, w3 = args.w1, args.w2, args.w3
    else:
        print("[ERROR] Sumber bobot harus ditentukan secara eksplisit.")
        print("Gunakan --load_weights untuk memuat dari ablation study, "
              "atau tentukan manual dengan --w1 --w2 --w3.")
        return

    print("=" * 60)
    print("TAHAP 4: PRUNING MULTI-KRITERIA")
    print("=" * 60)
    print(f"  Device  : {device}")
    print(f"  Rasio   : {[f'{r*100:.0f}%' for r in args.ratios]}")
    print(f"  Epoch   : {args.epochs}")
    print(f"  Bobot   : w1={w1:.4f}, w2={w2:.4f}, w3={w3:.4f}")
    print(f"  Sum(w)  : {w1+w2+w3:.4f}")

    # Muat DataLoader dan model baseline
    SPLIT_DIR = CFG.DATASET_DIR.parent / "dataset_split"
    dataloaders, dataset_sizes = get_dataloaders(SPLIT_DIR)

    BASELINE_CKPT = CFG.CHECKPOINT_DIR / "baseline_9class.pth"
    if not BASELINE_CKPT.exists():
        print("[ERROR] Checkpoint baseline tidak ditemukan!")
        print("Jalankan 02_train_baseline.py terlebih dahulu.")
        return

    model_baseline, _ = load_checkpoint(BASELINE_CKPT, device)

    # Hitung tiga skor kepentingan channel
    print("\n[INFO] Menghitung tiga skor kepentingan channel...")
    l1_scores = compute_l1_scores(model_baseline)
    bn_scores = compute_bn_scores(model_baseline)
    entropy_scores = compute_entropy_scores(
        model_baseline, dataloaders["train"], device
    )

    # Hitung skor gabungan
    importance_scores = compute_importance_scores(
        l1_scores, bn_scores, entropy_scores,
        w1=w1, w2=w2, w3=w3
    )

    # Evaluasi baseline untuk perbandingan
    print("\n[INFO] Evaluasi baseline sebagai titik acuan...")
    metrics_baseline, _, _ = evaluate_model(
        model_baseline, dataloaders["test"], device
    )

    # Jalankan pruning pada setiap rasio
    all_results = {
        "baseline": {
            "accuracy": metrics_baseline["accuracy"],
            "f1_score": metrics_baseline["f1_score"],
            "num_params": metrics_baseline["num_params"],
            "model_size_mb": metrics_baseline["model_size_mb"],
            "flops": metrics_baseline["flops"],
            "inference_ms": metrics_baseline["inference_ms"],
        }
    }

    for ratio in args.ratios:
        ratio_key = f"{int(ratio * 100)}%"
        print(f"\n{'=' * 60}")
        print(f"PRUNING MULTI-KRITERIA: {ratio_key}")
        print(f"{'=' * 60}")

        # Buat mask dan pangkas
        masks = get_pruning_mask(importance_scores, ratio)
        pruned_model = apply_pruning(model_baseline, masks)

        # Fine-tuning
        ckpt_name = f"multicriteria_{int(ratio*100)}pct_{args.epochs}ep.pth"
        pruned_model, history = train_model(
            model=pruned_model,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            num_epochs=args.epochs,
            lr=CFG.FINETUNE_LR,
            device=device,
            checkpoint_path=CFG.CHECKPOINT_DIR / ckpt_name,
            phase_name=f"Multi-Kriteria {ratio_key}"
        )

        # Evaluasi
        metrics, preds, labels = evaluate_model(
            pruned_model, dataloaders["test"], device
        )

        all_results[ratio_key] = {
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
            "num_params": metrics["num_params"],
            "model_size_mb": metrics["model_size_mb"],
            "flops": metrics["flops"],
            "inference_ms": metrics["inference_ms"],
        }

        # Confusion matrix
        plot_confusion_matrix(
            metrics["confusion_matrix"],
            title=f"CM Multi-Kriteria {ratio_key} (Acc: {metrics['accuracy']*100:.2f}%)",
            save_path=CFG.OUTPUT_DIR / f"cm_multicriteria_{int(ratio*100)}pct.png"
        )

        print_results_table(metrics, f"Multi-Kriteria {ratio_key}")

    # Cetak tabel ringkasan
    print(f"\n{'=' * 80}")
    print(f"RINGKASAN PRUNING MULTI-KRITERIA")
    print(f"{'=' * 80}")
    print(f"{'Rasio':<8} {'Akurasi':<10} {'F1':<8} {'Param':<12} "
          f"{'MB':<8} {'FLOPs(M)':<12} {'Inf(ms)':<10}")
    print(f"{'-' * 80}")

    for key, m in all_results.items():
        label = "Base" if key == "baseline" else key
        print(f"{label:<8} {m['accuracy']*100:>6.2f}%   "
              f"{m['f1_score']:.4f}  "
              f"{m['num_params']:>10,}  "
              f"{m['model_size_mb']:>6.2f}  "
              f"{m['flops']/1e6:>10.3f}  "
              f"{m['inference_ms']:>8.2f}")

    print(f"{'=' * 80}")

    # Identifikasi rasio optimal
    pruned_results = {k: v for k, v in all_results.items() if k != "baseline"}
    if pruned_results:
        best_ratio = max(pruned_results, key=lambda k: pruned_results[k]["accuracy"])
        best_acc = pruned_results[best_ratio]["accuracy"]
        print(f"\n>> Rasio optimal: {best_ratio} (Akurasi: {best_acc*100:.2f}%)")

    # Simpan hasil
    save_data = {
        "weights_used": {"w1": w1, "w2": w2, "w3": w3},
        "results": {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                        for kk, vv in v.items()}
                    for k, v in all_results.items()},
    }
    save_path = CFG.OUTPUT_DIR / "multicriteria_results.json"
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n[INFO] Hasil disimpan: {save_path}")
    print(f"\n[SELESAI] Pruning multi-kriteria selesai.")
    print(f"Lanjutkan ke 05_generate_report.py untuk laporan lengkap.")


if __name__ == "__main__":
    main()
