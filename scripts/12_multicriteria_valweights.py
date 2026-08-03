"""
12_multicriteria_valweights.py
Menjalankan ulang pruning multi-kriteria pada tujuh rasio (10% - 70%, 30
epoch) memakai bobot baru hasil recompute dari akurasi val (lihat
07_recompute_weights_val.py), bukan bobot lama yang diturunkan dari akurasi
test. Skor entropi memakai kalibrasi train yang sudah diperbaiki (lihat
compute_entropy_scores di src/pruning.py).

Bobot dimuat dari outputs/ablation_val_results.json (optimal_weights_val):
    w1 (L1)      = 0.3360824742268041
    w2 (BN)      = 0.3319587628865979
    w3 (Entropi) = 0.3319587628865979

Checkpoint lama (checkpoints/multicriteria_*pct_*ep.pth, dari bobot test
lama) TIDAK ditimpa -- checkpoint baru memakai akhiran _valweights. Hasil
metrik dan confusion matrix disimpan terpisah, bukan menimpa
outputs/multicriteria_results.json / outputs/cm_multicriteria_*.png.

Jalankan:
    venv/Scripts/python.exe scripts/12_multicriteria_valweights.py
"""

import sys
import os
import json
import argparse
import random
import time
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

VAL_WEIGHTS_PATH = CFG.OUTPUT_DIR / "ablation_val_results.json"
RESULTS_PATH = CFG.OUTPUT_DIR / "multicriteria_valweights_results.json"
SUFFIX = "valweights"


def set_seed(seed=CFG.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_val_weights():
    """Muat bobot hasil recompute dari akurasi val."""
    if not VAL_WEIGHTS_PATH.exists():
        print(f"[ERROR] {VAL_WEIGHTS_PATH} tidak ditemukan.")
        print("Jalankan 07_recompute_weights_val.py terlebih dahulu.")
        return None

    with open(VAL_WEIGHTS_PATH) as f:
        data = json.load(f)
    weights = data.get("optimal_weights_val") or {}
    if not all(k in weights for k in ("w1_l1", "w2_bn", "w3_entropy")):
        print(f"[ERROR] Kunci 'optimal_weights_val' tidak lengkap di {VAL_WEIGHTS_PATH}")
        return None

    return weights["w1_l1"], weights["w2_bn"], weights["w3_entropy"]


def run_single_ratio(model_baseline, importance_scores, dataloaders,
                      dataset_sizes, ratio, device, epochs):
    """Jalankan satu skenario: pruning multi-kriteria + fine-tuning + evaluasi."""

    ratio_key = f"{int(ratio * 100)}%"
    ckpt_name = f"multicriteria_{int(ratio*100)}pct_{epochs}ep_{SUFFIX}.pth"
    ckpt_path = CFG.CHECKPOINT_DIR / ckpt_name

    if ckpt_path.exists():
        print(f"\n{'=' * 60}")
        print(f"SKIP: {ratio_key} (checkpoint sudah ada: {ckpt_name})")
        print(f"{'=' * 60}")
        pruned_model, _ = load_checkpoint(ckpt_path, device)
        metrics, _, _ = evaluate_model(pruned_model, dataloaders["test"], device)
        print_results_table(metrics, f"Multi-Kriteria (val) {ratio_key}")
        return metrics

    print(f"\n{'=' * 60}")
    print(f"PRUNING MULTI-KRITERIA (bobot val): {ratio_key}")
    print(f"{'=' * 60}")

    masks = get_pruning_mask(importance_scores, ratio)
    pruned_model = apply_pruning(model_baseline, masks)

    pruned_model, history = train_model(
        model=pruned_model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        num_epochs=epochs,
        lr=CFG.FINETUNE_LR,
        device=device,
        checkpoint_path=ckpt_path,
        phase_name=f"Multi-Kriteria (val) {ratio_key}"
    )

    metrics, preds, labels = evaluate_model(pruned_model, dataloaders["test"], device)

    plot_confusion_matrix(
        metrics["confusion_matrix"],
        title=f"CM Multi-Kriteria (val) {ratio_key} (Acc: {metrics['accuracy']*100:.2f}%)",
        save_path=CFG.OUTPUT_DIR / f"cm_multicriteria_{int(ratio*100)}pct_{SUFFIX}.png"
    )

    print_results_table(metrics, f"Multi-Kriteria (val) {ratio_key}")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Ulang pruning multi-kriteria dengan bobot dari val"
    )
    parser.add_argument("--epochs", type=int, default=CFG.BASELINE_EPOCHS,
                        help="Jumlah epoch fine-tuning, sama dengan baseline (default: 30)")
    args = parser.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weights = load_val_weights()
    if weights is None:
        return
    w1, w2, w3 = weights

    ratios_to_test = CFG.PRUNING_RATIOS  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    print("=" * 60)
    print("ULANG PRUNING MULTI-KRITERIA -- BOBOT DARI VAL")
    print("=" * 60)
    print(f"  Device     : {device}")
    print(f"  Rasio      : {[f'{r*100:.0f}%' for r in ratios_to_test]}")
    print(f"  Epoch      : {args.epochs} (sama dengan baseline)")
    print(f"  Bobot      : w1={w1:.10f}, w2={w2:.10f}, w3={w3:.10f}")
    print(f"  Sum(w)     : {w1+w2+w3:.4f}")
    print(f"  Kalibrasi  : dataloaders['train'], 20 gambar/kelas, seed={CFG.SEED}")
    print(f"  Checkpoint : akhiran _{SUFFIX} (checkpoint lama tidak ditimpa)")
    print(f"  Hasil JSON : {RESULTS_PATH}")

    # Muat DataLoader dan model baseline
    SPLIT_DIR = CFG.DATASET_DIR.parent / "dataset_split"
    dataloaders, dataset_sizes = get_dataloaders(SPLIT_DIR)

    BASELINE_CKPT = CFG.CHECKPOINT_DIR / "baseline_9class.pth"
    if not BASELINE_CKPT.exists():
        print("[ERROR] Checkpoint baseline tidak ditemukan!")
        print("Jalankan 02_train_baseline.py terlebih dahulu.")
        return

    model_baseline, _ = load_checkpoint(BASELINE_CKPT, device)

    # Hitung tiga skor kepentingan channel (entropi dari train, sudah diperbaiki)
    print("\n[INFO] Menghitung tiga skor kepentingan channel...")
    l1_scores = compute_l1_scores(model_baseline)
    bn_scores = compute_bn_scores(model_baseline)
    entropy_scores = compute_entropy_scores(
        model_baseline, dataloaders["train"], device
    )

    importance_scores = compute_importance_scores(
        l1_scores, bn_scores, entropy_scores, w1=w1, w2=w2, w3=w3
    )

    # Evaluasi baseline untuk perbandingan (aman -- measure_flops sudah tidak
    # lagi mengontaminasi model_baseline secara in-place)
    print("\n[INFO] Evaluasi baseline sebagai titik acuan...")
    metrics_baseline, _, _ = evaluate_model(
        model_baseline, dataloaders["test"], device
    )

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

    start_time = time.time()

    for i, ratio in enumerate(ratios_to_test, 1):
        ratio_key = f"{int(ratio * 100)}%"
        elapsed = time.time() - start_time
        print(f"\n>>> Eksperimen {i}/{len(ratios_to_test)} "
              f"| Waktu berjalan: {elapsed/60:.1f} menit")

        metrics = run_single_ratio(
            model_baseline, importance_scores, dataloaders, dataset_sizes,
            ratio, device, args.epochs
        )

        all_results[ratio_key] = {
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
            "num_params": metrics["num_params"],
            "model_size_mb": metrics["model_size_mb"],
            "flops": metrics["flops"],
            "inference_ms": metrics["inference_ms"],
        }

        # Simpan progres setiap selesai satu rasio (untuk keamanan)
        save_data = {
            "weights_used": {"w1": w1, "w2": w2, "w3": w3},
            "weights_source": "val (07_recompute_weights_val.py)",
            "calibration_source": "train",
            "results": all_results,
            "epochs_used": args.epochs,
            "status": f"{i}/{len(ratios_to_test)} selesai",
        }
        with open(RESULTS_PATH, "w") as f:
            json.dump(save_data, f, indent=2)

    total_time = time.time() - start_time

    # Cetak tabel ringkasan
    print(f"\n{'=' * 80}")
    print(f"RINGKASAN PRUNING MULTI-KRITERIA (bobot val, {args.epochs} epoch)")
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
    print(f"Total waktu: {total_time/60:.1f} menit ({total_time/3600:.1f} jam)")

    pruned_results = {k: v for k, v in all_results.items() if k != "baseline"}
    if pruned_results:
        best_ratio = max(pruned_results, key=lambda k: pruned_results[k]["accuracy"])
        best_acc = pruned_results[best_ratio]["accuracy"]
        print(f"\n>> Rasio optimal: {best_ratio} (Akurasi: {best_acc*100:.2f}%)")

    # Simpan hasil final (TIDAK menimpa outputs/multicriteria_results.json)
    save_data = {
        "weights_used": {"w1": w1, "w2": w2, "w3": w3},
        "weights_source": "val (07_recompute_weights_val.py)",
        "calibration_source": "train",
        "results": {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                        for kk, vv in v.items()}
                    for k, v in all_results.items()},
        "epochs_used": args.epochs,
        "total_time_seconds": total_time,
        "status": "SELESAI",
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n[INFO] Hasil disimpan: {RESULTS_PATH}")
    print(f"[INFO] Checkpoint lama (bobot test) tidak tersentuh.")
    print(f"\n[SELESAI] Pruning multi-kriteria (bobot val) selesai.")


if __name__ == "__main__":
    main()
