"""
03_ablation_study.py
Ablation study LENGKAP: menjalankan pruning dengan tiga kriteria tunggal
pada SELURUH 7 rasio dengan epoch yang SAMA dengan baseline (30 epoch)
untuk perbandingan yang sepenuhnya setara.

Total eksperimen: 3 kriteria x 7 rasio = 21 eksperimen

Jalankan:
    python scripts/03_ablation_study.py
    python scripts/03_ablation_study.py --epochs 30
    python scripts/03_ablation_study.py --weight_ratio 0.3
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
    run_ablation_single_criterion, compute_optimal_weights
)
from src.visualize import print_results_table


def set_seed(seed=CFG.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_single_ablation(model_baseline, dataloaders, dataset_sizes,
                         criterion_name, ratio, device, epochs):
    """Jalankan satu skenario ablation: pruning + fine-tuning + evaluasi."""

    # Cek apakah checkpoint sudah ada (untuk resume)
    ckpt_name = f"ablation_{criterion_name}_{int(ratio*100)}pct_{epochs}ep.pth"
    ckpt_path = CFG.CHECKPOINT_DIR / ckpt_name

    if ckpt_path.exists():
        print(f"\n{'=' * 60}")
        print(f"SKIP: {criterion_name.upper()} | {ratio*100:.0f}% | {epochs} ep (sudah ada)")
        print(f"{'=' * 60}")
        # Muat hasil dari checkpoint
        pruned_model, ckpt = load_checkpoint(ckpt_path, device)
        metrics, _, _ = evaluate_model(pruned_model, dataloaders["test"], device)
        print_results_table(metrics, f"{criterion_name.upper()} | {ratio*100:.0f}%")
        return metrics

    print(f"\n{'=' * 60}")
    print(f"ABLATION: {criterion_name.upper()} saja | Rasio {ratio*100:.0f}% | {epochs} epoch")
    print(f"{'=' * 60}")

    # Pruning dengan kriteria tunggal
    pruned_model, masks = run_ablation_single_criterion(
        model_baseline, dataloaders["train"],
        ratio, criterion_name, device
    )

    # Fine-tuning dengan epoch yang sama dengan baseline
    pruned_model, history = train_model(
        model=pruned_model,
        dataloaders=dataloaders,
        dataset_sizes=dataset_sizes,
        num_epochs=epochs,
        lr=CFG.FINETUNE_LR,
        device=device,
        checkpoint_path=ckpt_path,
        phase_name=f"Ablation {criterion_name.upper()} {ratio*100:.0f}%"
    )

    # Evaluasi
    metrics, _, _ = evaluate_model(pruned_model, dataloaders["test"], device)
    print_results_table(metrics, f"{criterion_name.upper()} | {ratio*100:.0f}%")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Ablation Study Lengkap - Semua Rasio")
    parser.add_argument("--epochs", type=int, default=CFG.BASELINE_EPOCHS,
                        help="Jumlah epoch fine-tuning, sama dengan baseline (default: 30)")
    parser.add_argument("--weight_ratio", type=float, default=0.3,
                        help="Rasio acuan untuk menghitung bobot optimal (default: 0.3)")
    args = parser.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Selalu jalankan semua rasio
    ratios_to_test = CFG.PRUNING_RATIOS  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    total_experiments = len(ratios_to_test) * 3  # 3 kriteria per rasio

    print("=" * 60)
    print("TAHAP 3: ABLATION STUDY LENGKAP")
    print("=" * 60)
    print(f"  Device     : {device}")
    print(f"  Rasio      : {[f'{r*100:.0f}%' for r in ratios_to_test]}")
    print(f"  Epoch      : {args.epochs} (sama dengan baseline)")
    print(f"  Kriteria   : L1, BN Gamma, Entropi Shannon")
    print(f"  Total      : {total_experiments} eksperimen")
    print(f"  Rasio acuan bobot: {args.weight_ratio*100:.0f}%")

    # Muat DataLoader dan model baseline
    SPLIT_DIR = CFG.DATASET_DIR.parent / "dataset_split"
    dataloaders, dataset_sizes = get_dataloaders(SPLIT_DIR)

    BASELINE_CKPT = CFG.CHECKPOINT_DIR / "baseline_9class.pth"
    if not BASELINE_CKPT.exists():
        print("[ERROR] Checkpoint baseline tidak ditemukan!")
        print("Jalankan 02_train_baseline.py terlebih dahulu.")
        return

    model_baseline, _ = load_checkpoint(BASELINE_CKPT, device)

    # Jalankan ablation untuk SELURUH rasio dan SELURUH kriteria
    all_ablation_results = {}
    experiment_count = 0
    start_time = time.time()

    for ratio in ratios_to_test:
        ratio_key = f"{int(ratio * 100)}%"
        all_ablation_results[ratio_key] = {}

        for criterion_name in ["l1", "bn", "entropy"]:
            experiment_count += 1
            elapsed = time.time() - start_time
            print(f"\n>>> Eksperimen {experiment_count}/{total_experiments} "
                  f"| Waktu berjalan: {elapsed/60:.1f} menit")

            metrics = run_single_ablation(
                model_baseline, dataloaders, dataset_sizes,
                criterion_name, ratio, device, args.epochs
            )
            all_ablation_results[ratio_key][criterion_name] = {
                "accuracy": metrics["accuracy"],
                "f1_score": metrics["f1_score"],
                "num_params": metrics["num_params"],
                "model_size_mb": metrics["model_size_mb"],
                "flops": metrics["flops"],
                "inference_ms": metrics["inference_ms"],
            }

            # Simpan hasil setiap selesai satu eksperimen (untuk keamanan)
            save_data = {
                "ablation_results": all_ablation_results,
                "epochs_used": args.epochs,
                "status": f"{experiment_count}/{total_experiments} selesai",
            }
            save_path = CFG.OUTPUT_DIR / "ablation_results.json"
            with open(save_path, "w") as f:
                json.dump(save_data, f, indent=2)

    # Hitung bobot optimal dari rasio acuan
    acuan_key = f"{int(args.weight_ratio * 100)}%"
    if acuan_key in all_ablation_results:
        acc_l1 = all_ablation_results[acuan_key]["l1"]["accuracy"]
        acc_bn = all_ablation_results[acuan_key]["bn"]["accuracy"]
        acc_entropy = all_ablation_results[acuan_key]["entropy"]["accuracy"]

        w1, w2, w3 = compute_optimal_weights(acc_l1, acc_bn, acc_entropy)

        weights_data = {
            "w1_l1": w1,
            "w2_bn": w2,
            "w3_entropy": w3,
            "source_ratio": args.weight_ratio,
            "source_epochs": args.epochs,
            "acc_l1": acc_l1,
            "acc_bn": acc_bn,
            "acc_entropy": acc_entropy,
        }
    else:
        weights_data = {}

    # Cetak tabel ringkasan ablation lengkap
    total_time = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"RINGKASAN ABLATION STUDY LENGKAP ({args.epochs} epoch per eksperimen)")
    print(f"{'=' * 80}")
    print(f"{'Rasio':<8} {'L1 Acc':<12} {'BN Acc':<12} {'Entropi Acc':<12} {'Terbaik'}")
    print(f"{'-' * 60}")

    for ratio_key, criteria in all_ablation_results.items():
        accs = {k: v["accuracy"] for k, v in criteria.items()}
        best = max(accs, key=accs.get)
        marker = " <-- acuan bobot" if ratio_key == acuan_key else ""
        print(f"{ratio_key:<8} "
              f"{accs.get('l1', 0)*100:>8.2f}%   "
              f"{accs.get('bn', 0)*100:>8.2f}%   "
              f"{accs.get('entropy', 0)*100:>8.2f}%   "
              f"{best.upper()}{marker}")

    print(f"{'=' * 80}")
    print(f"Total waktu: {total_time/60:.1f} menit ({total_time/3600:.1f} jam)")

    if weights_data:
        print(f"\nBOBOT OPTIMAL (dari rasio {acuan_key}):")
        print(f"  w1 (L1)      = {weights_data['w1_l1']:.4f}")
        print(f"  w2 (BN)      = {weights_data['w2_bn']:.4f}")
        print(f"  w3 (Entropi) = {weights_data['w3_entropy']:.4f}")

    # Simpan hasil final lengkap
    save_data = {
        "ablation_results": all_ablation_results,
        "optimal_weights": weights_data,
        "epochs_used": args.epochs,
        "total_time_seconds": total_time,
        "status": "SELESAI",
    }
    save_path = CFG.OUTPUT_DIR / "ablation_results.json"
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n[INFO] Hasil disimpan: {save_path}")
    print(f"\n[SELESAI] Ablation study lengkap selesai.")
    print(f"Lanjutkan ke 04_pruning_multicriteria.py --load_weights")


if __name__ == "__main__":
    main()
