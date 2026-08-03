"""
07_recompute_weights_val.py
Menghitung ulang bobot w1, w2, w3 menggunakan akurasi VALIDASI, bukan akurasi
TEST seperti pada 03_ablation_study.py.

Latar belakang: aturan metodologis proyek (lihat CLAUDE.md butir 4.1)
mewajibkan bobot ablation diturunkan dari akurasi data validasi, bukan data
uji, karena menurunkan bobot dari data uji lalu mengevaluasi model akhir pada
data uji yang sama adalah kebocoran data dan penalaran melingkar.

Skrip ini TIDAK melatih ulang model. Ia hanya memuat 21 checkpoint hasil
ablation study yang sudah ada (3 kriteria x 7 rasio) dan mengevaluasinya
ulang pada dataloaders["val"].

Untuk kriteria entropi, checkpoint yang dimuat adalah hasil kalibrasi train
yang sudah diperbaiki (ablation_entropy_{rasio}pct_30ep_traincal.pth, dari
11_ablation_entropy_traincal.py), bukan checkpoint entropi lama yang
dikalibrasi dari val. Kriteria l1 dan bn tetap memuat checkpoint lama
(ablation_l1_{rasio}pct_30ep.pth, ablation_bn_{rasio}pct_30ep.pth) karena
keduanya data-free dan tidak terpengaruh masalah kalibrasi.

Skrip ini TIDAK menimpa outputs/ablation_results.json (hasil test lama).
Hasil evaluasi val disimpan terpisah di outputs/ablation_val_results.json.

Jalankan:
    python scripts/07_recompute_weights_val.py
    python scripts/07_recompute_weights_val.py --weight_ratio 0.3
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
from src.model import evaluate_model, load_checkpoint
from src.pruning import compute_optimal_weights

CRITERIA = ["l1", "bn", "entropy"]
CRITERIA_LABELS = {"l1": "L1", "bn": "BN", "entropy": "Entropi"}

OLD_TEST_RESULTS_PATH = CFG.OUTPUT_DIR / "ablation_results.json"
VAL_RESULTS_PATH = CFG.OUTPUT_DIR / "ablation_val_results.json"


def set_seed(seed=CFG.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_old_test_results():
    """Muat hasil ablation lama (akurasi test) untuk tabel perbandingan."""
    if not OLD_TEST_RESULTS_PATH.exists():
        print(f"[ERROR] {OLD_TEST_RESULTS_PATH} tidak ditemukan.")
        print("Jalankan 03_ablation_study.py terlebih dahulu.")
        return None
    with open(OLD_TEST_RESULTS_PATH) as f:
        return json.load(f)


def evaluate_all_checkpoints(dataloaders, device, epochs):
    """
    Muat 21 checkpoint ablation (3 kriteria x 7 rasio) dan evaluasi
    masing-masing pada dataloaders["val"].

    Returns:
        dict: {rasio_key: {kriteria: metrics}}
    """
    val_results = {}
    found = 0
    missing = []

    for ratio in CFG.PRUNING_RATIOS:
        ratio_key = f"{int(ratio * 100)}%"
        val_results[ratio_key] = {}

        for criterion_name in CRITERIA:
            if criterion_name == "entropy":
                # Kalibrasi train yang sudah diperbaiki (lihat
                # 11_ablation_entropy_traincal.py), bukan checkpoint lama
                # yang dikalibrasi dari val.
                ckpt_name = f"ablation_entropy_{int(ratio*100)}pct_{epochs}ep_traincal.pth"
            else:
                ckpt_name = f"ablation_{criterion_name}_{int(ratio*100)}pct_{epochs}ep.pth"
            ckpt_path = CFG.CHECKPOINT_DIR / ckpt_name

            if not ckpt_path.exists():
                print(f"[PERINGATAN] Checkpoint tidak ditemukan, dilewati: {ckpt_name}")
                missing.append(ckpt_name)
                continue

            print(f"\n{'=' * 60}")
            print(f"EVALUASI VAL: {criterion_name.upper()} | {ratio*100:.0f}% | {ckpt_name}")
            print(f"{'=' * 60}")

            model, _ = load_checkpoint(ckpt_path, device)
            metrics, _, _ = evaluate_model(model, dataloaders["val"], device)

            val_results[ratio_key][criterion_name] = {
                "accuracy": metrics["accuracy"],
                "f1_score": metrics["f1_score"],
                "num_params": metrics["num_params"],
                "model_size_mb": metrics["model_size_mb"],
                "flops": metrics["flops"],
                "inference_ms": metrics["inference_ms"],
            }
            found += 1

            # Simpan progres setiap selesai satu checkpoint (untuk keamanan)
            save_data = {
                "ablation_val_results": val_results,
                "epochs_used": epochs,
                "status": f"{found}/21 selesai",
            }
            with open(VAL_RESULTS_PATH, "w") as f:
                json.dump(save_data, f, indent=2)

    print(f"\n[INFO] Checkpoint berhasil dievaluasi: {found}/21")
    if missing:
        print(f"[PERINGATAN] {len(missing)} checkpoint hilang:")
        for name in missing:
            print(f"  - {name}")

    return val_results


def print_val_vs_test_table(val_results, old_test_results):
    """Cetak tabel perbandingan akurasi val vs test per kriteria dan rasio."""
    old_ablation = old_test_results.get("ablation_results", {}) if old_test_results else {}

    print(f"\n{'=' * 80}")
    print("PERBANDINGAN AKURASI: VALIDASI vs TEST")
    print(f"{'=' * 80}")
    print(f"{'Rasio':<8} {'Kriteria':<10} {'Val Acc':<12} {'Test Acc':<12} {'Selisih (Val-Test)'}")
    print(f"{'-' * 80}")

    for ratio in CFG.PRUNING_RATIOS:
        ratio_key = f"{int(ratio * 100)}%"
        for criterion_name in CRITERIA:
            val_entry = val_results.get(ratio_key, {}).get(criterion_name)
            test_entry = old_ablation.get(ratio_key, {}).get(criterion_name)

            val_acc = val_entry["accuracy"] if val_entry else None
            test_acc = test_entry["accuracy"] if test_entry else None

            val_str = f"{val_acc*100:.2f}%" if val_acc is not None else "N/A"
            test_str = f"{test_acc*100:.2f}%" if test_acc is not None else "N/A"
            if val_acc is not None and test_acc is not None:
                diff_str = f"{(val_acc - test_acc)*100:+.2f}%"
            else:
                diff_str = "N/A"

            print(f"{ratio_key:<8} {CRITERIA_LABELS[criterion_name]:<10} "
                  f"{val_str:<12} {test_str:<12} {diff_str}")

    print(f"{'=' * 80}")


def print_weights_comparison(old_weights, new_weights, acuan_key):
    """Cetak perbandingan bobot lama (dari test) dan bobot baru (dari val)."""
    print(f"\n{'=' * 60}")
    print(f"PERBANDINGAN BOBOT (rasio acuan: {acuan_key})")
    print(f"{'=' * 60}")
    print(f"{'':<14} {'Lama (test)':<14} {'Baru (val)':<14} {'Selisih'}")
    print(f"{'-' * 60}")

    labels = [
        ("w1 (L1)", "w1_l1"),
        ("w2 (BN)", "w2_bn"),
        ("w3 (Entropi)", "w3_entropy"),
    ]
    for label, key in labels:
        old_val = old_weights.get(key) if old_weights else None
        new_val = new_weights.get(key)

        old_str = f"{old_val:.4f}" if old_val is not None else "N/A"
        new_str = f"{new_val:.4f}" if new_val is not None else "N/A"
        diff_str = f"{(new_val - old_val):+.4f}" if (old_val is not None and new_val is not None) else "N/A"

        print(f"{label:<14} {old_str:<14} {new_str:<14} {diff_str}")

    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="Hitung ulang bobot w1, w2, w3 dari akurasi validasi"
    )
    parser.add_argument("--epochs", type=int, default=CFG.BASELINE_EPOCHS,
                        help="Jumlah epoch pada nama file checkpoint (default: 30)")
    parser.add_argument("--weight_ratio", type=float, default=None,
                        help="Rasio acuan untuk menghitung bobot. Default: "
                             "sama dengan source_ratio pada ablation_results.json lama")
    args = parser.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    old_test_results = load_old_test_results()
    if old_test_results is None:
        return
    old_weights = old_test_results.get("optimal_weights", {})

    # Rasio acuan default = rasio yang sama dipakai sebelumnya (03_ablation_study.py)
    if args.weight_ratio is not None:
        weight_ratio = args.weight_ratio
    else:
        weight_ratio = old_weights.get("source_ratio", 0.3)
    acuan_key = f"{int(weight_ratio * 100)}%"

    print("=" * 60)
    print("TAHAP 7: HITUNG ULANG BOBOT DARI AKURASI VALIDASI")
    print("=" * 60)
    print(f"  Device            : {device}")
    print(f"  Epoch (nama file) : {args.epochs}")
    print(f"  Rasio acuan bobot : {acuan_key}")

    # Muat DataLoader (hanya "val" yang dipakai untuk evaluasi ulang)
    SPLIT_DIR = CFG.DATASET_DIR.parent / "dataset_split"
    dataloaders, _ = get_dataloaders(SPLIT_DIR)

    # Evaluasi 21 checkpoint ablation pada val
    val_results = evaluate_all_checkpoints(dataloaders, device, args.epochs)

    # Hitung bobot baru dari akurasi val pada rasio acuan
    new_weights = {}
    if acuan_key in val_results and all(c in val_results[acuan_key] for c in CRITERIA):
        acc_l1 = val_results[acuan_key]["l1"]["accuracy"]
        acc_bn = val_results[acuan_key]["bn"]["accuracy"]
        acc_entropy = val_results[acuan_key]["entropy"]["accuracy"]

        w1, w2, w3 = compute_optimal_weights(acc_l1, acc_bn, acc_entropy)

        new_weights = {
            "w1_l1": w1,
            "w2_bn": w2,
            "w3_entropy": w3,
            "source_ratio": weight_ratio,
            "source_epochs": args.epochs,
            "source_split": "val",
            "acc_l1": acc_l1,
            "acc_bn": acc_bn,
            "acc_entropy": acc_entropy,
        }
    else:
        print(f"\n[PERINGATAN] Data val pada rasio acuan {acuan_key} tidak lengkap, "
              f"bobot baru tidak dapat dihitung.")

    # Cetak tabel perbandingan val vs test
    print_val_vs_test_table(val_results, old_test_results)

    # Cetak perbandingan bobot lama vs baru
    if new_weights:
        print_weights_comparison(old_weights, new_weights, acuan_key)

    # Simpan hasil akhir (TIDAK menimpa outputs/ablation_results.json)
    save_data = {
        "ablation_val_results": val_results,
        "optimal_weights_val": new_weights,
        "optimal_weights_test_old": old_weights,
        "epochs_used": args.epochs,
        "status": "SELESAI",
    }
    with open(VAL_RESULTS_PATH, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n[INFO] Hasil disimpan: {VAL_RESULTS_PATH}")
    print(f"[INFO] File hasil test lama tidak diubah: {OLD_TEST_RESULTS_PATH}")


if __name__ == "__main__":
    main()
