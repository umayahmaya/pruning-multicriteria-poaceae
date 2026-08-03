"""
11_ablation_entropy_traincal.py
Menjalankan ulang 7 eksperimen ablation kriteria ENTROPI SAJA (rasio 10%
sampai 70%, 30 epoch) memakai kalibrasi entropi yang sudah diperbaiki:
diambil dari dataloaders["train"], seimbang 20 gambar per kelas (9 kelas =
180 gambar), transform evaluasi (tanpa augmentasi), deterministik seed 42.
Lihat compute_entropy_scores() di src/pruning.py.

Checkpoint hasil ablation entropi LAMA (kalibrasi val, num_batches=10) di
checkpoints/ablation_entropy_*pct_30ep.pth TIDAK ditimpa -- checkpoint baru
memakai akhiran _traincal. Hasil metrik disimpan ke file JSON terpisah,
bukan menimpa outputs/ablation_results.json.

Jalankan:
    python scripts/11_ablation_entropy_traincal.py
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
from src.pruning import run_ablation_single_criterion
from src.visualize import print_results_table

RESULTS_PATH = CFG.OUTPUT_DIR / "ablation_entropy_traincal_results.json"


def set_seed(seed=CFG.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_single(model_baseline, dataloaders, dataset_sizes, ratio, device, epochs):
    """Jalankan satu skenario: pruning entropi (kalibrasi train) + fine-tuning + evaluasi."""

    ckpt_name = f"ablation_entropy_{int(ratio*100)}pct_{epochs}ep_traincal.pth"
    ckpt_path = CFG.CHECKPOINT_DIR / ckpt_name

    if ckpt_path.exists():
        print(f"\n{'=' * 60}")
        print(f"SKIP: ENTROPY traincal | {ratio*100:.0f}% | {epochs} ep (sudah ada)")
        print(f"{'=' * 60}")
        pruned_model, ckpt = load_checkpoint(ckpt_path, device)
        metrics, _, _ = evaluate_model(pruned_model, dataloaders["test"], device)
        print_results_table(metrics, f"ENTROPY traincal | {ratio*100:.0f}%")
        return metrics

    print(f"\n{'=' * 60}")
    print(f"ABLATION ENTROPI (kalibrasi train): Rasio {ratio*100:.0f}% | {epochs} epoch")
    print(f"{'=' * 60}")

    # Pruning dengan kriteria entropi, kalibrasi dari train
    pruned_model, masks = run_ablation_single_criterion(
        model_baseline, dataloaders["train"],
        ratio, "entropy", device
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
        phase_name=f"Ablation ENTROPI traincal {ratio*100:.0f}%"
    )

    # Evaluasi
    metrics, _, _ = evaluate_model(pruned_model, dataloaders["test"], device)
    print_results_table(metrics, f"ENTROPY traincal | {ratio*100:.0f}%")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Ulang ablation entropi dengan kalibrasi train"
    )
    parser.add_argument("--epochs", type=int, default=CFG.BASELINE_EPOCHS,
                        help="Jumlah epoch fine-tuning, sama dengan baseline (default: 30)")
    args = parser.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ratios_to_test = CFG.PRUNING_RATIOS  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    print("=" * 60)
    print("ULANG ABLATION ENTROPI -- KALIBRASI DARI TRAIN")
    print("=" * 60)
    print(f"  Device     : {device}")
    print(f"  Rasio      : {[f'{r*100:.0f}%' for r in ratios_to_test]}")
    print(f"  Epoch      : {args.epochs} (sama dengan baseline)")
    print(f"  Kriteria   : Entropi saja")
    print(f"  Kalibrasi  : dataloaders['train'], 20 gambar/kelas, seed={CFG.SEED}")
    print(f"  Checkpoint : akhiran _traincal (checkpoint lama tidak ditimpa)")
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

    all_results = {}
    start_time = time.time()

    for i, ratio in enumerate(ratios_to_test, 1):
        ratio_key = f"{int(ratio * 100)}%"
        elapsed = time.time() - start_time
        print(f"\n>>> Eksperimen {i}/{len(ratios_to_test)} "
              f"| Waktu berjalan: {elapsed/60:.1f} menit")

        metrics = run_single(model_baseline, dataloaders, dataset_sizes,
                              ratio, device, args.epochs)

        all_results[ratio_key] = {
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
            "num_params": metrics["num_params"],
            "model_size_mb": metrics["model_size_mb"],
            "flops": metrics["flops"],
            "inference_ms": metrics["inference_ms"],
        }

        # Simpan progres setiap selesai satu eksperimen (untuk keamanan)
        save_data = {
            "criterion": "entropy",
            "calibration_source": "train",
            "calibration_samples_per_class": 20,
            "calibration_seed": CFG.SEED,
            "results": all_results,
            "epochs_used": args.epochs,
            "status": f"{i}/{len(ratios_to_test)} selesai",
        }
        with open(RESULTS_PATH, "w") as f:
            json.dump(save_data, f, indent=2)

    total_time = time.time() - start_time

    # Cetak tabel ringkasan
    print(f"\n{'=' * 80}")
    print(f"RINGKASAN ABLATION ENTROPI (kalibrasi train, {args.epochs} epoch)")
    print(f"{'=' * 80}")
    print(f"{'Rasio':<8} {'Akurasi':<12} {'F1':<10}")
    print(f"{'-' * 40}")
    for ratio_key, m in all_results.items():
        print(f"{ratio_key:<8} {m['accuracy']*100:>8.2f}%   {m['f1_score']:.4f}")
    print(f"{'=' * 80}")
    print(f"Total waktu: {total_time/60:.1f} menit ({total_time/3600:.1f} jam)")

    # Simpan hasil final
    save_data = {
        "criterion": "entropy",
        "calibration_source": "train",
        "calibration_samples_per_class": 20,
        "calibration_seed": CFG.SEED,
        "results": all_results,
        "epochs_used": args.epochs,
        "total_time_seconds": total_time,
        "status": "SELESAI",
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n[INFO] Hasil disimpan: {RESULTS_PATH}")
    print(f"[INFO] Checkpoint lama (kalibrasi val) tidak tersentuh.")
    print(f"\n[SELESAI] Ablation entropi (kalibrasi train) selesai.")


if __name__ == "__main__":
    main()
