"""
17_generate_final_table.py
Memuat ulang 29 checkpoint dari disk (baseline + 21 ablation kriteria
tunggal + 7 multi-kriteria bobot val) dan mengevaluasi masing-masing dengan
SATU protokol yang seragam untuk memperoleh kedelapan metrik lengkap:
akurasi, precision, recall, F1, jumlah parameter, ukuran MB, FLOPs, dan
waktu inferensi (median, 1 thread, 300 runs -- lihat measure_inference_time
di src/model.py).

TIDAK ADA training di skrip ini -- murni muat checkpoint + evaluasi.

Cakupan (29 checkpoint):
    - checkpoints/baseline_9class.pth
    - checkpoints/ablation_l1_{10..70}pct_30ep.pth        (7, data-free)
    - checkpoints/ablation_bn_{10..70}pct_30ep.pth         (7, data-free)
    - checkpoints/ablation_entropy_{10..70}pct_30ep_traincal.pth
      (7, kalibrasi TRAIN yang sudah diperbaiki -- BUKAN versi lama)
    - checkpoints/multicriteria_{10..70}pct_30ep_valweights.pth
      (7, bobot dari akurasi VAL hasil skrip 07 -- BUKAN versi test-derived lama)

Evaluasi dilakukan pada dataloaders["test"] (175 gambar), sesuai aturan
CLAUDE.md bahwa pelaporan akhir memakai subset test.

Output: outputs/tabel_hasil_lengkap.json, dengan blok metadata di bagian
atas yang mencatat sumber bobot, sumber kalibrasi entropi, protokol
pengukuran latensi, jumlah epoch, dan subset evaluasi.

Jalankan:
    venv/Scripts/python.exe scripts/17_generate_final_table.py
"""

import sys
import os
import json
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CFG
from src.dataset import get_dataloaders
from src.model import load_checkpoint, evaluate_model

RESULTS_PATH = CFG.OUTPUT_DIR / "tabel_hasil_lengkap.json"
VAL_WEIGHTS_PATH = CFG.OUTPUT_DIR / "ablation_val_results.json"
RATIOS = [10, 20, 30, 40, 50, 60, 70]

METRIC_KEYS = [
    "accuracy", "precision", "recall", "f1_score",
    "num_params", "model_size_mb", "flops",
    "inference_ms", "inference_std_ms", "inference_runs",
]


def load_weights_metadata():
    """Ambil nilai w1/w2/w3 (bobot val) untuk dicatat di metadata -- tidak
    dipakai untuk menghitung apa pun di sini, hanya untuk pencatatan."""
    if not VAL_WEIGHTS_PATH.exists():
        return None
    with open(VAL_WEIGHTS_PATH) as f:
        data = json.load(f)
    return data.get("optimal_weights_val")


def build_plan():
    """Susun daftar (skenario, rasio_key, path_checkpoint)."""
    plan = [("baseline", None, CFG.CHECKPOINT_DIR / "baseline_9class.pth")]

    for ratio in RATIOS:
        ratio_key = f"{ratio}%"
        plan.append(("l1", ratio_key, CFG.CHECKPOINT_DIR / f"ablation_l1_{ratio}pct_30ep.pth"))
    for ratio in RATIOS:
        ratio_key = f"{ratio}%"
        plan.append(("bn", ratio_key, CFG.CHECKPOINT_DIR / f"ablation_bn_{ratio}pct_30ep.pth"))
    for ratio in RATIOS:
        ratio_key = f"{ratio}%"
        plan.append(("entropy", ratio_key,
                      CFG.CHECKPOINT_DIR / f"ablation_entropy_{ratio}pct_30ep_traincal.pth"))
    for ratio in RATIOS:
        ratio_key = f"{ratio}%"
        plan.append(("multicriteria", ratio_key,
                      CFG.CHECKPOINT_DIR / f"multicriteria_{ratio}pct_30ep_valweights.pth"))

    return plan


def extract_metrics(metrics):
    return {k: metrics[k] for k in METRIC_KEYS}


def main():
    device = torch.device("cpu")

    print("=" * 60)
    print("TUGAS 3: TABEL HASIL LENGKAP (evaluasi ulang, tanpa training)")
    print("=" * 60)

    plan = build_plan()
    missing = [str(p) for _, _, p in plan if not p.exists()]
    if missing:
        print("[ERROR] Checkpoint berikut tidak ditemukan:")
        for m in missing:
            print(f"  - {m}")
        return

    weights_val = load_weights_metadata()

    SPLIT_DIR = CFG.DATASET_DIR.parent / "dataset_split"
    dataloaders, _ = get_dataloaders(SPLIT_DIR)

    metadata = {
        "weights_source": "outputs/ablation_val_results.json (optimal_weights_val, dari 07_recompute_weights_val.py)",
        "weights_used": weights_val,
        "entropy_calibration": {
            "source_split": "train",
            "samples_per_class": 20,
            "num_classes": CFG.NUM_CLASSES,
            "total_calibration_images": 20 * CFG.NUM_CLASSES,
            "seed": CFG.SEED,
            "transform": "get_transforms('test') -- resize + normalisasi saja, TANPA augmentasi",
            "checkpoint_suffix": "_traincal",
        },
        "inference_protocol": {
            "runs": CFG.INFERENCE_RUNS,
            "warmup": CFG.WARMUP_RUNS,
            "num_threads": 1,
            "device": "cpu",
            "statistic_reported": "median (juga menyimpan std dan jumlah runs)",
        },
        "epochs": {
            "baseline": CFG.BASELINE_EPOCHS,
            "finetune_pruned": CFG.FINETUNE_EPOCHS,
        },
        "eval_subset": "test (175 gambar)",
        "checkpoints_evaluated": 29,
        "generated_by": "17_generate_final_table.py",
    }

    results = {
        "baseline": None,
        "l1": {},
        "bn": {},
        "entropy": {},
        "multicriteria": {},
    }

    start_time = time.time()
    for i, (scenario, ratio_key, ckpt_path) in enumerate(plan, 1):
        elapsed = time.time() - start_time
        label = scenario if ratio_key is None else f"{scenario} {ratio_key}"
        print(f"\n>>> [{i}/{len(plan)}] {label} | Waktu berjalan: {elapsed/60:.1f} menit")
        print(f"    Checkpoint: {ckpt_path.name}")

        model, _ = load_checkpoint(ckpt_path, device)
        metrics, _, _ = evaluate_model(model, dataloaders["test"], device)
        entry = extract_metrics(metrics)
        entry["checkpoint"] = ckpt_path.name

        if scenario == "baseline":
            results["baseline"] = entry
        else:
            results[scenario][ratio_key] = entry

        del model

        # Simpan progres setiap satu checkpoint selesai
        save_data = {
            "metadata": metadata,
            "results": results,
            "status": f"{i}/{len(plan)} selesai",
        }
        with open(RESULTS_PATH, "w") as f:
            json.dump(save_data, f, indent=2)

    total_time = time.time() - start_time
    metadata["total_time_seconds"] = total_time

    save_data = {
        "metadata": metadata,
        "results": results,
        "status": "SELESAI",
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"[SELESAI] {len(plan)} checkpoint dievaluasi dalam {total_time/60:.1f} menit")
    print(f"[INFO] Hasil disimpan: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
