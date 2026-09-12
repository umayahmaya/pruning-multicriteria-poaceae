"""
13_remeasure_inference.py
Mengukur ulang waktu inferensi seluruh model (baseline + 7 checkpoint
multicriteria _valweights) dalam SATU sesi berurutan, supaya kondisi sistem
(beban CPU, cache, dll.) konsisten saat dibandingkan antar model.

torch.set_num_threads(1) dipanggil di awal supaya jumlah thread konsisten
dan tidak dipengaruhi penjadwalan OS.

Untuk tiap model: 20 warmup + 300 pengukuran, dilaporkan median, rata-rata,
simpangan baku, serta persentil 5 dan 95 (bukan cuma rata-rata seperti
measure_inference_time() di src/model.py, dan bukan cuma std seperti versi
skrip ini sebelumnya).

Jalankan:
    venv/Scripts/python.exe scripts/13_remeasure_inference.py
"""

import sys
import os
import json
import copy
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CFG
from src.model import load_checkpoint

RESULTS_PATH = CFG.OUTPUT_DIR / "inference_remeasured.json"

# Override khusus skrip ini -- tidak mengubah CFG.INFERENCE_RUNS global
# supaya metrik inference_ms yang sudah tercatat di skrip lain tetap konsisten.
WARMUP_RUNS = CFG.WARMUP_RUNS  # 20
INFERENCE_RUNS = 300  # dinaikkan dari 100

RATIOS = [10, 20, 30, 40, 50, 60, 70]
MODELS = [("baseline", CFG.CHECKPOINT_DIR / "baseline_9class.pth")] + [
    (
        f"multicriteria_{r}pct_valweights",
        CFG.CHECKPOINT_DIR / f"multicriteria_{r}pct_30ep_valweights.pth",
    )
    for r in RATIOS
]


def measure_inference_stats(model, runs=INFERENCE_RUNS, warmup=WARMUP_RUNS):
    """
    Ukur waktu inferensi model pada CPU, kembalikan median, rata-rata,
    simpangan baku, serta persentil 5 dan 95 dari seluruh pengukuran.
    """
    cpu_model = copy.deepcopy(model).cpu().eval()
    dummy = torch.randn(1, 3, CFG.IMG_SIZE, CFG.IMG_SIZE)

    with torch.no_grad():
        for _ in range(warmup):
            _ = cpu_model(dummy)

    times = []
    with torch.no_grad():
        for _ in range(runs):
            start = time.perf_counter()
            _ = cpu_model(dummy)
            end = time.perf_counter()
            times.append((end - start) * 1000)

    times = np.array(times)
    return {
        "p5_ms": float(np.percentile(times, 5)),
        "median_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "mean_ms": float(np.mean(times)),
        "std_ms": float(np.std(times)),
        "runs": runs,
        "warmup": warmup,
    }


def main():
    torch.set_num_threads(1)
    device = torch.device("cpu")

    print("=" * 60)
    print("PENGUKURAN ULANG WAKTU INFERENSI")
    print("=" * 60)
    print(f"  Threads: {torch.get_num_threads()} (torch.set_num_threads(1))")
    print(f"  Warmup : {WARMUP_RUNS}")
    print(f"  Runs   : {INFERENCE_RUNS}")
    print(f"  Model  : {len(MODELS)} (baseline + 7 multicriteria valweights)")

    missing = [str(p) for _, p in MODELS if not p.exists()]
    if missing:
        print("\n[ERROR] Checkpoint berikut tidak ditemukan:")
        for m in missing:
            print(f"  - {m}")
        return

    results = {}
    start_time = time.time()

    for i, (name, ckpt_path) in enumerate(MODELS, 1):
        print(f"\n{'-' * 60}")
        print(f"[{i}/{len(MODELS)}] Memuat: {name} ({ckpt_path.name})")
        model, _ = load_checkpoint(ckpt_path, device)

        stats = measure_inference_stats(model)
        print(f"  P5     : {stats['p5_ms']:.3f} ms")
        print(f"  Median : {stats['median_ms']:.3f} ms")
        print(f"  P95    : {stats['p95_ms']:.3f} ms")
        print(f"  Rata2  : {stats['mean_ms']:.3f} ms")
        print(f"  Std    : {stats['std_ms']:.3f} ms")

        results[name] = {
            "checkpoint": str(ckpt_path),
            **stats,
        }

        # Simpan progres setiap selesai satu model (untuk keamanan)
        with open(RESULTS_PATH, "w") as f:
            json.dump({
                "warmup": WARMUP_RUNS,
                "runs": INFERENCE_RUNS,
                "num_threads": torch.get_num_threads(),
                "results": results,
                "status": f"{i}/{len(MODELS)} selesai",
            }, f, indent=2)

        del model

    total_time = time.time() - start_time

    # Ringkasan tabel
    print(f"\n{'=' * 95}")
    print("RINGKASAN WAKTU INFERENSI (satu sesi berurutan, 1 thread)")
    print(f"{'=' * 95}")
    print(f"{'Model':<32} {'P5 (ms)':<10} {'Median (ms)':<13} {'P95 (ms)':<10} "
          f"{'Rata2 (ms)':<12} {'Std (ms)'}")
    print(f"{'-' * 95}")
    for name, r in results.items():
        print(f"{name:<32} {r['p5_ms']:<10.3f} {r['median_ms']:<13.3f} "
              f"{r['p95_ms']:<10.3f} {r['mean_ms']:<12.3f} {r['std_ms']:.3f}")
    print(f"{'=' * 95}")
    print(f"Total waktu: {total_time/60:.1f} menit")

    with open(RESULTS_PATH, "w") as f:
        json.dump({
            "warmup": WARMUP_RUNS,
            "runs": INFERENCE_RUNS,
            "num_threads": torch.get_num_threads(),
            "results": results,
            "total_time_seconds": total_time,
            "status": "SELESAI",
        }, f, indent=2)
    print(f"\n[INFO] Hasil disimpan: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
