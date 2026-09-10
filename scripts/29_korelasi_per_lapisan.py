"""
29_korelasi_per_lapisan.py
Analisis per lapisan (bukan gabungan seluruh lapisan seperti
08_compare_channel_selection.py): untuk model baseline, hitung skor L1,
BN gamma, dan entropi (kalibrasi train, seed 42) pada seluruh 16 lapisan
yang bisa dipangkas, lalu untuk SETIAP lapisan secara terpisah laporkan:
  1. Korelasi Spearman antara skor L1 dan skor entropi di lapisan itu
  2. Korelasi Spearman antara skor L1 dan skor BN gamma di lapisan itu
  3. Rata-rata dan simpangan baku skor entropi MENTAH (sebelum
     normalisasi min-maks) di lapisan itu

Tujuan: mengetahui apakah hubungan antar kriteria seragam di semua
lapisan atau berbeda-beda, dan apakah ada lapisan yang entropinya jauh
lebih rendah/lebih tersebar dibanding lapisan lain.

Skrip ini TIDAK melatih apa pun dan TIDAK mengubah kode lain -- murni
memuat checkpoint baseline, menghitung skor, dan melaporkan.

Jalankan:
    venv/Scripts/python.exe scripts/29_korelasi_per_lapisan.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from scipy.stats import spearmanr

from src.config import CFG
from src.dataset import get_dataloaders
from src.model import load_checkpoint
from src.pruning import (
    compute_l1_scores, compute_bn_scores, compute_entropy_scores,
    _get_prunable_layers,
)

CHECKPOINT_NAME = "baseline_9class.pth"
RESULTS_PATH = CFG.OUTPUT_DIR / "korelasi_per_lapisan.json"


def main():
    device = torch.device("cpu")
    ckpt_path = CFG.CHECKPOINT_DIR / CHECKPOINT_NAME
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint tidak ditemukan: {ckpt_path}")
        return

    print("=" * 70)
    print("KORELASI ANTAR KRITERIA PER LAPISAN (MODEL BASELINE)")
    print("=" * 70)

    model, _ = load_checkpoint(ckpt_path, device)
    model.eval()

    SPLIT_DIR = CFG.DATASET_DIR.parent / "dataset_split"
    dataloaders, _ = get_dataloaders(SPLIT_DIR)

    # Block index MobileNetV2 asli, untuk konteks tiap layer_idx (0-15)
    prunable = _get_prunable_layers(model)
    block_idx_map = {layer_idx: block_idx for layer_idx, (block_idx, _) in enumerate(prunable)}

    print("\n[INFO] Menghitung tiga skor kepentingan channel...")
    l1_scores = compute_l1_scores(model)
    bn_scores = compute_bn_scores(model)
    entropy_scores = compute_entropy_scores(
        model, dataloaders["train"], device, samples_per_class=20, seed=CFG.SEED
    )

    common_layers = sorted(set(l1_scores) & set(bn_scores) & set(entropy_scores))
    if len(common_layers) != 16:
        print(f"[PERINGATAN] Jumlah lapisan yang punya ketiga skor: {len(common_layers)} (diharapkan 16)")

    hasil_per_lapisan = []
    for idx in common_layers:
        l1 = l1_scores[idx].numpy()
        bn = bn_scores[idx].numpy()
        entropi = entropy_scores[idx].numpy()

        rho_l1_entropi, p_l1_entropi = spearmanr(l1, entropi)
        rho_l1_bn, p_l1_bn = spearmanr(l1, bn)

        entropi_mean = float(entropi.mean())
        entropi_std = float(entropi.std())

        hasil_per_lapisan.append({
            "layer_idx": idx,
            "block_idx_mobilenetv2": block_idx_map.get(idx),
            "jumlah_channel": int(len(l1)),
            "spearman_l1_vs_entropi": float(rho_l1_entropi),
            "spearman_l1_vs_entropi_pvalue": float(p_l1_entropi),
            "spearman_l1_vs_bn": float(rho_l1_bn),
            "spearman_l1_vs_bn_pvalue": float(p_l1_bn),
            "entropi_mentah_rata_rata": entropi_mean,
            "entropi_mentah_simpangan_baku": entropi_std,
        })

    # ----- Tabel -----
    print(f"\n{'=' * 100}")
    print("TABEL KORELASI DAN STATISTIK ENTROPI PER LAPISAN")
    print("=" * 100)
    header = (f"{'Layer':>5} {'Block':>5} {'#Ch':>5} "
              f"{'rho(L1,Entropi)':>16} {'rho(L1,BN)':>12} "
              f"{'Entropi mean':>13} {'Entropi std':>12}")
    print(header)
    print("-" * 100)
    for r in hasil_per_lapisan:
        print(f"{r['layer_idx']:>5} {r['block_idx_mobilenetv2']:>5} {r['jumlah_channel']:>5} "
              f"{r['spearman_l1_vs_entropi']:>16.4f} {r['spearman_l1_vs_bn']:>12.4f} "
              f"{r['entropi_mentah_rata_rata']:>13.4f} {r['entropi_mentah_simpangan_baku']:>12.4f}")
    print("=" * 100)

    # ----- Ringkasan sebaran (bantu jawab pertanyaan penelitian) -----
    rho_entropi_vals = [r["spearman_l1_vs_entropi"] for r in hasil_per_lapisan]
    rho_bn_vals = [r["spearman_l1_vs_bn"] for r in hasil_per_lapisan]
    entropi_means = [r["entropi_mentah_rata_rata"] for r in hasil_per_lapisan]

    print("\nRingkasan sebaran antar lapisan:")
    print(f"  rho(L1,Entropi)  : min={min(rho_entropi_vals):.4f}  maks={max(rho_entropi_vals):.4f}  "
          f"rentang={max(rho_entropi_vals) - min(rho_entropi_vals):.4f}")
    print(f"  rho(L1,BN)       : min={min(rho_bn_vals):.4f}  maks={max(rho_bn_vals):.4f}  "
          f"rentang={max(rho_bn_vals) - min(rho_bn_vals):.4f}")
    layer_entropi_terendah = min(hasil_per_lapisan, key=lambda r: r["entropi_mentah_rata_rata"])
    layer_entropi_tertinggi = max(hasil_per_lapisan, key=lambda r: r["entropi_mentah_rata_rata"])
    print(f"  Rata-rata entropi mentah terendah  : layer {layer_entropi_terendah['layer_idx']} "
          f"({layer_entropi_terendah['entropi_mentah_rata_rata']:.4f})")
    print(f"  Rata-rata entropi mentah tertinggi : layer {layer_entropi_tertinggi['layer_idx']} "
          f"({layer_entropi_tertinggi['entropi_mentah_rata_rata']:.4f})")

    # ----- Simpan -----
    save_data = {
        "checkpoint": CHECKPOINT_NAME,
        "entropy_calibration": {
            "source_split": "train",
            "samples_per_class": 20,
            "seed": CFG.SEED,
        },
        "per_layer": hasil_per_lapisan,
        "ringkasan": {
            "rho_l1_entropi_min": min(rho_entropi_vals),
            "rho_l1_entropi_maks": max(rho_entropi_vals),
            "rho_l1_bn_min": min(rho_bn_vals),
            "rho_l1_bn_maks": max(rho_bn_vals),
            "layer_entropi_mentah_terendah": layer_entropi_terendah["layer_idx"],
            "layer_entropi_mentah_tertinggi": layer_entropi_tertinggi["layer_idx"],
        },
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Hasil disimpan: {RESULTS_PATH}")


if __name__ == "__main__":
    main()