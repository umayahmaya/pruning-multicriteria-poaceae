"""
08_compare_channel_selection.py
Membandingkan seleksi channel I(c) antara bobot lama (dari test) dan bobot
baru (dari val), serta antara skor multi-kriteria dan kriteria tunggal.

Menjawab item terbuka CLAUDE.md butir 8.3: korelasi peringkat (Spearman)
antara I(c) dan tiap kriteria tunggal, serta persentase channel yang
berbeda pilihannya -- untuk SETIAP rasio 10-70 persen, bukan cuma rasio
acuan 30 persen.

Skor L1, BN, dan Entropi dihitung SEKALI dari model baseline (skor entropi
memakai kalibrasi train yang sudah diperbaiki -- lihat compute_entropy_scores
di src/pruning.py), lalu dipakai ulang untuk kedua set bobot supaya
perbandingan apple-to-apple (hanya bobot kombinasinya yang berbeda).

Bobot lama dimuat dari outputs/ablation_results.json (optimal_weights).
Bobot baru dimuat dari outputs/ablation_val_results.json (optimal_weights_val).

Skrip ini TIDAK melatih model apa pun, hanya menghitung skor dan mask.

Jalankan:
    venv/Scripts/python.exe scripts/08_compare_channel_selection.py
"""

import sys
import os
import json

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CFG
from src.dataset import get_dataloaders
from src.model import load_checkpoint
from src.pruning import (
    compute_l1_scores, compute_bn_scores, compute_entropy_scores,
    normalize_min_max, compute_importance_scores, get_pruning_mask
)

OLD_WEIGHTS_PATH = CFG.OUTPUT_DIR / "ablation_results.json"
NEW_WEIGHTS_PATH = CFG.OUTPUT_DIR / "ablation_val_results.json"
RESULTS_PATH = CFG.OUTPUT_DIR / "channel_selection_comparison.json"


def load_weights():
    """Muat bobot lama (dari test) dan bobot baru (dari val) dari JSON."""
    if not OLD_WEIGHTS_PATH.exists():
        print(f"[ERROR] {OLD_WEIGHTS_PATH} tidak ditemukan.")
        print("Jalankan 03_ablation_study.py terlebih dahulu.")
        return None, None

    if not NEW_WEIGHTS_PATH.exists():
        print(f"[ERROR] {NEW_WEIGHTS_PATH} tidak ditemukan.")
        print("Jalankan 07_recompute_weights_val.py terlebih dahulu.")
        return None, None

    with open(OLD_WEIGHTS_PATH) as f:
        old_data = json.load(f)["optimal_weights"]
    with open(NEW_WEIGHTS_PATH) as f:
        new_data = json.load(f)["optimal_weights_val"]

    old_weights = (old_data["w1_l1"], old_data["w2_bn"], old_data["w3_entropy"])
    new_weights = (new_data["w1_l1"], new_data["w2_bn"], new_data["w3_entropy"])
    return old_weights, new_weights


def flatten(scores_dict):
    """Gabungkan skor semua lapisan jadi satu vektor 1D (urut layer_idx)."""
    return torch.cat([scores_dict[idx] for idx in sorted(scores_dict)]).numpy()


def masks_differ_count(masks_a, masks_b):
    """Hitung jumlah channel yang keputusan pangkas/pertahankan-nya berbeda."""
    total_diff = 0
    total_channels = 0
    for idx in sorted(masks_a):
        diff = (masks_a[idx] != masks_b[idx]).sum().item()
        total_diff += diff
        total_channels += len(masks_a[idx])
    return total_diff, total_channels


def main():
    device = torch.device("cpu")

    print("=" * 60)
    print("PERBANDINGAN SELEKSI CHANNEL I(c)")
    print("=" * 60)

    old_weights, new_weights = load_weights()
    if old_weights is None:
        return
    w1_old, w2_old, w3_old = old_weights
    w1_new, w2_new, w3_new = new_weights

    print(f"  Bobot lama (test) : w1={w1_old:.4f}, w2={w2_old:.4f}, w3={w3_old:.4f}")
    print(f"  Bobot baru (val)  : w1={w1_new:.4f}, w2={w2_new:.4f}, w3={w3_new:.4f}")

    # Muat model baseline dan data train (untuk kalibrasi entropi)
    SPLIT_DIR = CFG.DATASET_DIR.parent / "dataset_split"
    dataloaders, _ = get_dataloaders(SPLIT_DIR)

    BASELINE_CKPT = CFG.CHECKPOINT_DIR / "baseline_9class.pth"
    if not BASELINE_CKPT.exists():
        print("[ERROR] Checkpoint baseline tidak ditemukan!")
        print("Jalankan 02_train_baseline.py terlebih dahulu.")
        return

    model_baseline, _ = load_checkpoint(BASELINE_CKPT, device)

    # Hitung tiga skor mentah SEKALI, dipakai ulang untuk kedua bobot
    print("\n[INFO] Menghitung skor L1, BN, dan Entropi (kalibrasi train)...")
    l1_scores = compute_l1_scores(model_baseline)
    bn_scores = compute_bn_scores(model_baseline)
    entropy_scores = compute_entropy_scores(
        model_baseline, dataloaders["train"], device,
        samples_per_class=20, seed=CFG.SEED
    )

    total_channels = sum(len(s) for s in l1_scores.values())
    print(f"\n[INFO] Total channel intermediate: {total_channels} "
          f"({len(l1_scores)} lapisan)")
    if total_channels != 7104:
        print(f"[PERINGATAN] Total channel ({total_channels}) berbeda dari "
              f"7104 yang diharapkan -- periksa arsitektur/checkpoint baseline.")

    # I(c) untuk kedua set bobot (pakai skor mentah yang sama)
    print("\n[INFO] Menghitung I(c) dengan bobot lama...")
    importance_old = compute_importance_scores(
        l1_scores, bn_scores, entropy_scores, w1=w1_old, w2=w2_old, w3=w3_old
    )
    print("\n[INFO] Menghitung I(c) dengan bobot baru...")
    importance_new = compute_importance_scores(
        l1_scores, bn_scores, entropy_scores, w1=w1_new, w2=w2_new, w3=w3_new
    )

    # ================================================================
    # 1 & 2: per rasio -- channel berbeda (lama vs baru) + korelasi Spearman
    # ================================================================
    flat_old = flatten(importance_old)
    flat_new = flatten(importance_new)
    rho_old_new, pval_old_new = spearmanr(flat_old, flat_new)

    per_ratio_results = {}
    print(f"\n{'=' * 80}")
    print("1-2. SELEKSI CHANNEL: BOBOT LAMA vs BOBOT BARU (per rasio)")
    print(f"{'=' * 80}")
    print(f"Korelasi Spearman I(c)_lama vs I(c)_baru (seluruh {total_channels} "
          f"channel): rho={rho_old_new:.4f} (p={pval_old_new:.2e})")
    print("(rho ini SAMA untuk semua rasio -- tidak bergantung pada ambang "
          "pemangkasan, hanya dicantumkan ulang per baris untuk referensi)")
    print(f"{'-' * 80}")
    print(f"{'Rasio':<8} {'Channel Berbeda':<18} {'Persentase':<12} {'Spearman rho'}")
    print(f"{'-' * 80}")

    for ratio in CFG.PRUNING_RATIOS:
        ratio_key = f"{int(ratio * 100)}%"
        masks_old = get_pruning_mask(importance_old, ratio)
        masks_new = get_pruning_mask(importance_new, ratio)
        n_diff, n_total = masks_differ_count(masks_old, masks_new)
        pct_diff = n_diff / n_total * 100

        print(f"{ratio_key:<8} {n_diff:<18} {pct_diff:>6.2f}%      {rho_old_new:.4f}")

        per_ratio_results[ratio_key] = {
            "channels_different": n_diff,
            "total_channels": n_total,
            "percent_different": pct_diff,
            "spearman_rho_old_vs_new": rho_old_new,
        }

    print(f"{'=' * 80}")

    # ================================================================
    # 3-5: MULTI-KRITERIA (bobot baru) vs KRITERIA TUNGGAL, per rasio 10-70%
    # ================================================================
    print(f"\n{'=' * 80}")
    print("3-5. MULTI-KRITERIA (bobot val) vs KRITERIA TUNGGAL, PER RASIO")
    print(f"{'=' * 80}")

    norm_l1 = normalize_min_max(l1_scores)
    norm_bn = normalize_min_max(bn_scores)
    norm_entropy = normalize_min_max(entropy_scores)

    flat_multi = flatten(importance_new)
    flat_l1 = flatten(norm_l1)
    flat_bn = flatten(norm_bn)
    flat_entropy = flatten(norm_entropy)

    # Korelasi Spearman tidak bergantung pada ambang rasio (murni peringkat
    # skor penuh), jadi dihitung sekali dan berlaku untuk semua rasio.
    rho_l1, pval_l1 = spearmanr(flat_multi, flat_l1)
    rho_bn, pval_bn = spearmanr(flat_multi, flat_bn)
    rho_entropy, pval_entropy = spearmanr(flat_multi, flat_entropy)

    print(f"Korelasi Spearman I(c)_multi-kriteria vs tiap kriteria tunggal "
          f"(seluruh {total_channels} channel, SAMA untuk semua rasio):")
    print(f"  vs L1 saja      : rho={rho_l1:.4f} (p={pval_l1:.2e})")
    print(f"  vs BN saja      : rho={rho_bn:.4f} (p={pval_bn:.2e})")
    print(f"  vs Entropi saja : rho={rho_entropy:.4f} (p={pval_entropy:.2e})")
    print(f"{'-' * 80}")
    print(f"{'Rasio':<8} {'vs L1':<20} {'vs BN':<20} {'vs Entropi'}")
    print(f"{'-' * 80}")

    single_criterion_results = {}
    for ratio in CFG.PRUNING_RATIOS:
        ratio_key = f"{int(ratio * 100)}%"
        mask_multi = get_pruning_mask(importance_new, ratio)
        mask_l1 = get_pruning_mask(norm_l1, ratio)
        mask_bn = get_pruning_mask(norm_bn, ratio)
        mask_entropy = get_pruning_mask(norm_entropy, ratio)

        n_diff_l1, n_total = masks_differ_count(mask_multi, mask_l1)
        n_diff_bn, _ = masks_differ_count(mask_multi, mask_bn)
        n_diff_entropy, _ = masks_differ_count(mask_multi, mask_entropy)

        pct_l1 = n_diff_l1 / n_total * 100
        pct_bn = n_diff_bn / n_total * 100
        pct_entropy = n_diff_entropy / n_total * 100

        print(f"{ratio_key:<8} {pct_l1:>6.2f}% ({n_diff_l1:>4})   "
              f"{pct_bn:>6.2f}% ({n_diff_bn:>4})   "
              f"{pct_entropy:>6.2f}% ({n_diff_entropy:>4})")

        single_criterion_results[ratio_key] = {
            "vs_l1": {"channels_different": n_diff_l1, "percent_different": pct_l1,
                      "spearman_rho": rho_l1, "spearman_pvalue": pval_l1},
            "vs_bn": {"channels_different": n_diff_bn, "percent_different": pct_bn,
                      "spearman_rho": rho_bn, "spearman_pvalue": pval_bn},
            "vs_entropy": {"channels_different": n_diff_entropy, "percent_different": pct_entropy,
                           "spearman_rho": rho_entropy, "spearman_pvalue": pval_entropy},
            "total_channels": n_total,
        }

    print(f"{'=' * 80}")

    # Simpan hasil
    save_data = {
        "old_weights": {"w1_l1": w1_old, "w2_bn": w2_old, "w3_entropy": w3_old},
        "new_weights": {"w1_l1": w1_new, "w2_bn": w2_new, "w3_entropy": w3_new},
        "total_channels": total_channels,
        "num_layers": len(l1_scores),
        "old_vs_new_by_ratio": per_ratio_results,
        "spearman_old_vs_new_overall": {
            "rho": rho_old_new,
            "pvalue": pval_old_new,
        },
        "multi_vs_single_criterion_by_ratio": single_criterion_results,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n[INFO] Hasil disimpan: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
