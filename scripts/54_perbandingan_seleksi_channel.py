"""
54_perbandingan_seleksi_channel.py
Menjawab item terbuka CLAUDE.md Bagian 8 butir 9 (sub-item "Item lama..."):
outputs/archive/bn_gamma/channel_selection_comparison.json (dari
scripts/archive/08_compare_channel_selection.py) hanya membandingkan bobot
KONSTAN lama vs lebih lama pada formula LAMA (S_BN). Skrip ini menghitung
ulang analisis yang sama untuk formula AKTIF (L1 + GM-Ekspansi + Entropi,
bobot w_k(r) per rasio -- lihat CLAUDE.md Bagian 2).

Dua bagian analisis, KEDUANYA dihitung per rasio (BUKAN sekali untuk semua
rasio seperti skrip 08 lama) karena bobot w_k(r) sekarang berbeda di setiap
rasio, sehingga baik I(c) itu sendiri maupun korelasinya terhadap tiap
kriteria tunggal juga ikut berbeda per rasio:

  A. UJI SENSITIVITAS BOBOT
     Bobot hasil ablation per-rasio (outputs/multicriteria_per_rasio.json)
     dibandingkan dengan bobot rata sepertiga (1/3, 1/3, 1/3) sebagai
     pembanding naif. Ini pengganti perbandingan "bobot lama vs baru" pada
     skrip 08 lama -- pembandingnya berubah karena pada formula lama hanya
     ada satu set bobot konstan yang dikoreksi (test->val), sedangkan pada
     formula aktif tidak ada "bobot lama" yang setara (bobot SELALU per
     rasio sejak awal, lihat CLAUDE.md Bagian 2). Bobot rata sepertiga
     dipilih sebagai pembanding paling netral karena beberapa rasio (60%,
     70%) punya bobot yang JAUH dari 1/3 masing-masing -- berbeda dari
     formula lama yang bobotnya sudah dekat 1/3 di semua kriteria (lihat
     CLAUDE.md Bagian 8 butir 4).

  B. MULTI-KRITERIA (bobot per-rasio) vs KRITERIA TUNGGAL
     Korelasi Spearman antara I(c) dan tiap skor kriteria tunggal
     ternormalisasi (L1, GM-Ekspansi, Entropi), plus persentase channel
     yang berbeda status pangkas/pertahankan, untuk ketujuh rasio.

Rekonstruksi skor: pola IDENTIK dengan scripts/49/51/52 -- fungsi dari
src/pruning.py diimpor apa adanya (TIDAK diubah), hitung_skor_gm_ekspansi_
semua_layer() diimpor dinamis dari scripts/43_ablation_gm_expansion.py.
Bobot w_k(r) dibaca apa adanya dari outputs/multicriteria_per_rasio.json.

Skrip ini TIDAK melatih model apa pun, hanya menghitung skor dan mask
(non-destruktif, tidak menyentuh checkpoint mana pun).

Jalankan:
    venv/Scripts/python.exe scripts/54_perbandingan_seleksi_channel.py
"""

import sys
import os
import json
import importlib.util

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from scipy.stats import spearmanr

from src.config import CFG
from src.dataset import get_dataloaders
from src.model import load_checkpoint
from src.pruning import (
    compute_l1_scores, compute_entropy_scores, normalize_min_max,
    compute_importance_scores, get_pruning_mask,
)

# ----- Impor dinamis hitung_skor_gm_ekspansi_semua_layer dari skrip 43 -----
# (pola identik dengan scripts/44/45/46/48/49/51/52, src/pruning.py TIDAK diubah)
_JALUR_SKRIP_43 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "43_ablation_gm_expansion.py")
_spec_43 = importlib.util.spec_from_file_location("_skrip_43_ablation_gm_expansion", _JALUR_SKRIP_43)
_mod_43 = importlib.util.module_from_spec(_spec_43)
_spec_43.loader.exec_module(_mod_43)
hitung_skor_gm_ekspansi_semua_layer = _mod_43.hitung_skor_gm_ekspansi_semua_layer

PER_RASIO_PATH = CFG.OUTPUT_DIR / "multicriteria_per_rasio.json"
RESULTS_PATH = CFG.OUTPUT_DIR / "channel_selection_comparison_per_rasio.json"


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

    print("=" * 78)
    print("PERBANDINGAN SELEKSI CHANNEL I(c) -- FORMULA AKTIF (per-rasio)")
    print("=" * 78)

    baseline_path = CFG.CHECKPOINT_DIR / "baseline_9class.pth"
    if not baseline_path.exists():
        print(f"[ERROR] {baseline_path} tidak ditemukan.")
        return
    if not PER_RASIO_PATH.exists():
        print(f"[ERROR] {PER_RASIO_PATH} tidak ditemukan. "
              f"Jalankan scripts/44_multicriteria_per_rasio.py terlebih dahulu.")
        return

    with open(PER_RASIO_PATH, encoding="utf-8") as f:
        bobot_per_rasio = json.load(f)["bobot_per_rasio"]

    model_baseline, _ = load_checkpoint(baseline_path, device)
    model_baseline.eval()

    SPLIT_DIR = CFG.DATASET_DIR.parent / "dataset_split"
    dataloaders, _ = get_dataloaders(SPLIT_DIR)

    print("\n[INFO] Menghitung skor L1, GM-Ekspansi, Entropi sekali "
          "(tidak bergantung rasio)...")
    l1_scores = compute_l1_scores(model_baseline)
    gm_scores = hitung_skor_gm_ekspansi_semua_layer(model_baseline)
    entropy_scores = compute_entropy_scores(
        model_baseline, dataloaders["train"], device,
        samples_per_class=20, seed=CFG.SEED
    )

    total_channels = sum(len(s) for s in l1_scores.values())
    print(f"[INFO] Total channel intermediate: {total_channels} "
          f"({len(l1_scores)} lapisan)")
    if total_channels != 7104:
        print(f"[PERINGATAN] Total channel ({total_channels}) berbeda dari "
              f"7104 yang diharapkan -- periksa arsitektur/checkpoint baseline.")

    norm_l1 = normalize_min_max(l1_scores)
    norm_gm = normalize_min_max(gm_scores)
    norm_entropy = normalize_min_max(entropy_scores)

    # ================================================================
    # A. UJI SENSITIVITAS BOBOT: bobot ablation per-rasio vs bobot 1/3-1/3-1/3
    # ================================================================
    print(f"\n{'=' * 78}")
    print("A. UJI SENSITIVITAS BOBOT (bobot ablation per-rasio vs bobot 1/3 rata)")
    print(f"{'=' * 78}")
    print(f"{'Rasio':<8} {'w_L1/w_GM/w_Ent (ablation)':<30} {'Channel Beda':<14} "
          f"{'Persentase':<12} {'Spearman rho'}")
    print(f"{'-' * 78}")

    sensitivitas_bobot = {}
    for ratio in CFG.PRUNING_RATIOS:
        pct = int(round(ratio * 100))
        ratio_key = f"{pct}%"
        if ratio_key not in bobot_per_rasio:
            print(f"[PERINGATAN] Bobot untuk rasio {ratio_key} tidak ditemukan, dilewati.")
            continue
        w = bobot_per_rasio[ratio_key]
        w_l1, w_gm, w_ent = w["w_l1"], w["w_gm"], w["w_entropi"]

        importance_ablation = compute_importance_scores(
            l1_scores, gm_scores, entropy_scores, w1=w_l1, w2=w_gm, w3=w_ent
        )
        importance_equal = compute_importance_scores(
            l1_scores, gm_scores, entropy_scores, w1=1 / 3, w2=1 / 3, w3=1 / 3
        )

        flat_ablation = flatten(importance_ablation)
        flat_equal = flatten(importance_equal)
        rho, pval = spearmanr(flat_ablation, flat_equal)

        mask_ablation = get_pruning_mask(importance_ablation, ratio)
        mask_equal = get_pruning_mask(importance_equal, ratio)
        n_diff, n_total = masks_differ_count(mask_ablation, mask_equal)
        pct_diff = n_diff / n_total * 100

        print(f"{ratio_key:<8} {f'{w_l1:.3f}/{w_gm:.3f}/{w_ent:.3f}':<30} "
              f"{n_diff:<14} {pct_diff:>6.2f}%      {rho:.4f}")

        sensitivitas_bobot[ratio_key] = {
            "bobot_ablation": {"w_l1": w_l1, "w_gm": w_gm, "w_entropi": w_ent},
            "bobot_pembanding": {"w_l1": 1 / 3, "w_gm": 1 / 3, "w_entropi": 1 / 3},
            "channels_different": n_diff,
            "total_channels": n_total,
            "percent_different": pct_diff,
            "spearman_rho": rho,
            "spearman_pvalue": pval,
        }

    print(f"{'=' * 78}")

    # ================================================================
    # B. MULTI-KRITERIA (bobot per-rasio) vs KRITERIA TUNGGAL, per rasio
    # ================================================================
    print(f"\n{'=' * 78}")
    print("B. MULTI-KRITERIA (bobot per-rasio) vs KRITERIA TUNGGAL, PER RASIO")
    print(f"{'=' * 78}")
    print(f"{'Rasio':<8} {'vs L1':<20} {'vs GM-Ekspansi':<20} {'vs Entropi'}")
    print(f"{'-' * 78}")

    flat_l1 = flatten(norm_l1)
    flat_gm = flatten(norm_gm)
    flat_entropy = flatten(norm_entropy)
    mask_l1_ref = {r: get_pruning_mask(norm_l1, r) for r in CFG.PRUNING_RATIOS}
    mask_gm_ref = {r: get_pruning_mask(norm_gm, r) for r in CFG.PRUNING_RATIOS}
    mask_entropy_ref = {r: get_pruning_mask(norm_entropy, r) for r in CFG.PRUNING_RATIOS}

    multi_vs_single = {}
    for ratio in CFG.PRUNING_RATIOS:
        pct = int(round(ratio * 100))
        ratio_key = f"{pct}%"
        if ratio_key not in bobot_per_rasio:
            continue
        w = bobot_per_rasio[ratio_key]
        importance_multi = compute_importance_scores(
            l1_scores, gm_scores, entropy_scores,
            w1=w["w_l1"], w2=w["w_gm"], w3=w["w_entropi"]
        )
        flat_multi = flatten(importance_multi)

        rho_l1, pval_l1 = spearmanr(flat_multi, flat_l1)
        rho_gm, pval_gm = spearmanr(flat_multi, flat_gm)
        rho_entropy, pval_entropy = spearmanr(flat_multi, flat_entropy)

        mask_multi = get_pruning_mask(importance_multi, ratio)
        n_diff_l1, n_total = masks_differ_count(mask_multi, mask_l1_ref[ratio])
        n_diff_gm, _ = masks_differ_count(mask_multi, mask_gm_ref[ratio])
        n_diff_entropy, _ = masks_differ_count(mask_multi, mask_entropy_ref[ratio])

        pct_l1 = n_diff_l1 / n_total * 100
        pct_gm = n_diff_gm / n_total * 100
        pct_entropy = n_diff_entropy / n_total * 100

        print(f"{ratio_key:<8} {pct_l1:>6.2f}% ({n_diff_l1:>4}, rho={rho_l1:.3f})   "
              f"{pct_gm:>6.2f}% ({n_diff_gm:>4}, rho={rho_gm:.3f})   "
              f"{pct_entropy:>6.2f}% ({n_diff_entropy:>4}, rho={rho_entropy:.3f})")

        multi_vs_single[ratio_key] = {
            "bobot": {"w_l1": w["w_l1"], "w_gm": w["w_gm"], "w_entropi": w["w_entropi"]},
            "vs_l1": {"channels_different": n_diff_l1, "percent_different": pct_l1,
                      "spearman_rho": rho_l1, "spearman_pvalue": pval_l1},
            "vs_gm": {"channels_different": n_diff_gm, "percent_different": pct_gm,
                      "spearman_rho": rho_gm, "spearman_pvalue": pval_gm},
            "vs_entropy": {"channels_different": n_diff_entropy, "percent_different": pct_entropy,
                           "spearman_rho": rho_entropy, "spearman_pvalue": pval_entropy},
            "total_channels": n_total,
        }

    print(f"{'=' * 78}")

    save_data = {
        "catatan": (
            "Formula AKTIF (L1 + GM-Ekspansi + Entropi, bobot w_k(r) per rasio, "
            "CLAUDE.md Bagian 2). Menggantikan outputs/archive/bn_gamma/"
            "channel_selection_comparison.json (formula lama, S_BN, bobot konstan)."
        ),
        "total_channels": total_channels,
        "num_layers": len(l1_scores),
        "sensitivitas_bobot_ablation_vs_sepertiga_rata": sensitivitas_bobot,
        "multi_vs_single_criterion_by_ratio": multi_vs_single,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2)
    print(f"\n[INFO] Hasil disimpan: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
