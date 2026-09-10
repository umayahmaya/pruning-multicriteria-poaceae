"""
43_ablation_gm_expansion.py
Ablation study GM SAJA pada SELURUH 7 rasio (CFG.PRUNING_RATIOS, 0.1
s.d. 0.7), TAPI dihitung dari bobot EXPANSION CONV 1x1 (BUKAN dari
depthwise conv seperti src/gm_scores.py/42_ablation_gm.py -- itu modul
GM yang BERBEDA dan SENGAJA tidak dipakai di sini).

Definisi skor (SAMA formula "jumlah jarak berpasangan" dengan
src/gm_scores.py, TAPI sumber bobotnya EXPANSION CONV):

    s_GM(c) = sum_{j != c} ||W_c - W_j||_2

W_c = bobot filter EXPANSION CONV 1x1 channel c (baris ke-c matriks
bobot, bentuk asli [in_channels_blok, 1, 1]), DIRATAKAN jadi vektor.
Skor besar = unik, skor kecil = redundan -- arah ini SUDAH selaras
dengan konvensi get_pruning_mask() (memangkas skor terendah), tidak
perlu dibalik.

Identifikasi lapisan expansion (cari_conv_ekspansi) DISALIN PERSIS dari
36_skor_redundansi.py/37/38 (kriteria: conv 1x1, groups=1,
out_channels > in_channels -- BUKAN conv proyeksi yang out<in), TIDAK
diubah. TAPI rumus skornya BERBEDA dari 36/37/38/40/41 (yang memakai
jarak ke MEDIAN GEOMETRIK / algoritma Weiszfeld, satu skalar per
channel) -- di sini rumusnya jumlah jarak ke SELURUH channel lain,
identik dengan src/gm_scores.py, hanya sumber bobotnya (expansion,
bukan depthwise) yang disamakan dengan 36/37/38. Jarak dihitung lewat
torch.cdist (Euclidean, p=2) -- setara torch.norm per pasangan, dipilih
karena lebih hemat memori untuk vektor berdimensi lebih tinggi
(in_channels blok, sampai 160) dibanding vektor depthwise (dimensi 9).

TIDAK memakai src/gm_scores.py (itu GM depthwise, salah untuk tujuan
ini) dan TIDAK mengubah 36/37/38/42/src/pruning.py -- fungsi identifikasi
lapisan dan skor GM-ekspansi ditulis ulang lengkap di file BARU ini.

Prosedur fine-tuning SAMA PERSIS dengan 03_ablation_study.py: 30 epoch
(CFG.BASELINE_EPOCHS), CFG.FINETUNE_LR, Adam+StepLR (train_model()),
seed=CFG.SEED (42) dipanggil SEKALI di awal sebelum loop rasio --
keterbatasan metodologis yang sama berlaku (CLAUDE.md Bagian 9). Skor
dinormalisasi min-maks (normalize_min_max(), sama seperti L1/BN/entropi
di run_ablation_single_criterion() -- TIDAK diimpor/diubah, logikanya
ditulis ulang di sini karena fungsi itu tidak mengenal kriteria ini).
Mengevaluasi VALIDASI *dan* TEST (beda dari skrip 03 yang hanya test).

Checkpoint memakai nama ablation_gmexp_{rasio}pct_30ep.pth -- mengikuti
konvensi penamaan skrip 03 dengan kriteria baru "gmexp" (tidak pernah
dipakai skrip manapun sebelumnya, sehingga tidak ada checkpoint lama
yang tertimpa, termasuk checkpoint ablation_gm_* milik
42_ablation_gm.py yang kriterianya berbeda). Ada logika skip-jika-ada
untuk resume.

Hasil disimpan progresif ke outputs/ablation_gm_expansion.json.

Setelah ketujuh rasio selesai, skrip membaca hasil L1 dan Entropi yang
SUDAH ADA (TIDAK dihitung ulang):
    - Test    : outputs/tabel_hasil_lengkap.json -> results.l1 / results.entropy
                (entropy = hasil kalibrasi TRAIN yang sudah diperbaiki/traincal)
    - Validasi: outputs/ablation_val_results.json -> ablation_val_results.{rasio}%.l1 / .entropy
lalu menampilkan tabel gabungan tiga kriteria (L1, Entropi, GM-Ekspansi).

Mode operasi:
  - TANPA --jalankan-eksperimen (bawaan): tampilkan rencana (rasio mana
    yang sudah punya checkpoint / belum) dan perkiraan durasi, lalu
    BERHENTI. Tidak melatih apa pun.
  - DENGAN --jalankan-eksperimen: menjalankan ketujuh rasio (skip yang
    checkpoint-nya sudah ada), lalu tabel gabungan.

Jalankan:
    venv/Scripts/python.exe scripts/43_ablation_gm_expansion.py
        (mode tinjau -- rencana + perkiraan durasi, tidak melatih apa pun)
    venv/Scripts/python.exe scripts/43_ablation_gm_expansion.py --jalankan-eksperimen
"""

import sys
import os
import json
import argparse
import random
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn

from src.config import CFG
from src.dataset import get_dataloaders
from src.model import train_model, evaluate_model, load_checkpoint
from src.pruning import normalize_min_max, get_pruning_mask, apply_pruning, _get_prunable_layers
from src.visualize import print_results_table

CRITERION_NAME = "gmexp"
RESULTS_PATH = CFG.OUTPUT_DIR / "ablation_gm_expansion.json"
TABEL_LENGKAP_PATH = CFG.OUTPUT_DIR / "tabel_hasil_lengkap.json"
VAL_RESULTS_PATH = CFG.OUTPUT_DIR / "ablation_val_results.json"
EPOCHS = CFG.BASELINE_EPOCHS  # 30
# Rerata menit/rasio dari eksperimen 30-epoch serupa (lihat 42_ablation_gm.py
# untuk rincian riwayatnya) -- perkiraan berbasis riwayat, bukan kalibrasi langsung.
MENIT_PER_RASIO_HISTORIS = (43.75 + 45.95 + 71.67) / 3


def set_seed(seed=CFG.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Identifikasi lapisan expansion -- DISALIN PERSIS dari
# 36_skor_redundansi.py/37/38, TIDAK DIUBAH.
# ============================================================

def cari_conv_ekspansi(block):
    """Cari conv 1x1 EKSPANSI (SEBELUM depthwise) di block.conv."""
    for layer in block.conv:
        if hasattr(layer, "__getitem__"):
            for sub in layer:
                if isinstance(sub, nn.Conv2d) and sub.kernel_size == (1, 1) and sub.groups == 1 \
                        and sub.out_channels > sub.in_channels:
                    return sub
    return None


# ============================================================
# Skor GM-Ekspansi: rumus "jumlah jarak berpasangan" (BEDA dari
# median geometrik di 36/37/38), sumber bobot EXPANSION CONV.
# ============================================================

def hitung_skor_gm_ekspansi_semua_layer(model):
    """
    s_GM(c) = sum_{j != c} ||W_c - W_j||_2, W_c = baris bobot expansion
    conv 1x1 channel c (diratakan). Skor besar = unik, skor kecil =
    redundan -- sudah selaras konvensi get_pruning_mask (skor terendah
    dipangkas), tidak dibalik.

    Args:
        model: MobileNetV2

    Returns:
        dict: {layer_idx: tensor skor GM-Ekspansi per channel}
    """
    prunable = _get_prunable_layers(model)
    gm_scores = {}

    for layer_idx, (block_idx, dw_module) in enumerate(prunable):
        block = model.features[block_idx]
        exp_conv = cari_conv_ekspansi(block)
        if exp_conv is None:
            print(f"[PERINGATAN] layer {layer_idx} (block {block_idx}): conv ekspansi "
                  f"tidak ditemukan, dilewati.")
            continue
        W = exp_conv.weight.data  # (mid_channels, in_channels, 1, 1)
        X = W.reshape(W.shape[0], -1).float()  # (mid_channels, in_channels), W_c diratakan

        # torch.cdist: jarak Euclidean (p=2) berpasangan, setara torch.norm
        # per pasangan tapi lebih hemat memori untuk in_channels besar
        # (sampai 160) -- diagonal = 0 otomatis (jarak channel ke dirinya
        # sendiri), memenuhi "j != c" tanpa masking eksplisit.
        jarak = torch.cdist(X, X, p=2)  # (mid_channels, mid_channels)
        skor = jarak.sum(dim=1)  # sum_{j != c} ||W_c - W_j||_2

        gm_scores[layer_idx] = skor.cpu()

    print(f"[GM-Ekspansi] Dihitung untuk {len(gm_scores)} lapisan (skip t=1)")
    return gm_scores


def run_ablation_gmexp_single(model_baseline, ratio):
    """Mirror run_ablation_single_criterion() di src/pruning.py UNTUK
    KRITERIA GM-EKSPANSI -- fungsi itu tidak mengenalnya, logikanya
    ditulis ulang di sini. src/pruning.py TIDAK diubah."""
    scores = hitung_skor_gm_ekspansi_semua_layer(model_baseline)
    scores = normalize_min_max(scores)
    masks = get_pruning_mask(scores, ratio)
    pruned_model = apply_pruning(model_baseline, masks)
    return pruned_model, masks


def run_single_ratio(model_baseline, dataloaders, dataset_sizes, ratio, device):
    ratio_persen = int(round(ratio * 100))
    ckpt_name = f"ablation_{CRITERION_NAME}_{ratio_persen}pct_{EPOCHS}ep.pth"
    ckpt_path = CFG.CHECKPOINT_DIR / ckpt_name

    if ckpt_path.exists():
        print(f"\n{'=' * 60}")
        print(f"SKIP: GM-Ekspansi | {ratio_persen}% | {EPOCHS}ep (checkpoint sudah ada: {ckpt_name})")
        print(f"{'=' * 60}")
        pruned_model, _ = load_checkpoint(ckpt_path, device)
    else:
        print(f"\n{'=' * 60}")
        print(f"ABLATION: GM-Ekspansi saja | Rasio {ratio_persen}% | {EPOCHS} epoch")
        print(f"{'=' * 60}")

        pruned_model, masks = run_ablation_gmexp_single(model_baseline, ratio)

        pruned_model, history = train_model(
            model=pruned_model,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            num_epochs=EPOCHS,
            lr=CFG.FINETUNE_LR,
            device=device,
            checkpoint_path=ckpt_path,
            phase_name=f"Ablation GM-Ekspansi {ratio_persen}%"
        )

    metrics_val, _, _ = evaluate_model(pruned_model, dataloaders["val"], device)
    print_results_table(metrics_val, f"GM-Ekspansi | {ratio_persen}% -- VALIDASI")

    metrics_test, _, _ = evaluate_model(pruned_model, dataloaders["test"], device)
    print_results_table(metrics_test, f"GM-Ekspansi | {ratio_persen}% -- TEST")

    return metrics_val, metrics_test, ckpt_name


def muat_l1_entropi_existing():
    """Baca hasil L1 dan Entropi (traincal) yang SUDAH ADA -- test dari
    tabel_hasil_lengkap.json, validasi dari ablation_val_results.json.
    TIDAK menghitung ulang atau melatih apa pun untuk keduanya."""
    hasil = {"l1": {}, "entropi": {}}

    if TABEL_LENGKAP_PATH.exists():
        with open(TABEL_LENGKAP_PATH, encoding="utf-8") as f:
            tabel = json.load(f)
        for ratio_key, entry in tabel.get("results", {}).get("l1", {}).items():
            hasil["l1"].setdefault(ratio_key, {})["test"] = entry["accuracy"]
        for ratio_key, entry in tabel.get("results", {}).get("entropy", {}).items():
            hasil["entropi"].setdefault(ratio_key, {})["test"] = entry["accuracy"]

    if VAL_RESULTS_PATH.exists():
        with open(VAL_RESULTS_PATH, encoding="utf-8") as f:
            val_data = json.load(f)
        for ratio_key, entry in val_data.get("ablation_val_results", {}).items():
            if "l1" in entry:
                hasil["l1"].setdefault(ratio_key, {})["val"] = entry["l1"]["accuracy"]
            if "entropy" in entry:
                hasil["entropi"].setdefault(ratio_key, {})["val"] = entry["entropy"]["accuracy"]

    return hasil


def cetak_tabel_gabungan(hasil_gmexp, existing):
    print(f"\n{'=' * 100}")
    print("TABEL GABUNGAN TIGA KRITERIA TUNGGAL: L1, ENTROPI (traincal), GM-EKSPANSI")
    print("=" * 100)
    header = (f"{'Rasio':>6} {'L1 Val':>9} {'L1 Test':>9} {'Entropi Val':>12} {'Entropi Test':>13} "
              f"{'GMexp Val':>10} {'GMexp Test':>11}")
    print(header)
    print("-" * 100)
    for ratio in CFG.PRUNING_RATIOS:
        ratio_key = f"{int(ratio * 100)}%"
        l1v = existing["l1"].get(ratio_key, {}).get("val")
        l1t = existing["l1"].get(ratio_key, {}).get("test")
        env = existing["entropi"].get(ratio_key, {}).get("val")
        ent = existing["entropi"].get(ratio_key, {}).get("test")
        gmv = hasil_gmexp.get(ratio_key, {}).get("validasi", {}).get("accuracy")
        gmt = hasil_gmexp.get(ratio_key, {}).get("test", {}).get("accuracy")

        def fmt(x):
            return f"{x*100:.2f}%" if x is not None else "N/A"

        print(f"{ratio_key:>6} {fmt(l1v):>9} {fmt(l1t):>9} {fmt(env):>12} {fmt(ent):>13} "
              f"{fmt(gmv):>10} {fmt(gmt):>11}")
    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Ablation study kriteria tunggal GM-Ekspansi (expansion conv, sum jarak berpasangan), 7 rasio"
    )
    parser.add_argument(
        "--jalankan-eksperimen", action="store_true",
        help="Jalankan (atau lanjutkan) ketujuh rasio. Tanpa flag ini, skrip HANYA "
             "menampilkan rencana dan perkiraan durasi, lalu berhenti."
    )
    args = parser.parse_args()

    ratios = CFG.PRUNING_RATIOS  # [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    print("=" * 70)
    print("ABLATION STUDY: GM-EKSPANSI SAJA (expansion conv, sum jarak berpasangan) -- 7 RASIO")
    print("=" * 70)
    print(f"  Rasio   : {[f'{r*100:.0f}%' for r in ratios]}")
    print(f"  Epoch   : {EPOCHS}")
    print(f"  Seed    : {CFG.SEED}")

    # ----- Rencana: rasio mana yang sudah ada checkpoint (resume) -----
    sudah_ada, belum_ada = [], []
    for ratio in ratios:
        ckpt_name = f"ablation_{CRITERION_NAME}_{int(ratio*100)}pct_{EPOCHS}ep.pth"
        (sudah_ada if (CFG.CHECKPOINT_DIR / ckpt_name).exists() else belum_ada).append(ratio)

    print(f"\n{'=' * 70}")
    print("RENCANA")
    print("=" * 70)
    print(f"  Checkpoint sudah ada (akan di-skip dari training, dievaluasi ulang saja): "
          f"{[f'{r*100:.0f}%' for r in sudah_ada] if sudah_ada else '(tidak ada)'}")
    print(f"  Akan dilatih (30 epoch)  : {[f'{r*100:.0f}%' for r in belum_ada] if belum_ada else '(tidak ada)'}")

    estimasi_menit = len(belum_ada) * MENIT_PER_RASIO_HISTORIS
    print(f"\n{'=' * 70}")
    print("PERKIRAAN DURASI")
    print("=" * 70)
    print(f"  Rasio akan dilatih     : {len(belum_ada)}")
    print(f"  Basis perkiraan        : rerata {MENIT_PER_RASIO_HISTORIS:.1f} menit/rasio "
          f"(riwayat 34/37/38_multikriteria_gm.py, mekanisme fine-tuning 30-epoch identik)")
    print(f"  Estimasi total         : {estimasi_menit:.0f} menit (~{estimasi_menit/60:.1f} jam)")
    print("  (Perkiraan berbasis riwayat, BUKAN kalibrasi langsung -- durasi aktual bisa berbeda.)")
    print("=" * 70)

    if not args.jalankan_eksperimen:
        print("\n[INFO] Mode TINJAU SAJA (tanpa --jalankan-eksperimen). Berhenti di sini.")
        print("[INFO] Tidak ada fine-tuning yang dijalankan, model baseline TIDAK dimuat.")
        print("[INFO] Jalankan ulang dengan --jalankan-eksperimen setelah konfirmasi diberikan.")
        return

    # ----- Jalankan -----
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SPLIT_DIR = CFG.DATASET_DIR.parent / "dataset_split"
    dataloaders, dataset_sizes = get_dataloaders(SPLIT_DIR)

    BASELINE_CKPT = CFG.CHECKPOINT_DIR / "baseline_9class.pth"
    if not BASELINE_CKPT.exists():
        print("[ERROR] Checkpoint baseline tidak ditemukan!")
        return
    model_baseline, _ = load_checkpoint(BASELINE_CKPT, device)
    model_baseline.eval()

    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, encoding="utf-8") as f:
            hasil_gmexp = json.load(f).get("results", {})
        print(f"\n[INFO] Melanjutkan dari hasil sebagian yang sudah ada: {RESULTS_PATH}")
    else:
        hasil_gmexp = {}

    start_time = time.time()
    for i, ratio in enumerate(ratios, 1):
        ratio_key = f"{int(ratio * 100)}%"
        elapsed = time.time() - start_time
        print(f"\n>>> Rasio {i}/{len(ratios)} | {ratio_key} | Waktu berjalan: {elapsed/60:.1f} menit")

        metrics_val, metrics_test, ckpt_name = run_single_ratio(
            model_baseline, dataloaders, dataset_sizes, ratio, device
        )
        hasil_gmexp[ratio_key] = {
            "checkpoint": ckpt_name,
            "validasi": {"accuracy": metrics_val["accuracy"], "f1_score": metrics_val["f1_score"]},
            "test": {"accuracy": metrics_test["accuracy"], "f1_score": metrics_test["f1_score"]},
            "num_params": metrics_test["num_params"],
            "model_size_mb": metrics_test["model_size_mb"],
            "flops": metrics_test["flops"],
            "inference_ms": metrics_test["inference_ms"],
        }

        save_data = {
            "criterion": "gm_expansion",
            "definisi": "s_GM(c) = sum_{j!=c} ||W_c - W_j||_2, dari bobot EXPANSION CONV 1x1 "
                        "(identifikasi lapisan disalin dari 36/37/38, rumus skor sama dengan "
                        "src/gm_scores.py tapi sumber bobot beda) -- beda dari GM depthwise "
                        "42_ablation_gm.py dan GM median-geometrik 36/37/38/40/41.",
            "epochs_used": EPOCHS,
            "seed": CFG.SEED,
            "results": hasil_gmexp,
            "status": f"{i}/{len(ratios)} rasio diproses",
        }
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Hasil sementara disimpan: {RESULTS_PATH}")

    total_time = time.time() - start_time
    save_data["status"] = "SELESAI"
    save_data["total_time_seconds"] = total_time
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print(f"[SELESAI] Ablation GM-Ekspansi saja -- total waktu: {total_time/60:.1f} menit "
          f"({total_time/3600:.2f} jam)")
    print(f"Hasil: {RESULTS_PATH}")
    print("=" * 70)

    # ----- Tabel gabungan tiga kriteria -----
    existing = muat_l1_entropi_existing()
    cetak_tabel_gabungan(hasil_gmexp, existing)


if __name__ == "__main__":
    main()
