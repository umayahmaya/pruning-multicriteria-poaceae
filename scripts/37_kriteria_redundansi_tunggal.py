"""
37_kriteria_redundansi_tunggal.py
Menguji skor redundansi GM dan COS (didefinisikan di 36_skor_redundansi.py)
sebagai KRITERIA TUNGGAL pemangkasan -- kasus batas I(c) analog dengan
run_ablation_single_criterion() di src/pruning.py (satu bobot=1, dua
lainnya=0) -- untuk mengetahui apakah informasi redundansi bernilai
SEBELUM dipertimbangkan masuk ke skor gabungan I(c).

Empat eksperimen: GM murni 40%, GM murni 50%, COS murni 40%, COS murni 50%.

Fungsi penghitung skor GM/COS (cari_conv_ekspansi, hitung_median_geometrik,
hitung_skor_gm, hitung_skor_cos) DISALIN PERSIS dari 36_skor_redundansi.py
TIDAK DIUBAH -- supaya skornya identik dengan yang dianalisis di sana.
36_skor_redundansi.py sendiri TIDAK disentuh (instruksi: jangan ubah
skrip lain).

KONVENSI ARAH SKOR (WAJIB DIBACA) -- get_pruning_mask() di src/pruning.py
memangkas channel dengan skor TERENDAH pada tiap lapisan:
  - GM  : dipakai APA ADANYA (tidak dibalik). Jarak kecil ke median
          geometrik = redundan (lihat interpretasi FPGM di 36) = HARUS
          dipangkas = HARUS mendapat skor rendah. Arahnya sudah selaras
          dengan konvensi get_pruning_mask, tidak perlu tindakan apa pun.
  - COS : DIBALIK TANDANYA (dipakai sebagai -COS, BUKAN COS). COS tinggi
          = mirip banyak channel lain = redundan = HARUS dipangkas =
          HARUS mendapat skor rendah. Kalau COS dipakai apa adanya,
          get_pruning_mask justru memangkas channel dengan COS TERENDAH
          (paling TIDAK redundan/paling unik) -- arah terbalik dari yang
          dimaksud. Pembalikan tanda ini dilakukan SEBELUM normalisasi
          min-maks di bangun_skor_untuk_mask().

Setelah arah skor benar, dilakukan normalisasi min-maks PER LAPISAN
(normalize_min_max() dari src/pruning.py, sama seperti L1/BN/entropi di
run_ablation_single_criterion()) sebelum get_pruning_mask(). Untuk
kriteria TUNGGAL, normalisasi ini adalah transformasi linear naik per
lapisan sehingga TIDAK mengubah urutan/mask yang dihasilkan -- dilakukan
tetap demi konsistensi metodologis dan supaya skor akhir sebanding
dengan skala [0,1] yang dipakai kriteria lain, bukan karena mengubah
hasil pruning.

Prosedur fine-tuning SAMA PERSIS dengan 12_multicriteria_valweights.py:
30 epoch (CFG.BASELINE_EPOCHS), CFG.FINETUNE_LR, Adam+StepLR
(train_model()), seed=CFG.SEED (42) dipanggil SEKALI di awal main()
sebelum loop -- keterbatasan metodologis yang sama berlaku (CLAUDE.md
Bagian 9: RNG saat fine-tuning satu kombinasi bergantung pada kombinasi
sebelumnya dalam urutan proses).

Mengevaluasi VALIDASI *dan* TEST secara terpisah (validasi untuk
keputusan metodologis, test hanya untuk pelaporan -- sama seperti
34_multicriteria_beta20.py).

get_pruning_mask() memangkas rasio SERAGAM per lapisan terlepas dari
skor/kriteria, sehingga jumlah parameter akhir pada rasio yang sama
HARUS identik dengan ablation_l1 -- dikonfirmasi eksplisit terhadap
outputs/tabel_hasil_lengkap.json.

Checkpoint memakai akhiran _gm/_cos (mis. redundansi_gm_40pct_30ep.pth),
TIDAK menimpa checkpoint manapun yang sudah ada. Hasil disimpan ke
outputs/kriteria_redundansi_tunggal_results.json (baru, terpisah).

Mode operasi:
  - TANPA --jalankan-eksperimen (bawaan): hitung skor GM/COS/L1, tampilkan
    KONVENSI ARAH secara eksplisit, tampilkan persentase channel yang
    dipilih BERBEDA antara mask GM, mask COS, dan mask L1 pada rasio 40%
    (pemeriksaan sebelum menjalankan apa pun), lalu tampilkan perkiraan
    durasi, dan BERHENTI -- tidak melatih apa pun.
  - DENGAN --jalankan-eksperimen: menjalankan keempat eksperimen.

Jalankan:
    venv/Scripts/python.exe scripts/37_kriteria_redundansi_tunggal.py
        (mode tinjau -- pemeriksaan mask + perkiraan durasi, tidak melatih apa pun)
    venv/Scripts/python.exe scripts/37_kriteria_redundansi_tunggal.py --jalankan-eksperimen
"""

import sys
import os
import json
import argparse
import random
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np

from src.config import CFG
from src.dataset import get_dataloaders
from src.model import train_model, evaluate_model, load_checkpoint, count_parameters
from src.pruning import (
    compute_l1_scores, normalize_min_max, get_pruning_mask, apply_pruning,
    _get_prunable_layers,
)
from src.visualize import plot_confusion_matrix, print_results_table

CHECKPOINT_NAME = "baseline_9class.pth"
TABEL_LENGKAP_PATH = CFG.OUTPUT_DIR / "tabel_hasil_lengkap.json"
RESULTS_PATH = CFG.OUTPUT_DIR / "kriteria_redundansi_tunggal_results.json"
RATIOS = [0.4, 0.5]
KRITERIA_LIST = ["gm", "cos"]
EPOCHS = CFG.BASELINE_EPOCHS  # 30
RATIO_PRATINJAU = 0.4
# Perkiraan menit/rasio dari 34_multicriteria_beta20.py (mekanisme fine-tuning
# IDENTIK: get_pruning_mask rasio seragam + train_model 30 epoch) -- total
# 87.5 menit untuk 2 rasio (50%,60%) -> rerata 43.75 menit/rasio.
MENIT_PER_RASIO_HISTORIS = 87.5 / 2


def set_seed(seed=CFG.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# DISALIN PERSIS dari 36_skor_redundansi.py (TIDAK DIUBAH) --
# supaya skor GM/COS identik dengan yang dianalisis di sana.
# ============================================================

def cari_conv_ekspansi(block):
    """Cari conv 1x1 EKSPANSI (SEBELUM depthwise) di block.conv. Kriteria
    deteksi identik dengan _get_prunable_layers()/apply_pruning() di
    src/pruning.py: kernel 1x1, groups=1, out_channels > in_channels
    (beda dari conv proyeksi 1x1 yang out_channels < in_channels)."""
    for layer in block.conv:
        if hasattr(layer, "__getitem__"):
            for sub in layer:
                if isinstance(sub, nn.Conv2d) and sub.kernel_size == (1, 1) and sub.groups == 1 \
                        and sub.out_channels > sub.in_channels:
                    return sub
    return None


def hitung_median_geometrik(X, maks_iterasi=200, toleransi=1e-6, epsilon=1e-8):
    """Median geometrik (algoritma Weiszfeld) dari X (n x d). Lihat
    36_skor_redundansi.py untuk penjelasan lengkap."""
    y = X.mean(dim=0)
    for _ in range(maks_iterasi):
        jarak = torch.norm(X - y, dim=1) + epsilon
        bobot = 1.0 / jarak
        y_baru = (X * bobot.unsqueeze(1)).sum(dim=0) / bobot.sum()
        if torch.norm(y_baru - y) < toleransi:
            y = y_baru
            break
        y = y_baru
    return y


def hitung_skor_gm(X):
    """GM[c] = jarak Euclidean vektor channel c ke median geometrik
    seluruh channel di lapisan yang sama."""
    median = hitung_median_geometrik(X)
    return torch.norm(X - median, dim=1)


def hitung_skor_cos(X):
    """COS[c] = rata-rata kemiripan kosinus channel c terhadap seluruh
    channel lain (bukan termasuk dirinya) di lapisan yang sama."""
    norm = X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    X_norm = X / norm
    sim = X_norm @ X_norm.T
    n = X.shape[0]
    if n <= 1:
        return torch.zeros(n)
    total_tanpa_diri_sendiri = sim.sum(dim=1) - 1.0
    return total_tanpa_diri_sendiri / (n - 1)


def hitung_skor_gm_cos_semua_layer(model):
    """Loop atas seluruh lapisan yang bisa dipangkas, hitung GM & COS
    mentah (BELUM dibalik/dinormalisasi) dari bobot conv ekspansi."""
    prunable = _get_prunable_layers(model)
    gm_scores, cos_scores = {}, {}
    for layer_idx, (block_idx, dw_module) in enumerate(prunable):
        block = model.features[block_idx]
        exp_conv = cari_conv_ekspansi(block)
        if exp_conv is None:
            print(f"[PERINGATAN] layer {layer_idx} (block {block_idx}): conv ekspansi "
                  f"tidak ditemukan, dilewati.")
            continue
        W = exp_conv.weight.data
        X = W.reshape(W.shape[0], -1).float()
        gm_scores[layer_idx] = hitung_skor_gm(X)
        cos_scores[layer_idx] = hitung_skor_cos(X)
    return gm_scores, cos_scores


# ============================================================
# Bagian baru skrip ini (bukan salinan)
# ============================================================

def bangun_skor_untuk_mask(kriteria, gm_scores, cos_scores):
    """Terapkan KONVENSI ARAH SKOR (lihat docstring modul) lalu
    normalize_min_max() per lapisan, siap dipakai get_pruning_mask()."""
    if kriteria == "gm":
        mentah = gm_scores  # dipakai apa adanya, TIDAK dibalik
    elif kriteria == "cos":
        mentah = {idx: -skor for idx, skor in cos_scores.items()}  # DIBALIK
    else:
        raise ValueError(f"Kriteria tidak dikenal: {kriteria}")
    return normalize_min_max(mentah)


def hitung_persentase_beda_mask(mask_a, mask_b):
    total_channel = 0
    total_beda = 0
    for idx in sorted(mask_a):
        a, b = mask_a[idx], mask_b[idx]
        total_beda += int((a != b).sum())
        total_channel += len(a)
    persen = (total_beda / total_channel * 100) if total_channel else 0.0
    return total_beda, total_channel, persen


def muat_num_params_l1(rasio_persen):
    """num_params ablation_l1 pada rasio yang sama, untuk konfirmasi
    kesetaraan jumlah parameter (get_pruning_mask memangkas rasio
    SERAGAM per lapisan terlepas dari kriteria/skor)."""
    if not TABEL_LENGKAP_PATH.exists():
        return None
    with open(TABEL_LENGKAP_PATH, encoding="utf-8") as f:
        tabel = json.load(f)
    entry = tabel.get("results", {}).get("l1", {}).get(f"{rasio_persen}%")
    return entry["num_params"] if entry else None


def run_single(kriteria, ratio, skor_mask, model_baseline, dataloaders, dataset_sizes, device):
    ratio_persen = int(round(ratio * 100))
    ratio_key = f"{ratio_persen}%"
    label = kriteria.upper()
    ckpt_name = f"redundansi_{kriteria}_{ratio_persen}pct_{EPOCHS}ep.pth"
    ckpt_path = CFG.CHECKPOINT_DIR / ckpt_name

    masks = get_pruning_mask(skor_mask, ratio)

    if ckpt_path.exists():
        print(f"\n{'=' * 60}")
        print(f"SKIP: {label} {ratio_key} (checkpoint sudah ada: {ckpt_name})")
        print(f"{'=' * 60}")
        pruned_model, _ = load_checkpoint(ckpt_path, device)
        num_params = count_parameters(pruned_model)
    else:
        print(f"\n{'=' * 60}")
        print(f"PRUNING KRITERIA TUNGGAL {label}: {ratio_key}")
        print(f"{'=' * 60}")

        pruned_model_sebelum_ft = apply_pruning(model_baseline, masks)
        num_params = count_parameters(pruned_model_sebelum_ft)
        print(f"[INFO] Jumlah parameter setelah pemangkasan (sebelum fine-tuning): {num_params:,}")

        pruned_model, history = train_model(
            model=pruned_model_sebelum_ft,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            num_epochs=EPOCHS,
            lr=CFG.FINETUNE_LR,
            device=device,
            checkpoint_path=ckpt_path,
            phase_name=f"Redundansi-{label} {ratio_key}"
        )

    print(f"\n[INFO] Evaluasi VALIDASI -- {label} {ratio_key}")
    metrics_val, _, _ = evaluate_model(pruned_model, dataloaders["val"], device)
    print_results_table(metrics_val, f"Redundansi-{label} {ratio_key} -- VALIDASI")

    print(f"\n[INFO] Evaluasi TEST -- {label} {ratio_key}")
    metrics_test, _, _ = evaluate_model(pruned_model, dataloaders["test"], device)
    print_results_table(metrics_test, f"Redundansi-{label} {ratio_key} -- TEST")

    plot_confusion_matrix(
        metrics_test["confusion_matrix"],
        title=f"CM Redundansi-{label} {ratio_key} TEST (Acc: {metrics_test['accuracy']*100:.2f}%)",
        save_path=CFG.OUTPUT_DIR / f"cm_redundansi_{kriteria}_{ratio_persen}pct.png"
    )

    num_params_l1 = muat_num_params_l1(ratio_persen)
    if num_params_l1 is not None:
        identik = (num_params == num_params_l1)
        catatan = (f"IDENTIK dengan ablation_l1 ({num_params_l1:,})" if identik
                   else f"TIDAK IDENTIK -- {label}={num_params:,} vs ablation_l1={num_params_l1:,}")
        print(f"\n[INFO] Jumlah parameter: {num_params:,} -- {catatan}")
    else:
        identik = None
        catatan = f"Tidak bisa dikonfirmasi -- {TABEL_LENGKAP_PATH} tidak punya entri ablation_l1 untuk {ratio_key}."
        print(f"\n[PERINGATAN] {catatan}")

    return {
        "checkpoint": ckpt_name,
        "validasi": {
            "accuracy": metrics_val["accuracy"], "precision": metrics_val["precision"],
            "recall": metrics_val["recall"], "f1_score": metrics_val["f1_score"],
        },
        "test": {
            "accuracy": metrics_test["accuracy"], "precision": metrics_test["precision"],
            "recall": metrics_test["recall"], "f1_score": metrics_test["f1_score"],
        },
        "num_params": num_params,
        "model_size_mb": metrics_test["model_size_mb"],
        "flops": metrics_test["flops"],
        "inference_ms": metrics_test["inference_ms"],
        "num_params_l1_rasio_sama": num_params_l1,
        "num_params_identik_dengan_l1": identik,
        "catatan_perbandingan_params": catatan,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Kriteria tunggal redundansi (GM, COS) pada rasio 40%/50%"
    )
    parser.add_argument(
        "--jalankan-eksperimen", action="store_true",
        help="Jalankan keempat eksperimen (fine-tuning penuh). Tanpa flag ini, skrip HANYA "
             "menampilkan konvensi arah skor, pemeriksaan perbedaan mask (rasio 40%%), dan "
             "perkiraan durasi, lalu berhenti."
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    ckpt_path = CFG.CHECKPOINT_DIR / CHECKPOINT_NAME
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint tidak ditemukan: {ckpt_path}")
        return

    print("=" * 70)
    print("KRITERIA TUNGGAL REDUNDANSI: GM DAN COS (rasio 40%, 50%)")
    print("=" * 70)

    model_baseline, _ = load_checkpoint(ckpt_path, device)
    model_baseline.eval()

    print("\n[INFO] Menghitung skor L1 (pembanding) dan GM/COS (dari conv ekspansi, tanpa kalibrasi)...")
    l1_scores = compute_l1_scores(model_baseline)
    gm_scores, cos_scores = hitung_skor_gm_cos_semua_layer(model_baseline)

    skor_untuk_mask = {
        "gm": bangun_skor_untuk_mask("gm", gm_scores, cos_scores),
        "cos": bangun_skor_untuk_mask("cos", gm_scores, cos_scores),
    }
    l1_untuk_mask = normalize_min_max(l1_scores)

    print(f"\n{'=' * 70}")
    print("KONVENSI ARAH SKOR (get_pruning_mask memangkas skor TERENDAH)")
    print("=" * 70)
    print("  GM  : dipakai APA ADANYA (jarak kecil ke median geometrik = redundan = dipangkas)")
    print("  COS : DIBALIK TANDANYA, dipakai sebagai -COS (COS tinggi = redundan = harus")
    print("        berskor rendah setelah dibalik, supaya ikut terpangkas)")
    print("=" * 70)

    # ----- Pemeriksaan SEBELUM menjalankan apa pun: perbedaan mask pada 40% -----
    print(f"\n{'=' * 70}")
    print(f"PEMERIKSAAN SEBELUM EKSPERIMEN -- rasio {RATIO_PRATINJAU*100:.0f}%: "
          f"perbedaan pilihan channel antar mask")
    print("=" * 70)
    mask_gm = get_pruning_mask(skor_untuk_mask["gm"], RATIO_PRATINJAU)
    mask_cos = get_pruning_mask(skor_untuk_mask["cos"], RATIO_PRATINJAU)
    mask_l1 = get_pruning_mask(l1_untuk_mask, RATIO_PRATINJAU)

    _, total_ch, persen_gm_cos = hitung_persentase_beda_mask(mask_gm, mask_cos)
    _, _, persen_gm_l1 = hitung_persentase_beda_mask(mask_gm, mask_l1)
    _, _, persen_cos_l1 = hitung_persentase_beda_mask(mask_cos, mask_l1)

    print(f"  Total channel (di seluruh 16 lapisan): {total_ch}")
    print(f"  GM  vs COS : {persen_gm_cos:.2f}% channel dipilih berbeda")
    print(f"  GM  vs L1  : {persen_gm_l1:.2f}% channel dipilih berbeda")
    print(f"  COS vs L1  : {persen_cos_l1:.2f}% channel dipilih berbeda")
    print("=" * 70)

    # ----- Perkiraan durasi -----
    total_rasio = len(KRITERIA_LIST) * len(RATIOS)
    estimasi_menit = total_rasio * MENIT_PER_RASIO_HISTORIS
    print(f"\n{'=' * 70}")
    print("PERKIRAAN DURASI")
    print("=" * 70)
    print(f"  Eksperimen             : {total_rasio} (kriteria {KRITERIA_LIST} x rasio {RATIOS})")
    print(f"  Basis perkiraan        : rerata {MENIT_PER_RASIO_HISTORIS:.1f} menit/rasio dari "
          f"34_multicriteria_beta20.py (mekanisme fine-tuning identik: get_pruning_mask rasio "
          f"seragam + train_model 30 epoch)")
    print(f"  Estimasi total         : {estimasi_menit:.0f} menit (~{estimasi_menit/60:.1f} jam)")
    print("  (Perkiraan berbasis riwayat, BUKAN kalibrasi langsung -- durasi aktual bisa")
    print("   berbeda tergantung beban CPU saat itu.)")
    print("=" * 70)

    if not args.jalankan_eksperimen:
        print("\n[INFO] Mode TINJAU SAJA (tanpa --jalankan-eksperimen). Berhenti di sini.")
        print("[INFO] Tidak ada fine-tuning yang dijalankan.")
        print("[INFO] Jalankan ulang dengan --jalankan-eksperimen untuk menjalankan keempat eksperimen.")
        return

    # ----- Jalankan keempat eksperimen -----
    set_seed()
    SPLIT_DIR = CFG.DATASET_DIR.parent / "dataset_split"
    dataloaders, dataset_sizes = get_dataloaders(SPLIT_DIR)

    all_results = {
        "checkpoint_baseline": CHECKPOINT_NAME,
        "konvensi_arah_skor": {
            "gm": "dipakai apa adanya (jarak kecil ke median geometrik = redundan = dipangkas)",
            "cos": "dibalik tandanya (-COS); COS tinggi = redundan = harus berskor rendah "
                   "setelah dibalik supaya ikut terpangkas",
        },
        "pemeriksaan_mask_40pct": {
            "total_channel": total_ch,
            "gm_vs_cos_persen_beda": persen_gm_cos,
            "gm_vs_l1_persen_beda": persen_gm_l1,
            "cos_vs_l1_persen_beda": persen_cos_l1,
        },
        "seed": CFG.SEED,
        "epochs": EPOCHS,
        "catatan_evaluasi": "accuracy pada 'validasi' untuk keputusan metodologis; 'test' hanya "
                             "untuk pelaporan, TIDAK dipakai untuk memilih apa pun.",
        "results": {},
        "status": "BERJALAN",
    }

    start_time = time.time()
    i = 0
    for kriteria in KRITERIA_LIST:
        for ratio in RATIOS:
            i += 1
            ratio_key = f"{int(ratio * 100)}%"
            elapsed = time.time() - start_time
            print(f"\n>>> Eksperimen {i}/{total_rasio} | {kriteria.upper()} {ratio_key} "
                  f"| Waktu berjalan: {elapsed/60:.1f} menit")

            hasil = run_single(
                kriteria, ratio, skor_untuk_mask[kriteria],
                model_baseline, dataloaders, dataset_sizes, device
            )
            all_results["results"].setdefault(kriteria, {})[ratio_key] = hasil
            all_results["status"] = f"{i}/{total_rasio} selesai"

            with open(RESULTS_PATH, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"[INFO] Hasil sementara disimpan: {RESULTS_PATH}")

    total_time = time.time() - start_time
    all_results["status"] = "SELESAI"
    all_results["total_time_seconds"] = total_time
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 90}")
    print(f"RINGKASAN -- KRITERIA TUNGGAL REDUNDANSI ({EPOCHS} epoch, seed={CFG.SEED})")
    print("=" * 90)
    print(f"{'Kriteria':<10} {'Rasio':<8} {'Val Acc':<10} {'Test Acc':<10} {'#Param':<12} {'Params=L1?':<12}")
    print("-" * 90)
    for kriteria in KRITERIA_LIST:
        for ratio_key, r in all_results["results"].get(kriteria, {}).items():
            print(f"{kriteria.upper():<10} {ratio_key:<8} {r['validasi']['accuracy']*100:>7.2f}%  "
                  f"{r['test']['accuracy']*100:>7.2f}%  {r['num_params']:>10,}  "
                  f"{'YA' if r['num_params_identik_dengan_l1'] else 'TIDAK':<12}")
    print("=" * 90)
    print(f"Total waktu: {total_time/60:.1f} menit ({total_time/3600:.2f} jam)")
    print(f"\n[INFO] Hasil disimpan: {RESULTS_PATH}")
    print("[INFO] Tidak ada checkpoint/hasil lama yang ditimpa.")


if __name__ == "__main__":
    main()