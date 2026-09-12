"""
38_multikriteria_gm.py
Menguji komposisi skor gabungan BARU yang mengganti BN gamma (kriteria
paling lemah menurut 36_skor_redundansi.py/37_kriteria_redundansi_tunggal.py)
dengan skor redundansi GM:

    I(c) = w1 * S_norm_L1(c) + w2 * S_norm_Entropi(c) + w3 * S_norm_GM(c)

Beda dari I(c) resmi tesis (w1*L1 + w2*BN + w3*Entropi) -- BN tidak lagi
ikut, digantikan GM. Skrip ini TIDAK mengubah metodologi resmi, murni
eksperimen pembanding.

REVISI: versi pertama skrip ini menurunkan w1/w2/w3 dari akurasi validasi
ketiga kriteria tunggal (rumus rasio-terhadap-total), yang untuk rasio
40%/50% menghasilkan bobot dekat sepertiga-sepertiga -- KONFIGURASI ITU
TIDAK DIJALANKAN. Diganti dengan TIGA KONFIGURASI BOBOT MANUAL, HANYA
pada rasio 40%:

    A = (w1=0.50, w2=0.00, w3=0.50)  -- L1+GM saja, tanpa entropi
    B = (w1=0.45, w2=0.10, w3=0.45)  -- seperti A, entropi porsi kecil
    C = (w1=0.40, w2=0.20, w3=0.40)  -- entropi porsi sedang

Sumber angka: manual, ditentukan pengguna (bukan diturunkan dari akurasi
validasi kriteria tunggal). Ketiga bobot tetap w1+w2+w3=1, w_k>=0
(disyaratkan compute_importance_scores).

Skor GM dihitung dengan fungsi yang SAMA PERSIS dengan 37 (disalin
verbatim, TIDAK diubah, 37 sendiri TIDAK disentuh): cari_conv_ekspansi,
hitung_median_geometrik, hitung_skor_gm -- dari bobot conv 1x1 EKSPANSI
(sebelum depthwise). GM dipakai APA ADANYA (TIDAK dibalik tandanya --
jarak kecil ke median geometrik = redundan = HARUS dipangkas = HARUS
berskor rendah, sudah selaras dengan konvensi get_pruning_mask), lalu
dinormalisasi min-maks per lapisan seperti L1/Entropi (normalize_min_max
dari src/pruning.py, via compute_importance_scores()).

I(c) dihitung dengan MEMANGGIL compute_importance_scores() di
src/pruning.py APA ADANYA (fungsi generik: w1*norm(arg1) + w2*norm(arg2)
+ w3*norm(arg3)), tapi argumen posisi ke-2 dan ke-3 diisi entropy_scores
dan gm_scores (BUKAN bn_scores) -- src/pruning.py TIDAK diubah, hanya
urutan argumen di titik panggil yang disesuaikan. AKIBATNYA baris log
"[WSM] Bobot: ... w2(BN)=... w3(H)=..." yang dicetak fungsi itu SALAH
LABEL untuk pemanggilan ini -- w2(BN) sebenarnya bobot Entropi, dan
w3(H) sebenarnya bobot GM. Skrip ini mencetak baris klarifikasi eksplisit
tepat sebelum tiap pemanggilan.

Rasio: HANYA 40% (sesuai permintaan revisi -- 50% tidak diuji di sini).

Prosedur fine-tuning SAMA PERSIS dengan 12_multicriteria_valweights.py:
30 epoch (CFG.BASELINE_EPOCHS), CFG.FINETUNE_LR, Adam+StepLR
(train_model()), seed=CFG.SEED (42) dipanggil SEKALI di awal sebelum
loop konfigurasi -- keterbatasan metodologis yang sama berlaku (CLAUDE.md
Bagian 9: RNG saat fine-tuning konfigurasi B bergantung pada apa yang
terjadi selama konfigurasi A, dst.). Evaluasi VALIDASI *dan* TEST
terpisah (validasi untuk keputusan metodologis, test hanya pelaporan).

get_pruning_mask() memangkas rasio SERAGAM per lapisan terlepas dari
skor, sehingga jumlah parameter akhir pada rasio yang sama HARUS identik
dengan ablation_l1 -- dikonfirmasi eksplisit terhadap
outputs/tabel_hasil_lengkap.json.

Checkpoint memakai akhiran _config{A,B,C} (mis.
multikriteria_gm_configA_40pct_30ep.pth), TIDAK menimpa checkpoint
manapun yang sudah ada. Hasil disimpan ke
outputs/multikriteria_gm_results.json (baru; kalau berkas versi
sebelumnya -- hasil bobot-otomatis yang tidak jadi dijalankan -- ada,
TIDAK dipakai/diwarisi, skrip ini menimpa dengan struktur baru berbasis
konfigurasi A/B/C).

Mode operasi:
  - TANPA --jalankan-eksperimen (bawaan): tampilkan tabel ketiga
    konfigurasi bobot, lalu persentase channel yang dipilih BERBEDA
    antara mask tiap konfigurasi dengan mask L1 murni dan mask GM murni
    (rasio 40%). Kalau KETIGA konfigurasi berbeda < 25% dari L1 murni,
    cetak PERINGATAN eksplisit (indikasi ketiga konfigurasi terlalu mirip
    L1 murni untuk jadi uji yang informatif). BERHENTI di sini dalam
    kedua kasus -- tidak melatih apa pun, menunggu keputusan pengguna.
  - DENGAN --jalankan-eksperimen: menjalankan ketiga konfigurasi (fine-tuning penuh).

Jalankan:
    venv/Scripts/python.exe scripts/38_multikriteria_gm.py
        (mode tinjau -- tabel bobot + perbedaan mask, tidak melatih apa pun)
    venv/Scripts/python.exe scripts/38_multikriteria_gm.py --jalankan-eksperimen
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
    compute_l1_scores, compute_entropy_scores,
    compute_importance_scores, normalize_min_max, get_pruning_mask, apply_pruning,
    _get_prunable_layers,
)
from src.visualize import plot_confusion_matrix, print_results_table

CHECKPOINT_NAME = "baseline_9class.pth"
TABEL_LENGKAP_PATH = CFG.OUTPUT_DIR / "tabel_hasil_lengkap.json"
VAL_RESULTS_PATH = CFG.OUTPUT_DIR / "ablation_val_results.json"
REDUNDANSI_RESULTS_PATH = CFG.OUTPUT_DIR / "kriteria_redundansi_tunggal_results.json"
RESULTS_PATH = CFG.OUTPUT_DIR / "multikriteria_gm_results.json"
RATIO_TUNGGAL = 0.4
EPOCHS = CFG.BASELINE_EPOCHS  # 30
AMBANG_PERBEDAAN_PERSEN = 25.0

KONFIGURASI_MANUAL = {
    "A": {"w1_l1": 0.50, "w2_entropi": 0.00, "w3_gm": 0.50,
          "deskripsi": "Dua kriteria terkuat dan paling berbeda, tanpa entropi."},
    "B": {"w1_l1": 0.45, "w2_entropi": 0.10, "w3_gm": 0.45,
          "deskripsi": "Sama seperti A tetapi entropi diberi porsi kecil."},
    "C": {"w1_l1": 0.40, "w2_entropi": 0.20, "w3_gm": 0.40,
          "deskripsi": "Entropi diberi porsi sedang."},
}
# Konfigurasi D TIDAK dihardcode di sini -- dihitung di main() lewat
# hitung_bobot_otomatis_dari_akurasi_val() supaya presisi penuh (bukan
# angka yang sudah dibulatkan 4 digit, yang jumlahnya 0.3388+0.3264+0.3347
# = 0.9999 != 1.0 dan akan GAGAL di assert compute_importance_scores).


def set_seed(seed=CFG.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# DISALIN PERSIS dari 37_kriteria_redundansi_tunggal.py (dan itu sendiri
# disalin dari 36_skor_redundansi.py), TIDAK DIUBAH -- supaya skor GM
# identik. GM dipakai APA ADANYA di skrip ini (tidak dibalik).
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


def hitung_median_geometrik(X, maks_iterasi=200, toleransi=1e-6, epsilon=1e-8):
    """Median geometrik (algoritma Weiszfeld) dari X (n x d)."""
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


def hitung_skor_gm_semua_layer(model):
    prunable = _get_prunable_layers(model)
    gm_scores = {}
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
    return gm_scores


# ============================================================
# Bagian skrip ini
# ============================================================

def muat_akurasi_val_l1_entropi(rasio_persen):
    if not VAL_RESULTS_PATH.exists():
        return None, None
    with open(VAL_RESULTS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    entry = data.get("ablation_val_results", {}).get(f"{rasio_persen}%")
    if entry is None:
        return None, None
    return entry.get("l1", {}).get("accuracy"), entry.get("entropy", {}).get("accuracy")


def muat_akurasi_val_gm(rasio_persen):
    if not REDUNDANSI_RESULTS_PATH.exists():
        return None
    with open(REDUNDANSI_RESULTS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    entry = data.get("results", {}).get("gm", {}).get(f"{rasio_persen}%")
    return entry["validasi"]["accuracy"] if entry else None


def hitung_bobot_otomatis_dari_akurasi_val(rasio_persen):
    """Konfigurasi D: w_k = akurasi_k / sum(akurasi) -- rumus SAMA dengan
    compute_optimal_weights() di src/pruning.py, dihitung ULANG di sini
    dari sumber data mentah (BUKAN memakai angka yang sudah dibulatkan 4
    digit desimal, yang jumlahnya 0.9999 bukan 1.0 dan akan GAGAL di
    assert w1+w2+w3==1 milik compute_importance_scores). Mengembalikan
    None kalau salah satu sumber akurasi tidak ditemukan."""
    acc_l1, acc_entropi = muat_akurasi_val_l1_entropi(rasio_persen)
    acc_gm = muat_akurasi_val_gm(rasio_persen)
    if acc_l1 is None or acc_entropi is None or acc_gm is None:
        return None
    total = acc_l1 + acc_entropi + acc_gm
    return {
        "w1_l1": acc_l1 / total, "w2_entropi": acc_entropi / total, "w3_gm": acc_gm / total,
        "acc_l1_val": acc_l1, "acc_entropi_val": acc_entropi, "acc_gm_val": acc_gm,
    }


def muat_num_params_l1(rasio_persen):
    if not TABEL_LENGKAP_PATH.exists():
        return None
    with open(TABEL_LENGKAP_PATH, encoding="utf-8") as f:
        tabel = json.load(f)
    entry = tabel.get("results", {}).get("l1", {}).get(f"{rasio_persen}%")
    return entry["num_params"] if entry else None


def hitung_persentase_beda_mask(mask_a, mask_b):
    total_channel = 0
    total_beda = 0
    for idx in sorted(mask_a):
        a, b = mask_a[idx], mask_b[idx]
        total_beda += int((a != b).sum())
        total_channel += len(a)
    persen = (total_beda / total_channel * 100) if total_channel else 0.0
    return total_beda, total_channel, persen


def run_single(label, cfg, l1_scores, entropy_scores, gm_scores,
                model_baseline, dataloaders, dataset_sizes, device):
    ratio_persen = int(round(RATIO_TUNGGAL * 100))
    ckpt_name = f"multikriteria_gm_config{label}_{ratio_persen}pct_{EPOCHS}ep.pth"
    ckpt_path = CFG.CHECKPOINT_DIR / ckpt_name
    w1, w2, w3 = cfg["w1_l1"], cfg["w2_entropi"], cfg["w3_gm"]

    print(f"\n[INFO] Konfigurasi {label}: compute_importance_scores(l1, entropi, gm, "
          f"w1={w1:.2f}, w2={w2:.2f}, w3={w3:.2f}) -- label 'w2(BN)' dan 'w3(H)' di baris "
          f"[WSM] berikut SEBENARNYA berarti bobot Entropi dan GM.")
    importance_scores = compute_importance_scores(l1_scores, entropy_scores, gm_scores, w1=w1, w2=w2, w3=w3)
    masks = get_pruning_mask(importance_scores, RATIO_TUNGGAL)

    if ckpt_path.exists():
        print(f"\n{'=' * 60}")
        print(f"SKIP: Konfigurasi {label} (checkpoint sudah ada: {ckpt_name})")
        print(f"{'=' * 60}")
        pruned_model, _ = load_checkpoint(ckpt_path, device)
        num_params = count_parameters(pruned_model)
    else:
        print(f"\n{'=' * 60}")
        print(f"PRUNING MULTI-KRITERIA-GM KONFIGURASI {label}: {ratio_persen}%")
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
            phase_name=f"Multi-Kriteria-GM Konfigurasi {label} {ratio_persen}%"
        )

    print(f"\n[INFO] Evaluasi VALIDASI -- Konfigurasi {label}")
    metrics_val, _, _ = evaluate_model(pruned_model, dataloaders["val"], device)
    print_results_table(metrics_val, f"Multi-Kriteria-GM Konfigurasi {label} -- VALIDASI")

    print(f"\n[INFO] Evaluasi TEST -- Konfigurasi {label}")
    metrics_test, _, _ = evaluate_model(pruned_model, dataloaders["test"], device)
    print_results_table(metrics_test, f"Multi-Kriteria-GM Konfigurasi {label} -- TEST")

    plot_confusion_matrix(
        metrics_test["confusion_matrix"],
        title=f"CM Multi-Kriteria-GM Konfigurasi {label} TEST (Acc: {metrics_test['accuracy']*100:.2f}%)",
        save_path=CFG.OUTPUT_DIR / f"cm_multikriteria_gm_config{label}_{ratio_persen}pct.png"
    )

    num_params_l1 = muat_num_params_l1(ratio_persen)
    if num_params_l1 is not None:
        identik = (num_params == num_params_l1)
        catatan = (f"IDENTIK dengan ablation_l1 ({num_params_l1:,})" if identik
                   else f"TIDAK IDENTIK -- config{label}={num_params:,} vs ablation_l1={num_params_l1:,}")
        print(f"\n[INFO] Jumlah parameter: {num_params:,} -- {catatan}")
    else:
        identik = None
        catatan = f"Tidak bisa dikonfirmasi -- {TABEL_LENGKAP_PATH} tidak punya entri ablation_l1."
        print(f"\n[PERINGATAN] {catatan}")

    return {
        "deskripsi": cfg["deskripsi"],
        "bobot": {"w1_l1": w1, "w2_entropi": w2, "w3_gm": w3},
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
        description="I(c) = w1*L1 + w2*Entropi + w3*GM, tiga konfigurasi bobot manual, rasio 40%"
    )
    parser.add_argument(
        "--jalankan-eksperimen", action="store_true",
        help="Jalankan ketiga konfigurasi (fine-tuning penuh). Tanpa flag ini, skrip HANYA "
             "menampilkan tabel bobot dan perbedaan mask, lalu berhenti."
    )
    args = parser.parse_args()

    device = torch.device("cpu")
    ckpt_path = CFG.CHECKPOINT_DIR / CHECKPOINT_NAME
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint tidak ditemukan: {ckpt_path}")
        return

    print("=" * 70)
    print("MULTI-KRITERIA-GM: I(c) = w1*L1 + w2*Entropi + w3*GM -- 3 KONFIGURASI MANUAL, rasio 40%")
    print("=" * 70)

    model_baseline, _ = load_checkpoint(ckpt_path, device)
    model_baseline.eval()

    SPLIT_DIR = CFG.DATASET_DIR.parent / "dataset_split"
    dataloaders, dataset_sizes = get_dataloaders(SPLIT_DIR)

    print("\n[INFO] Menghitung skor L1, entropi, dan GM...")
    l1_scores = compute_l1_scores(model_baseline)
    entropy_scores = compute_entropy_scores(
        model_baseline, dataloaders["train"], device, samples_per_class=20, seed=CFG.SEED
    )
    gm_scores = hitung_skor_gm_semua_layer(model_baseline)

    # ----- Konfigurasi D: bobot diturunkan dari akurasi validasi (bukan hardcode) -----
    rasio_persen_tunggal = int(round(RATIO_TUNGGAL * 100))
    bobot_d = hitung_bobot_otomatis_dari_akurasi_val(rasio_persen_tunggal)
    konfigurasi_lengkap = dict(KONFIGURASI_MANUAL)
    if bobot_d is not None:
        konfigurasi_lengkap["D"] = {
            "w1_l1": bobot_d["w1_l1"], "w2_entropi": bobot_d["w2_entropi"], "w3_gm": bobot_d["w3_gm"],
            "deskripsi": (
                f"Diturunkan dari rasio akurasi validasi (L1={bobot_d['acc_l1_val']*100:.2f}%, "
                f"Entropi={bobot_d['acc_entropi_val']*100:.2f}%, GM={bobot_d['acc_gm_val']*100:.2f}%) "
                f"pada rasio {rasio_persen_tunggal}% -- rumus w_k=acc_k/sum(acc), sama dengan "
                f"compute_optimal_weights; ini bobot yang dihitung versi PERTAMA skrip ini, "
                f"sebelum diganti konfigurasi manual A/B/C."
            ),
        }
    else:
        print(f"\n[PERINGATAN] Data akurasi validasi sumber (L1/Entropi di {VAL_RESULTS_PATH}, "
              f"GM di {REDUNDANSI_RESULTS_PATH}) tidak lengkap untuk rasio {rasio_persen_tunggal}%. "
              f"Konfigurasi D DILEWATI, hanya A/B/C yang diproses.")

    # ----- 1. Tabel konfigurasi bobot -----
    print(f"\n{'=' * 100}")
    print(f"1. KONFIGURASI BOBOT (rasio {rasio_persen_tunggal}%)")
    print("=" * 100)
    header = f"{'Konfig':>7} {'w1 (L1)':>9} {'w2 (Entropi)':>13} {'w3 (GM)':>9}   Deskripsi"
    print(header)
    print("-" * 100)
    for label, cfg in konfigurasi_lengkap.items():
        print(f"{label:>7} {cfg['w1_l1']:>9.2f} {cfg['w2_entropi']:>13.2f} {cfg['w3_gm']:>9.2f}   "
              f"{cfg['deskripsi']}")
    print("=" * 100)

    # ----- 2. Perbedaan mask vs L1 murni dan GM murni (rasio 40%) -----
    mask_l1 = get_pruning_mask(normalize_min_max(l1_scores), RATIO_TUNGGAL)
    mask_gm = get_pruning_mask(normalize_min_max(gm_scores), RATIO_TUNGGAL)

    print(f"\n{'=' * 100}")
    print("2. PERBEDAAN PILIHAN CHANNEL PADA RASIO 40% (vs mask tiap konfigurasi)")
    print("=" * 100)
    print(f"{'Konfig':>7} {'vs L1 murni':>13} {'vs GM murni':>13}")
    print("-" * 100)

    hasil_mask = {}
    for label, cfg in konfigurasi_lengkap.items():
        w1, w2, w3 = cfg["w1_l1"], cfg["w2_entropi"], cfg["w3_gm"]
        importance = compute_importance_scores(l1_scores, entropy_scores, gm_scores, w1=w1, w2=w2, w3=w3)
        mask_cfg = get_pruning_mask(importance, RATIO_TUNGGAL)
        _, total_ch, persen_l1 = hitung_persentase_beda_mask(mask_cfg, mask_l1)
        _, _, persen_gm = hitung_persentase_beda_mask(mask_cfg, mask_gm)
        hasil_mask[label] = {"persen_vs_l1": persen_l1, "persen_vs_gm": persen_gm}
        print(f"{label:>7} {persen_l1:>12.2f}% {persen_gm:>12.2f}%")
    print("-" * 100)
    print(f"(Total channel: {total_ch})")
    print("=" * 100)

    semua_di_bawah_ambang = all(
        hasil_mask[label]["persen_vs_l1"] < AMBANG_PERBEDAAN_PERSEN for label in konfigurasi_lengkap
    )
    if semua_di_bawah_ambang:
        rincian_persen = ", ".join(
            f"{label}={hasil_mask[label]['persen_vs_l1']:.2f}%" for label in konfigurasi_lengkap
        )
        print(f"\n[PERINGATAN] SEMUA konfigurasi ({', '.join(konfigurasi_lengkap)}) memilih channel "
              f"yang berbeda KURANG DARI {AMBANG_PERBEDAAN_PERSEN:.0f}% dibanding L1 murni "
              f"({rincian_persen}). "
              f"Indikasi konfigurasi-konfigurasi ini terlalu mirip pemangkasan L1 murni untuk jadi "
              f"uji yang informatif. Menunggu keputusan Anda sebelum melatih.")

    if not args.jalankan_eksperimen:
        print("\n[INFO] Mode TINJAU SAJA (tanpa --jalankan-eksperimen). Berhenti di sini.")
        print("[INFO] Tidak ada fine-tuning yang dijalankan.")
        print("[INFO] Jalankan ulang dengan --jalankan-eksperimen setelah keputusan diambil.")
        return

    # ----- Jalankan seluruh konfigurasi (A/B/C sudah punya checkpoint -> skip -----
    # latih ulang, dievaluasi ulang saja secara deterministik; D baru -> dilatih) -----
    set_seed()

    all_results = {
        "checkpoint_baseline": CHECKPOINT_NAME,
        "formula": "I(c) = w1*S_norm_L1 + w2*S_norm_Entropi + w3*S_norm_GM (menggantikan BN)",
        "rasio": RATIO_TUNGGAL,
        "konfigurasi_bobot": konfigurasi_lengkap,
        "pemeriksaan_mask_40pct": hasil_mask,
        "semua_konfigurasi_di_bawah_ambang_25persen_vs_l1": semua_di_bawah_ambang,
        "seed": CFG.SEED,
        "epochs": EPOCHS,
        "catatan_evaluasi": "accuracy pada 'validasi' untuk keputusan metodologis; 'test' hanya "
                             "untuk pelaporan, TIDAK dipakai untuk memilih apa pun.",
        "results": {},
        "status": "BERJALAN",
    }

    start_time = time.time()
    for i, (label, cfg) in enumerate(konfigurasi_lengkap.items(), 1):
        elapsed = time.time() - start_time
        print(f"\n>>> Eksperimen {i}/{len(konfigurasi_lengkap)} | Konfigurasi {label} "
              f"| Waktu berjalan: {elapsed/60:.1f} menit")

        hasil = run_single(
            label, cfg, l1_scores, entropy_scores, gm_scores,
            model_baseline, dataloaders, dataset_sizes, device
        )
        all_results["results"][label] = hasil
        all_results["status"] = f"{i}/{len(konfigurasi_lengkap)} selesai"

        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Hasil sementara disimpan: {RESULTS_PATH}")

    total_time = time.time() - start_time
    all_results["status"] = "SELESAI"
    all_results["total_time_seconds"] = total_time
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 90}")
    print(f"RINGKASAN -- MULTI-KRITERIA-GM, {len(konfigurasi_lengkap)} KONFIGURASI "
          f"({EPOCHS} epoch, seed={CFG.SEED})")
    print("=" * 90)
    print(f"{'Konfig':<8} {'Val Acc':<10} {'Test Acc':<10} {'#Param':<12} {'Params=L1?':<12}")
    print("-" * 90)
    for label, r in all_results["results"].items():
        print(f"{label:<8} {r['validasi']['accuracy']*100:>7.2f}%  {r['test']['accuracy']*100:>7.2f}%  "
              f"{r['num_params']:>10,}  {'YA' if r['num_params_identik_dengan_l1'] else 'TIDAK':<12}")
    print("=" * 90)
    print(f"Total waktu: {total_time/60:.1f} menit ({total_time/3600:.2f} jam)")
    print(f"\n[INFO] Hasil disimpan: {RESULTS_PATH}")
    print("[INFO] Tidak ada checkpoint/hasil lama yang ditimpa.")


if __name__ == "__main__":
    main()