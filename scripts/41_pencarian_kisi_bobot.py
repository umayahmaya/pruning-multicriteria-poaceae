"""
41_pencarian_kisi_bobot.py
Pencarian kisi (grid) bobot secara sistematis pada simpleks w1+w2+w3=1
dengan langkah 0,25, untuk komposisi I(c) = w1*L1 + w2*Entropi + w3*GM
(GM menggantikan BN, sama seperti 38_multikriteria_gm.py), HANYA pada
rasio 40%.

Kisi lengkap: 15 kombinasi (semua (i,j,k) kelipatan 0,25 dengan
i+j+k=1), urutan PERSIS seperti diberikan di permintaan:
    G01 (1.00,0.00,0.00)   G06 (0.50,0.00,0.50)   G11 (0.00,1.00,0.00)
    G02 (0.75,0.25,0.00)   G07 (0.25,0.75,0.00)   G12 (0.00,0.75,0.25)
    G03 (0.75,0.00,0.25)   G08 (0.25,0.50,0.25)   G13 (0.00,0.50,0.50)
    G04 (0.50,0.50,0.00)   G09 (0.25,0.25,0.50)   G14 (0.00,0.25,0.75)
    G05 (0.50,0.25,0.25)   G10 (0.25,0.00,0.75)   G15 (0.00,0.00,1.00)

EMPAT titik SUDAH punya hasil, TIDAK dilatih ulang -- dibaca dari berkas
hasil yang tersimpan:
    G01 (1.00,0.00,0.00) = ablation L1 rasio 40%
        val:  outputs/ablation_val_results.json -> ablation_val_results.40%.l1
        test: outputs/tabel_hasil_lengkap.json  -> results.l1.40%
    G11 (0.00,1.00,0.00) = ablation entropi TRAINCAL rasio 40%
        val:  outputs/ablation_val_results.json -> ablation_val_results.40%.entropy
        test: outputs/tabel_hasil_lengkap.json  -> results.entropy.40%
              (diverifikasi: checkpoint "ablation_entropy_40pct_30ep_traincal.pth",
              BUKAN entropi lama yang dikalibrasi val)
    G15 (0.00,0.00,1.00) = GM tunggal rasio 40% dari 37_kriteria_redundansi_tunggal.py
        val & test: outputs/kriteria_redundansi_tunggal_results.json -> results.gm.40%
    G06 (0.50,0.00,0.50) = Konfigurasi A dari 38_multikriteria_gm.py
        val & test: outputs/multikriteria_gm_results.json -> results.A

SEBELAS titik sisanya DILATIH. Prosedur SAMA PERSIS dengan
38_multikriteria_gm.py: skor L1/entropi (kalibrasi train, 20 citra/kelas,
seed=CFG.SEED)/GM dihitung SEKALI dari checkpoints/baseline_9class.pth
(fungsi GM disalin verbatim dari 36/37/38, TIDAK diubah), I(c) dibangun
lewat compute_importance_scores(l1, entropi, gm, w1, w2, w3) -- argumen
posisi ke-2/ke-3 diisi entropi/GM BUKAN bn (src/pruning.py TIDAK diubah,
label "[WSM] w2(BN)"/"w3(H)" yang tercetak SEBENARNYA berarti bobot
Entropi/GM, sama seperti 38), get_pruning_mask() rasio seragam per
lapisan, fine-tuning 30 epoch (CFG.BASELINE_EPOCHS), CFG.FINETUNE_LR,
Adam+StepLR (train_model()), seed=CFG.SEED (42) dipanggil SEKALI sebelum
loop -- keterbatasan metodologis yang sama berlaku (CLAUDE.md Bagian 9).
Evaluasi VALIDASI *dan* TEST terpisah.

DUA titik TAMBAHAN di luar kisi (langkah bukan 0,25), TIDAK dilatih
ulang, murni pembanding: Konfigurasi B (0.45,0.10,0.45) dan D
(0.3388,0.3264,0.3347) dari 38_multikriteria_gm.py -- keduanya juga
dibaca dari outputs/multikriteria_gm_results.json.

get_pruning_mask() memangkas rasio SERAGAM per lapisan terlepas dari
skor, sehingga jumlah parameter akhir HARUS identik dengan ablation_l1
pada rasio yang sama -- dikonfirmasi eksplisit terhadap
outputs/tabel_hasil_lengkap.json untuk kesebelas titik yang dilatih.

Pemilihan/kesimpulan HANYA berdasarkan akurasi VALIDASI. Akurasi TEST
dilaporkan untuk ketransparanan tapi TIDAK dipakai memutuskan apa pun
(sesuai aturan metodologis proyek -- CLAUDE.md butir 4.1).

Checkpoint memakai akhiran kisi_bobot_{label} (mis.
kisi_bobot_G02_40pct_30ep.pth), TIDAK menimpa checkpoint manapun yang
sudah ada -- termasuk checkpoint milik keempat titik preloaded (tidak
disentuh sama sekali). Hasil disimpan ke
outputs/pencarian_kisi_bobot_results.json (baru).

Mode operasi:
  - TANPA --jalankan-eksperimen (bawaan): baca & tampilkan konfirmasi
    keempat titik preloaded (+ dua titik tambahan di luar kisi) beserta
    akurasi validasinya, tampilkan perkiraan durasi untuk melatih 11
    titik sisanya, lalu BERHENTI. Tidak melatih apa pun, TIDAK memuat
    model baseline sama sekali (murni baca JSON).
  - DENGAN --jalankan-eksperimen: menjalankan kesebelas titik yang belum
    ada hasilnya (fine-tuning penuh).

Jalankan:
    venv/Scripts/python.exe scripts/41_pencarian_kisi_bobot.py
        (mode tinjau -- konfirmasi titik preloaded + perkiraan durasi, tidak melatih apa pun)
    venv/Scripts/python.exe scripts/41_pencarian_kisi_bobot.py --jalankan-eksperimen
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
    compute_importance_scores, get_pruning_mask, apply_pruning,
    _get_prunable_layers,
)
from src.visualize import plot_confusion_matrix, print_results_table

CHECKPOINT_NAME = "baseline_9class.pth"
TABEL_LENGKAP_PATH = CFG.OUTPUT_DIR / "tabel_hasil_lengkap.json"
VAL_RESULTS_PATH = CFG.OUTPUT_DIR / "ablation_val_results.json"
REDUNDANSI_RESULTS_PATH = CFG.OUTPUT_DIR / "kriteria_redundansi_tunggal_results.json"
MULTIKRITERIA_GM_RESULTS_PATH = CFG.OUTPUT_DIR / "multikriteria_gm_results.json"
RESULTS_PATH = CFG.OUTPUT_DIR / "pencarian_kisi_bobot_results.json"

RATIO = 0.4
RATIO_PERSEN = 40
EPOCHS = CFG.BASELINE_EPOCHS  # 30
# Rerata menit/kombinasi dari 34_multicriteria_beta20.py (43.75),
# 37_kriteria_redundansi_tunggal.py (45.95), 38_multikriteria_gm.py run
# pertama (71.67) -- mekanisme fine-tuning identik (get_pruning_mask
# rasio seragam + train_model 30 epoch).
MENIT_PER_KOMBINASI_HISTORIS = (43.75 + 45.95 + 71.67) / 3

GRID_BOBOT = [
    ("G01", (1.00, 0.00, 0.00)),
    ("G02", (0.75, 0.25, 0.00)),
    ("G03", (0.75, 0.00, 0.25)),
    ("G04", (0.50, 0.50, 0.00)),
    ("G05", (0.50, 0.25, 0.25)),
    ("G06", (0.50, 0.00, 0.50)),
    ("G07", (0.25, 0.75, 0.00)),
    ("G08", (0.25, 0.50, 0.25)),
    ("G09", (0.25, 0.25, 0.50)),
    ("G10", (0.25, 0.00, 0.75)),
    ("G11", (0.00, 1.00, 0.00)),
    ("G12", (0.00, 0.75, 0.25)),
    ("G13", (0.00, 0.50, 0.50)),
    ("G14", (0.00, 0.25, 0.75)),
    ("G15", (0.00, 0.00, 1.00)),
]
TITIK_PRELOADED = {"G01", "G06", "G11", "G15"}
TITIK_TAMBAHAN_DILUAR_KISI = [
    ("B", (0.45, 0.10, 0.45)),
    ("D", (0.3388, 0.3264, 0.3347)),
]


def set_seed(seed=CFG.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# DISALIN PERSIS dari 36_skor_redundansi.py (dan 37/38), TIDAK DIUBAH.
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
# Sumber titik preloaded / tambahan
# ============================================================

def muat_l1_ablation_40():
    if not (VAL_RESULTS_PATH.exists() and TABEL_LENGKAP_PATH.exists()):
        return None
    with open(VAL_RESULTS_PATH, encoding="utf-8") as f:
        val_data = json.load(f)
    with open(TABEL_LENGKAP_PATH, encoding="utf-8") as f:
        tabel = json.load(f)
    val_entry = val_data.get("ablation_val_results", {}).get(f"{RATIO_PERSEN}%", {}).get("l1")
    test_entry = tabel.get("results", {}).get("l1", {}).get(f"{RATIO_PERSEN}%")
    if val_entry is None or test_entry is None:
        return None
    return {
        "validasi": {"accuracy": val_entry["accuracy"], "f1_score": val_entry["f1_score"]},
        "test": {"accuracy": test_entry["accuracy"], "f1_score": test_entry["f1_score"]},
        "num_params": test_entry["num_params"],
        "checkpoint": test_entry.get("checkpoint", "ablation_l1_40pct_30ep.pth"),
        "sumber": f"{VAL_RESULTS_PATH.name} (val) + {TABEL_LENGKAP_PATH.name} (test)",
    }


def muat_entropi_traincal_40():
    if not (VAL_RESULTS_PATH.exists() and TABEL_LENGKAP_PATH.exists()):
        return None
    with open(VAL_RESULTS_PATH, encoding="utf-8") as f:
        val_data = json.load(f)
    with open(TABEL_LENGKAP_PATH, encoding="utf-8") as f:
        tabel = json.load(f)
    val_entry = val_data.get("ablation_val_results", {}).get(f"{RATIO_PERSEN}%", {}).get("entropy")
    test_entry = tabel.get("results", {}).get("entropy", {}).get(f"{RATIO_PERSEN}%")
    if val_entry is None or test_entry is None:
        return None
    ckpt = test_entry.get("checkpoint", "")
    if "traincal" not in ckpt:
        print(f"[PERINGATAN] Checkpoint entropi di {TABEL_LENGKAP_PATH.name} ({ckpt}) tidak "
              f"mengandung 'traincal' -- periksa apakah ini benar hasil kalibrasi train yang "
              f"sudah diperbaiki.")
    return {
        "validasi": {"accuracy": val_entry["accuracy"], "f1_score": val_entry["f1_score"]},
        "test": {"accuracy": test_entry["accuracy"], "f1_score": test_entry["f1_score"]},
        "num_params": test_entry["num_params"],
        "checkpoint": ckpt,
        "sumber": f"{VAL_RESULTS_PATH.name} (val) + {TABEL_LENGKAP_PATH.name} (test)",
    }


def muat_gm_tunggal_40():
    if not REDUNDANSI_RESULTS_PATH.exists():
        return None
    with open(REDUNDANSI_RESULTS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    entry = data.get("results", {}).get("gm", {}).get(f"{RATIO_PERSEN}%")
    if entry is None:
        return None
    return {
        "validasi": {"accuracy": entry["validasi"]["accuracy"], "f1_score": entry["validasi"]["f1_score"]},
        "test": {"accuracy": entry["test"]["accuracy"], "f1_score": entry["test"]["f1_score"]},
        "num_params": entry["num_params"],
        "checkpoint": entry["checkpoint"],
        "sumber": REDUNDANSI_RESULTS_PATH.name,
    }


def muat_konfigurasi_gm_38(label):
    if not MULTIKRITERIA_GM_RESULTS_PATH.exists():
        return None
    with open(MULTIKRITERIA_GM_RESULTS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    entry = data.get("results", {}).get(label)
    if entry is None:
        return None
    return {
        "validasi": {"accuracy": entry["validasi"]["accuracy"], "f1_score": entry["validasi"]["f1_score"]},
        "test": {"accuracy": entry["test"]["accuracy"], "f1_score": entry["test"]["f1_score"]},
        "num_params": entry["num_params"],
        "checkpoint": entry["checkpoint"],
        "sumber": f"{MULTIKRITERIA_GM_RESULTS_PATH.name} (konfigurasi {label})",
    }


def muat_titik_preloaded():
    """Return dict {label: hasil-atau-None} untuk G01/G06/G11/G15."""
    return {
        "G01": muat_l1_ablation_40(),
        "G06": muat_konfigurasi_gm_38("A"),
        "G11": muat_entropi_traincal_40(),
        "G15": muat_gm_tunggal_40(),
    }


def muat_titik_tambahan():
    hasil = {}
    for label, _w in TITIK_TAMBAHAN_DILUAR_KISI:
        hasil[label] = muat_konfigurasi_gm_38(label)
    return hasil


def muat_num_params_l1(rasio_persen):
    if not TABEL_LENGKAP_PATH.exists():
        return None
    with open(TABEL_LENGKAP_PATH, encoding="utf-8") as f:
        tabel = json.load(f)
    entry = tabel.get("results", {}).get("l1", {}).get(f"{rasio_persen}%")
    return entry["num_params"] if entry else None


def run_single(label, w, l1_scores, entropy_scores, gm_scores,
                model_baseline, dataloaders, dataset_sizes, device):
    w1, w2, w3 = w
    ckpt_name = f"kisi_bobot_{label}_{RATIO_PERSEN}pct_{EPOCHS}ep.pth"
    ckpt_path = CFG.CHECKPOINT_DIR / ckpt_name

    print(f"\n[INFO] {label}: compute_importance_scores(l1, entropi, gm, "
          f"w1={w1:.4f}, w2={w2:.4f}, w3={w3:.4f}) -- label 'w2(BN)'/'w3(H)' di baris "
          f"[WSM] berikut SEBENARNYA berarti bobot Entropi dan GM.")
    importance_scores = compute_importance_scores(l1_scores, entropy_scores, gm_scores, w1=w1, w2=w2, w3=w3)
    masks = get_pruning_mask(importance_scores, RATIO)

    if ckpt_path.exists():
        print(f"\n{'=' * 60}")
        print(f"SKIP: {label} (checkpoint sudah ada: {ckpt_name})")
        print(f"{'=' * 60}")
        pruned_model, _ = load_checkpoint(ckpt_path, device)
        num_params = count_parameters(pruned_model)
    else:
        print(f"\n{'=' * 60}")
        print(f"PRUNING KISI BOBOT {label}: w=({w1:.4f},{w2:.4f},{w3:.4f}) rasio {RATIO_PERSEN}%")
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
            phase_name=f"Kisi-Bobot {label}"
        )

    print(f"\n[INFO] Evaluasi VALIDASI -- {label}")
    metrics_val, _, _ = evaluate_model(pruned_model, dataloaders["val"], device)
    print_results_table(metrics_val, f"Kisi-Bobot {label} -- VALIDASI")

    print(f"\n[INFO] Evaluasi TEST -- {label}")
    metrics_test, _, _ = evaluate_model(pruned_model, dataloaders["test"], device)
    print_results_table(metrics_test, f"Kisi-Bobot {label} -- TEST")

    plot_confusion_matrix(
        metrics_test["confusion_matrix"],
        title=f"CM Kisi-Bobot {label} TEST (Acc: {metrics_test['accuracy']*100:.2f}%)",
        save_path=CFG.OUTPUT_DIR / f"cm_kisi_bobot_{label}_{RATIO_PERSEN}pct.png"
    )

    num_params_l1 = muat_num_params_l1(RATIO_PERSEN)
    if num_params_l1 is not None:
        identik = (num_params == num_params_l1)
        catatan = (f"IDENTIK dengan ablation_l1 ({num_params_l1:,})" if identik
                   else f"TIDAK IDENTIK -- {label}={num_params:,} vs ablation_l1={num_params_l1:,}")
    else:
        identik = None
        catatan = f"Tidak bisa dikonfirmasi -- {TABEL_LENGKAP_PATH} tidak punya entri ablation_l1."
    print(f"\n[INFO] Jumlah parameter: {num_params:,} -- {catatan}")

    return {
        "bobot": list(w),
        "dilatih_ulang": True,
        "checkpoint": ckpt_name,
        "sumber": "dilatih di skrip ini",
        "validasi": {"accuracy": metrics_val["accuracy"], "f1_score": metrics_val["f1_score"]},
        "test": {"accuracy": metrics_test["accuracy"], "f1_score": metrics_test["f1_score"]},
        "num_params": num_params,
        "num_params_l1_rasio_sama": num_params_l1,
        "num_params_identik_dengan_l1": identik,
        "catatan_perbandingan_params": catatan,
    }


def cetak_tabel_konfirmasi(judul, label_bobot_list, hasil_dict):
    print(f"\n{'=' * 100}")
    print(judul)
    print("=" * 100)
    print(f"{'Label':>8} {'Bobot (w1,w2,w3)':>22} {'Akurasi Val':>13} {'Akurasi Test':>14}  Sumber")
    print("-" * 100)
    semua_ok = True
    for label, w in label_bobot_list:
        h = hasil_dict.get(label)
        if h is None:
            print(f"{label:>8} {str(w):>22} {'GAGAL DIBACA':>13} {'':>14}")
            semua_ok = False
        else:
            print(f"{label:>8} {str(w):>22} {h['validasi']['accuracy']*100:>12.2f}% "
                  f"{h['test']['accuracy']*100:>13.2f}%  {h['sumber']}")
    print("=" * 100)
    return semua_ok


def main():
    parser = argparse.ArgumentParser(
        description="Pencarian kisi bobot w1+w2+w3=1 langkah 0.25 (L1/Entropi/GM), rasio 40%"
    )
    parser.add_argument(
        "--jalankan-eksperimen", action="store_true",
        help="Latih 11 titik kisi yang belum ada hasilnya. Tanpa flag ini, skrip HANYA "
             "menampilkan konfirmasi titik preloaded dan perkiraan durasi, lalu berhenti."
    )
    args = parser.parse_args()

    print("=" * 70)
    print("PENCARIAN KISI BOBOT w1+w2+w3=1, LANGKAH 0.25 (L1, ENTROPI, GM), RASIO 40%")
    print("=" * 70)

    # ----- Baca 4 titik preloaded (TIDAK perlu memuat model apa pun) -----
    preloaded = muat_titik_preloaded()
    label_bobot_preloaded = [(l, w) for l, w in GRID_BOBOT if l in TITIK_PRELOADED]
    semua_ok = cetak_tabel_konfirmasi(
        "KONFIRMASI 4 TITIK PRELOADED (TIDAK dilatih ulang)",
        label_bobot_preloaded, preloaded
    )
    if not semua_ok:
        print("\n[ERROR] Ada titik preloaded yang GAGAL dibaca. Perbaiki sumber data dulu "
              "sebelum melanjutkan -- skrip berhenti.")
        return

    # ----- Baca 2 titik tambahan di luar kisi -----
    tambahan = muat_titik_tambahan()
    cetak_tabel_konfirmasi(
        "TITIK TAMBAHAN DI LUAR KISI (pembanding, TIDAK dilatih ulang)",
        TITIK_TAMBAHAN_DILUAR_KISI, tambahan
    )

    # ----- Perkiraan durasi -----
    jumlah_dilatih = len(GRID_BOBOT) - len(TITIK_PRELOADED)
    estimasi_menit = jumlah_dilatih * MENIT_PER_KOMBINASI_HISTORIS
    print(f"\n{'=' * 70}")
    print("PERKIRAAN DURASI")
    print("=" * 70)
    print(f"  Titik kisi total       : {len(GRID_BOBOT)}")
    print(f"  Sudah ada (tidak dilatih ulang): {len(TITIK_PRELOADED)}")
    print(f"  Akan dilatih           : {jumlah_dilatih}")
    print(f"  Basis perkiraan        : rerata {MENIT_PER_KOMBINASI_HISTORIS:.1f} menit/kombinasi "
          f"dari 34/37/38_multikriteria_gm.py (43.75, 45.95, 71.67 menit/kombinasi)")
    print(f"  Estimasi total         : {estimasi_menit:.0f} menit (~{estimasi_menit/60:.1f} jam)")
    print("  (Perkiraan berbasis riwayat, BUKAN kalibrasi langsung -- durasi aktual bisa berbeda.)")
    print("=" * 70)

    if not args.jalankan_eksperimen:
        print("\n[INFO] Mode TINJAU SAJA (tanpa --jalankan-eksperimen). Berhenti di sini.")
        print("[INFO] Tidak ada fine-tuning yang dijalankan, model baseline TIDAK dimuat.")
        print("[INFO] Jalankan ulang dengan --jalankan-eksperimen untuk melatih 11 titik sisanya.")
        return

    # ----- Jalankan 11 titik yang belum ada hasilnya -----
    device = torch.device("cpu")
    ckpt_path = CFG.CHECKPOINT_DIR / CHECKPOINT_NAME
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint tidak ditemukan: {ckpt_path}")
        return

    model_baseline, _ = load_checkpoint(ckpt_path, device)
    model_baseline.eval()

    SPLIT_DIR = CFG.DATASET_DIR.parent / "dataset_split"
    dataloaders, dataset_sizes = get_dataloaders(SPLIT_DIR)

    print("\n[INFO] Menghitung skor L1, entropi, dan GM (sekali, dipakai ulang seluruh titik)...")
    l1_scores = compute_l1_scores(model_baseline)
    entropy_scores = compute_entropy_scores(
        model_baseline, dataloaders["train"], device, samples_per_class=20, seed=CFG.SEED
    )
    gm_scores = hitung_skor_gm_semua_layer(model_baseline)

    set_seed()

    all_results = {
        "checkpoint_baseline": CHECKPOINT_NAME,
        "formula": "I(c) = w1*S_norm_L1 + w2*S_norm_Entropi + w3*S_norm_GM (menggantikan BN)",
        "rasio": RATIO,
        "langkah_kisi": 0.25,
        "seed": CFG.SEED,
        "epochs": EPOCHS,
        "catatan_evaluasi": "accuracy pada 'validasi' untuk keputusan metodologis; 'test' hanya "
                             "untuk pelaporan, TIDAK dipakai untuk memilih apa pun.",
        "titik_tambahan_di_luar_kisi": tambahan,
        "grid": {},
        "status": "BERJALAN",
    }

    # Isi titik preloaded dulu (tidak dilatih)
    for label, w in GRID_BOBOT:
        if label in TITIK_PRELOADED:
            entry = dict(preloaded[label])
            entry["bobot"] = list(w)
            entry["dilatih_ulang"] = False
            all_results["grid"][label] = entry

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    label_to_train = [(l, w) for l, w in GRID_BOBOT if l not in TITIK_PRELOADED]
    start_time = time.time()
    for i, (label, w) in enumerate(label_to_train, 1):
        elapsed = time.time() - start_time
        print(f"\n>>> Eksperimen {i}/{len(label_to_train)} | {label} w={w} "
              f"| Waktu berjalan: {elapsed/60:.1f} menit")

        hasil = run_single(
            label, w, l1_scores, entropy_scores, gm_scores,
            model_baseline, dataloaders, dataset_sizes, device
        )
        all_results["grid"][label] = hasil
        all_results["status"] = f"{i}/{len(label_to_train)} titik baru selesai"

        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Hasil sementara disimpan: {RESULTS_PATH}")

    total_time = time.time() - start_time
    all_results["status"] = "SELESAI"
    all_results["total_time_seconds"] = total_time
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # ----- Ringkasan terurut menurut akurasi VALIDASI -----
    terurut = sorted(all_results["grid"].items(), key=lambda kv: kv[1]["validasi"]["accuracy"], reverse=True)

    print(f"\n{'=' * 100}")
    print(f"RINGKASAN -- KISI BOBOT, 15 TITIK, DIURUTKAN MENURUT AKURASI VALIDASI "
          f"({EPOCHS} epoch, seed={CFG.SEED})")
    print("=" * 100)
    print(f"{'Peringkat':>9} {'Label':>7} {'Bobot (w1,w2,w3)':>22} {'Val Acc':>10} {'Test Acc':>10} "
          f"{'Dilatih?':>9} {'Params=L1?':>11}")
    print("-" * 100)
    for i, (label, r) in enumerate(terurut, 1):
        w = r["bobot"]
        params_ok = r.get("num_params_identik_dengan_l1")
        params_str = "YA" if params_ok else ("TIDAK" if params_ok is False else "N/A")
        print(f"{i:>9} {label:>7} {str(tuple(round(x,4) for x in w)):>22} "
              f"{r['validasi']['accuracy']*100:>9.2f}% {r['test']['accuracy']*100:>9.2f}% "
              f"{'YA' if r['dilatih_ulang'] else 'tidak':>9} {params_str:>11}")
    print("=" * 100)

    label_terbaik, r_terbaik = terurut[0]
    print(f"\n>> KOMBINASI TERTINGGI PADA VALIDASI: {label_terbaik} "
          f"w={tuple(round(x,4) for x in r_terbaik['bobot'])} "
          f"-- akurasi validasi = {r_terbaik['validasi']['accuracy']*100:.2f}% "
          f"(akurasi test = {r_terbaik['test']['accuracy']*100:.2f}%, HANYA untuk info, "
          f"TIDAK dipakai memutuskan).")

    print(f"\nTotal waktu (11 titik dilatih): {total_time/60:.1f} menit ({total_time/3600:.2f} jam)")
    print(f"\n[INFO] Hasil disimpan: {RESULTS_PATH}")
    print("[INFO] Tidak ada checkpoint/hasil lama yang ditimpa.")


if __name__ == "__main__":
    main()