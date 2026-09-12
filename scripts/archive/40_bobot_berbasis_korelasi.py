"""
40_bobot_berbasis_korelasi.py
Menghitung bobot w1(L1)/w2(Entropi)/w3(GM) dengan TIGA rumus berbeda,
salah satunya memakai "keunikan" kriteria (u_k) -- seberapa TIDAK
berkorelasi sebuah kriteria terhadap kedua kriteria lain, per lapisan,
dirata-ratakan atas seluruh lapisan.

Definisi keunikan kriteria k (dari 3 kriteria: L1, Entropi, GM):
    u_k = 1 - rata-rata |Spearman(k, j)| untuk seluruh j != k
    (rata-rata dua nilai |rho| di sini, karena hanya ada 2 kriteria "j"
    lain selain k), dihitung PER LAPISAN lalu dirata-ratakan atas 16
    lapisan yang bisa dipangkas. u_k tinggi = skor kriteria k berbeda
    (tidak berkorelasi) dari kedua kriteria lain -- membawa informasi
    yang lebih independen.

Tiga rumus bobot (w_k dinormalisasi supaya w1+w2+w3=1 di ketiganya):
    Rumus 1 (akurasi saja)  : w_k sebanding Akurasi_k
    Rumus 2 (keunikan saja) : w_k sebanding u_k
    Rumus 3 (gabungan)      : w_k sebanding Akurasi_k * u_k

Akurasi validasi kriteria tunggal pada rasio 40% DIBACA dari sumber yang
sama dipakai 38_multikriteria_gm.py (outputs/ablation_val_results.json
untuk L1/Entropi, outputs/kriteria_redundansi_tunggal_results.json untuk
GM), TIDAK dihitung ulang -- lalu dicocokkan (sanity check) terhadap
angka yang diberikan di permintaan (L1=94.25%, Entropi=90.80%, GM=93.10%).

Skor L1, entropi (kalibrasi train, 20 citra/kelas, seed=CFG.SEED), dan GM
(dari bobot conv 1x1 ekspansi, fungsi disalin PERSIS dari
36_skor_redundansi.py/37/38, TIDAK diubah) dihitung dari
checkpoints/baseline_9class.pth -- SAMA seperti skrip 36/37/38.

Untuk setiap rumus, dilaporkan: nilai u_k (sama di ketiga rumus, murni
properti data -- ditampilkan ulang di tiap rumus untuk kemudahan baca),
bobot w1/w2/w3 hasil rumus itu, dan persentase channel yang dipilih
BERBEDA dibanding mask konfigurasi B (0.45, 0.10, 0.45) dari
38_multikriteria_gm.py, pada rasio 40%.

Skrip ini TIDAK melatih dan TIDAK mengevaluasi model apa pun -- murni
skor, korelasi Spearman, dan perbandingan mask. TIDAK mengubah skrip
lain.

Jalankan:
    venv/Scripts/python.exe scripts/40_bobot_berbasis_korelasi.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from scipy.stats import spearmanr

from src.config import CFG
from src.dataset import get_dataloaders
from src.model import load_checkpoint
from src.pruning import (
    compute_l1_scores, compute_entropy_scores,
    compute_importance_scores, get_pruning_mask,
    _get_prunable_layers,
)

CHECKPOINT_NAME = "baseline_9class.pth"
VAL_RESULTS_PATH = CFG.OUTPUT_DIR / "ablation_val_results.json"
REDUNDANSI_RESULTS_PATH = CFG.OUTPUT_DIR / "kriteria_redundansi_tunggal_results.json"
RESULTS_PATH = CFG.OUTPUT_DIR / "bobot_berbasis_korelasi_results.json"
RATIO = 0.4
RATIO_PERSEN = 40

AKURASI_ACUAN_DARI_PERMINTAAN = {"l1": 0.9425, "entropi": 0.9080, "gm": 0.9310}
KONFIGURASI_B = {"l1": 0.45, "entropi": 0.10, "gm": 0.45}
LABEL_KRITERIA = ["l1", "entropi", "gm"]
LABEL_TAMPIL = {"l1": "L1", "entropi": "Entropi", "gm": "GM"}


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
# Bagian baru skrip ini
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


def hitung_u_k(skor_dict, common_layers):
    """u_k = 1 - rata-rata |Spearman(k,j)| atas j != k, dihitung per
    lapisan lalu dirata-ratakan atas seluruh lapisan di common_layers."""
    rho_abs = {}  # (k, j) -> list |rho| per lapisan (urutan = common_layers)
    for i in range(len(LABEL_KRITERIA)):
        for j in range(i + 1, len(LABEL_KRITERIA)):
            k1, k2 = LABEL_KRITERIA[i], LABEL_KRITERIA[j]
            nilai = []
            for idx in common_layers:
                r, _ = spearmanr(skor_dict[k1][idx].numpy(), skor_dict[k2][idx].numpy())
                nilai.append(abs(float(r)))
            rho_abs[(k1, k2)] = nilai
            rho_abs[(k2, k1)] = nilai

    u = {}
    detail_per_layer = {}
    for k in LABEL_KRITERIA:
        lainnya = [j for j in LABEL_KRITERIA if j != k]
        rata2_per_layer = []
        for pos in range(len(common_layers)):
            nilai_j = [rho_abs[(k, j)][pos] for j in lainnya]
            rata2_per_layer.append(sum(nilai_j) / len(nilai_j))
        u_per_layer = [1.0 - v for v in rata2_per_layer]
        u[k] = float(sum(u_per_layer) / len(u_per_layer))
        detail_per_layer[k] = {
            "rata_rata_absrho_per_layer": rata2_per_layer,
            "u_per_layer": u_per_layer,
        }
    return u, detail_per_layer, rho_abs


def hitung_persentase_beda_mask(mask_a, mask_b):
    total_channel = 0
    total_beda = 0
    for idx in sorted(mask_a):
        a, b = mask_a[idx], mask_b[idx]
        total_beda += int((a != b).sum())
        total_channel += len(a)
    persen = (total_beda / total_channel * 100) if total_channel else 0.0
    return total_beda, total_channel, persen


def bangun_mask(l1_scores, entropy_scores, gm_scores, w):
    """w: dict {"l1":..,"entropi":..,"gm":..}. compute_importance_scores
    dipanggil dengan argumen posisi (l1, entropi, gm) -- BUKAN (l1, bn,
    entropi) -- label '[WSM] w2(BN)'/'w3(H)' yang tercetak fungsi itu
    SEBENARNYA berarti bobot Entropi dan GM (sama seperti 38)."""
    importance = compute_importance_scores(
        l1_scores, entropy_scores, gm_scores,
        w1=w["l1"], w2=w["entropi"], w3=w["gm"]
    )
    return get_pruning_mask(importance, RATIO)


def rumus1_akurasi(akurasi):
    total = sum(akurasi.values())
    return {k: v / total for k, v in akurasi.items()}


def rumus2_keunikan(u):
    total = sum(u.values())
    return {k: v / total for k, v in u.items()}


def rumus3_gabungan(akurasi, u):
    produk = {k: akurasi[k] * u[k] for k in LABEL_KRITERIA}
    total = sum(produk.values())
    return {k: v / total for k, v in produk.items()}


def main():
    device = torch.device("cpu")
    ckpt_path = CFG.CHECKPOINT_DIR / CHECKPOINT_NAME
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint tidak ditemukan: {ckpt_path}")
        return

    print("=" * 70)
    print("BOBOT L1/ENTROPI/GM BERBASIS AKURASI, KEUNIKAN (KORELASI), DAN GABUNGAN")
    print("=" * 70)

    model = load_checkpoint(ckpt_path, device)[0]
    model.eval()

    SPLIT_DIR = CFG.DATASET_DIR.parent / "dataset_split"
    dataloaders, _ = get_dataloaders(SPLIT_DIR)

    print("\n[INFO] Menghitung skor L1, entropi, dan GM (dari baseline)...")
    l1_scores = compute_l1_scores(model)
    entropy_scores = compute_entropy_scores(
        model, dataloaders["train"], device, samples_per_class=20, seed=CFG.SEED
    )
    gm_scores = hitung_skor_gm_semua_layer(model)
    skor_dict = {"l1": l1_scores, "entropi": entropy_scores, "gm": gm_scores}

    common_layers = sorted(set(l1_scores) & set(entropy_scores) & set(gm_scores))
    if len(common_layers) != 16:
        print(f"[PERINGATAN] Jumlah lapisan dengan ketiga skor lengkap: "
              f"{len(common_layers)} (diharapkan 16)")

    # ----- Akurasi validasi (dibaca dari sumber, dicocokkan dengan permintaan) -----
    acc_l1, acc_entropi = muat_akurasi_val_l1_entropi(RATIO_PERSEN)
    acc_gm = muat_akurasi_val_gm(RATIO_PERSEN)
    if acc_l1 is None or acc_entropi is None or acc_gm is None:
        print(f"\n[PERINGATAN] Sumber akurasi validasi tidak lengkap untuk rasio {RATIO_PERSEN}%, "
              f"memakai angka dari permintaan sebagai fallback: {AKURASI_ACUAN_DARI_PERMINTAAN}")
        akurasi = dict(AKURASI_ACUAN_DARI_PERMINTAAN)
    else:
        akurasi = {"l1": acc_l1, "entropi": acc_entropi, "gm": acc_gm}

    print(f"\n{'=' * 70}")
    print("PEMERIKSAAN AKURASI VALIDASI (rasio 40%) vs ANGKA DI PERMINTAAN")
    print("=" * 70)
    for k in LABEL_KRITERIA:
        acuan = AKURASI_ACUAN_DARI_PERMINTAAN[k] * 100
        aktual = akurasi[k] * 100
        cocok = "OK" if abs(aktual - acuan) < 0.05 else "BEDA -- PERIKSA!"
        print(f"  {LABEL_TAMPIL[k]:<8}: sumber={aktual:.2f}%  permintaan={acuan:.2f}%  [{cocok}]")
    print("=" * 70)

    # ----- Hitung u_k -----
    u, detail_per_layer, rho_abs = hitung_u_k(skor_dict, common_layers)

    print(f"\n{'=' * 70}")
    print("KORELASI SPEARMAN ANTAR KRITERIA (rata-rata |rho| atas 16 lapisan)")
    print("=" * 70)
    for (k1, k2) in [("l1", "entropi"), ("l1", "gm"), ("entropi", "gm")]:
        rata2 = float(np.mean(rho_abs[(k1, k2)]))
        print(f"  |Spearman({LABEL_TAMPIL[k1]}, {LABEL_TAMPIL[k2]})| rata-rata = {rata2:.4f}")
    print("-" * 70)
    print("NILAI KEUNIKAN u_k (1 - rata-rata |rho| terhadap 2 kriteria lain, per lapisan, "
          "dirata-rata 16 lapisan):")
    for k in LABEL_KRITERIA:
        print(f"  u_{LABEL_TAMPIL[k]:<8} = {u[k]:.4f}")
    print("=" * 70)

    # ----- Mask konfigurasi B (pembanding) -----
    mask_b = bangun_mask(l1_scores, entropy_scores, gm_scores, KONFIGURASI_B)

    # ----- Tiga rumus -----
    w_rumus1 = rumus1_akurasi(akurasi)
    w_rumus2 = rumus2_keunikan(u)
    w_rumus3 = rumus3_gabungan(akurasi, u)

    hasil_rumus = {}
    for nama_rumus, w in [("rumus1_akurasi", w_rumus1), ("rumus2_keunikan", w_rumus2),
                            ("rumus3_gabungan", w_rumus3)]:
        print(f"\n{'=' * 90}")
        judul = {"rumus1_akurasi": "RUMUS 1 -- w_k sebanding AKURASI_k",
                  "rumus2_keunikan": "RUMUS 2 -- w_k sebanding u_k (KEUNIKAN)",
                  "rumus3_gabungan": "RUMUS 3 -- w_k sebanding AKURASI_k * u_k"}[nama_rumus]
        print(judul)
        print("=" * 90)
        print(f"{'Kriteria':>10} {'Akurasi':>10} {'u_k':>8} {'Bobot w_k':>10}")
        print("-" * 90)
        for k in LABEL_KRITERIA:
            print(f"{LABEL_TAMPIL[k]:>10} {akurasi[k]*100:>9.2f}% {u[k]:>8.4f} {w[k]:>10.4f}")
        print(f"  Jumlah bobot: {sum(w.values()):.4f}")

        mask_cfg = bangun_mask(l1_scores, entropy_scores, gm_scores, {
            "l1": w["l1"], "entropi": w["entropi"], "gm": w["gm"]
        })
        _, total_ch, persen_beda = hitung_persentase_beda_mask(mask_cfg, mask_b)
        print(f"  Perbedaan vs mask Konfigurasi B (0.45, 0.10, 0.45): "
              f"{persen_beda:.2f}% channel (dari {total_ch} channel)")
        print("=" * 90)

        hasil_rumus[nama_rumus] = {
            "judul": judul,
            "bobot": {"w1_l1": w["l1"], "w2_entropi": w["entropi"], "w3_gm": w["gm"]},
            "persen_beda_vs_konfigurasi_b": persen_beda,
            "total_channel": total_ch,
        }

    # ----- Simpan -----
    save_data = {
        "checkpoint": CHECKPOINT_NAME,
        "rasio": RATIO,
        "akurasi_validasi": akurasi,
        "akurasi_validasi_acuan_permintaan": AKURASI_ACUAN_DARI_PERMINTAAN,
        "definisi_u_k": "u_k = 1 - rata-rata |Spearman(k,j)| atas j != k, per lapisan, "
                        "dirata-ratakan atas 16 lapisan",
        "u_k": u,
        "detail_u_k_per_layer": detail_per_layer,
        "korelasi_rata_rata_antar_kriteria": {
            f"{k1}_vs_{k2}": float(np.mean(rho_abs[(k1, k2)]))
            for (k1, k2) in [("l1", "entropi"), ("l1", "gm"), ("entropi", "gm")]
        },
        "konfigurasi_b_pembanding": KONFIGURASI_B,
        "rumus": hasil_rumus,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Hasil disimpan: {RESULTS_PATH}")
    print("[INFO] Tidak ada model yang dilatih atau dievaluasi.")


if __name__ == "__main__":
    main()