"""
36_skor_redundansi.py
Menghitung skor REDUNDANSI antar channel intermediate (dari bobot conv
1x1 EKSPANSI, sebelum depthwise) dan mengukur seberapa berbeda informasi
yang dibawanya dibandingkan tiga kriteria yang sudah dipakai di seluruh
tesis (L1, BN gamma, entropi).

Dua varian skor redundansi, dihitung dari vektor bobot per channel
(baris conv ekspansi, diratakan sepanjang jumlah channel masukan blok):
  - GM  : jarak Euclidean channel tsb ke MEDIAN GEOMETRIK seluruh channel
          di lapisan yang sama (algoritma Weiszfeld, BUKAN rata-rata --
          median geometrik meminimumkan total JARAK, rata-rata
          meminimumkan total JARAK KUADRAT; skrip ini memakai median
          geometrik sungguhan, rata-rata hanya dipakai sebagai tebakan
          awal iterasi). Interpretasi umum di literatur pruning berbasis
          median geometrik (mis. FPGM, He et al. 2019): channel yang
          DEKAT ke median geometrik paling mudah digantikan channel lain
          (paling redundan) karena mendekati arah yang dibagi banyak
          channel. Skrip ini melaporkan JARAK MENTAH apa adanya (tidak
          dibalik tandanya) -- jarak kecil = dekat median = (menurut
          interpretasi di atas) redundan.
  - COS : rata-rata kemiripan kosinus channel tsb terhadap SELURUH
          channel lain (bukan termasuk dirinya) di lapisan yang sama.
          Nilai tinggi = mirip banyak channel lain = redundan (arah
          skornya sudah selaras: tinggi = redundan, tidak perlu dibalik).

Channel dengan bobot ekspansi identik/mirip dianggap redundan karena
keduanya memproyeksikan input ke arah fitur yang hampir sama sebelum
depthwise -- salah satunya bisa dibuang tanpa banyak kehilangan
informasi UNIK (walau keduanya bisa saja tetap "penting" menurut L1/BN/
entropi, karena kriteria itu tidak melihat KEMIRIPAN antar channel sama
sekali, hanya BESAR/KEKAYAAN informasi channel itu SENDIRI). Itulah
pertanyaan yang mau dijawab skrip ini: seberapa berbeda info GM/COS
dari tiga kriteria yang ada -- korelasi Spearman rendah berarti
redundansi adalah SUMBER INFORMASI BARU yang belum ditangkap L1/BN/
entropi.

Prosedur:
  1. Muat checkpoints/baseline_9class.pth.
  2. Untuk tiap lapisan yang bisa dipangkas (16 lapisan, sama seperti
     _get_prunable_layers() di src/pruning.py), ambil bobot conv 1x1
     EKSPANSI blok tsb (kriteria deteksi IDENTIK dengan yang dipakai
     apply_pruning()/_get_prunable_layers(): kernel 1x1, groups=1,
     out_channels > in_channels) -- SATU baris bobot per channel
     intermediate, diratakan jadi vektor sepanjang channel masukan blok.
  3. Hitung GM dan COS dari vektor-vektor itu (lihat definisi di atas).
  4. Hitung ulang skor L1, BN gamma, entropi (kalibrasi TRAIN, 20 citra/
     kelas, seed=CFG.SEED) -- fungsi yang SAMA PERSIS dipakai seluruh
     eksperimen lain (compute_l1_scores, compute_bn_scores,
     compute_entropy_scores dari src/pruning.py, TIDAK diubah).

Untuk SETIAP lapisan, laporkan: Spearman(GM,L1), Spearman(GM,BN),
Spearman(GM,Entropi), Spearman(COS,L1), Spearman(COS,BN),
Spearman(COS,Entropi), dan Spearman(GM,COS). Lalu ringkasan lintas
16 lapisan: rata-rata, rentang (min-maks), dan jumlah lapisan dengan
|korelasi| < 0.3 (dianggap "lemah", ambang eksplorasi -- bukan uji
signifikansi formal) untuk tiap pasangan korelasi.

Skrip ini TIDAK melatih dan TIDAK mengevaluasi model apa pun -- murni
memuat checkpoint baseline, menghitung skor, dan melaporkan. TIDAK
mengubah skrip/modul lain.

Jalankan:
    venv/Scripts/python.exe scripts/36_skor_redundansi.py
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
    compute_l1_scores, compute_bn_scores, compute_entropy_scores,
    _get_prunable_layers,
)

CHECKPOINT_NAME = "baseline_9class.pth"
RESULTS_PATH = CFG.OUTPUT_DIR / "skor_redundansi_results.json"
AMBANG_KORELASI_LEMAH = 0.3

KOLOM_KORELASI = [
    "spearman_gm_vs_l1", "spearman_gm_vs_bn", "spearman_gm_vs_entropi",
    "spearman_cos_vs_l1", "spearman_cos_vs_bn", "spearman_cos_vs_entropi",
    "spearman_gm_vs_cos",
]
LABEL_KOLOM = {
    "spearman_gm_vs_l1": "GM~L1", "spearman_gm_vs_bn": "GM~BN", "spearman_gm_vs_entropi": "GM~Ent",
    "spearman_cos_vs_l1": "COS~L1", "spearman_cos_vs_bn": "COS~BN", "spearman_cos_vs_entropi": "COS~Ent",
    "spearman_gm_vs_cos": "GM~COS",
}


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
    """Median geometrik (algoritma Weiszfeld) dari X (n x d): titik yang
    meminimumkan total JARAK EUCLIDEAN ke seluruh titik (beda dari
    rata-rata, yang meminimumkan total jarak KUADRAT). Dimulai dari
    rata-rata sebagai tebakan awal. epsilon ditambahkan ke jarak untuk
    menghindari pembagian nol kalau iterasi bertepatan persis dengan
    salah satu titik data -- penyederhanaan praktis yang lazim dipakai,
    bukan mempengaruhi hasil akhir secara berarti."""
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
    SELURUH channel di lapisan yang sama. Lihat catatan interpretasi
    (jarak kecil = redundan) di docstring modul."""
    median = hitung_median_geometrik(X)
    return torch.norm(X - median, dim=1)


def hitung_skor_cos(X):
    """COS[c] = rata-rata kemiripan kosinus channel c terhadap SELURUH
    channel lain (bukan termasuk dirinya) di lapisan yang sama. Tinggi =
    redundan (mirip banyak channel lain)."""
    norm = X.norm(dim=1, keepdim=True).clamp_min(1e-12)
    X_norm = X / norm
    sim = X_norm @ X_norm.T  # (n, n), diagonal = kemiripan-ke-diri-sendiri = 1.0
    n = X.shape[0]
    if n <= 1:
        return torch.zeros(n)
    total_tanpa_diri_sendiri = sim.sum(dim=1) - 1.0
    return total_tanpa_diri_sendiri / (n - 1)


def spearman_rp(a, b):
    r, p = spearmanr(a, b)
    return float(r), float(p)


def main():
    device = torch.device("cpu")
    ckpt_path = CFG.CHECKPOINT_DIR / CHECKPOINT_NAME
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint tidak ditemukan: {ckpt_path}")
        return

    print("=" * 70)
    print("SKOR REDUNDANSI ANTAR CHANNEL (GM & COS) vs L1 / BN GAMMA / ENTROPI")
    print("=" * 70)

    model, _ = load_checkpoint(ckpt_path, device)
    model.eval()

    SPLIT_DIR = CFG.DATASET_DIR.parent / "dataset_split"
    dataloaders, _ = get_dataloaders(SPLIT_DIR)

    prunable = _get_prunable_layers(model)
    block_idx_map = {layer_idx: block_idx for layer_idx, (block_idx, _) in enumerate(prunable)}

    print("\n[INFO] Menghitung skor L1, BN gamma, entropi (fungsi sama persis dengan eksperimen lain)...")
    l1_scores = compute_l1_scores(model)
    bn_scores = compute_bn_scores(model)
    entropy_scores = compute_entropy_scores(
        model, dataloaders["train"], device, samples_per_class=20, seed=CFG.SEED
    )

    print("\n[INFO] Menghitung skor redundansi GM (median geometrik, Weiszfeld) dan COS "
          "dari bobot conv ekspansi...")
    gm_scores = {}
    cos_scores = {}
    for layer_idx, (block_idx, dw_module) in enumerate(prunable):
        block = model.features[block_idx]
        exp_conv = cari_conv_ekspansi(block)
        if exp_conv is None:
            print(f"[PERINGATAN] layer {layer_idx} (block {block_idx}): conv ekspansi "
                  f"tidak ditemukan, dilewati.")
            continue
        W = exp_conv.weight.data  # (mid_channels, in_channels, 1, 1)
        X = W.reshape(W.shape[0], -1).float()  # (mid_channels, in_channels), diratakan

        gm_scores[layer_idx] = hitung_skor_gm(X)
        cos_scores[layer_idx] = hitung_skor_cos(X)

    common_layers = sorted(
        set(l1_scores) & set(bn_scores) & set(entropy_scores) & set(gm_scores) & set(cos_scores)
    )
    if len(common_layers) != 16:
        print(f"[PERINGATAN] Jumlah lapisan dengan seluruh skor lengkap: "
              f"{len(common_layers)} (diharapkan 16)")

    hasil_per_lapisan = []
    for idx in common_layers:
        l1 = l1_scores[idx].numpy()
        bn = bn_scores[idx].numpy()
        entropi = entropy_scores[idx].numpy()
        gm = gm_scores[idx].numpy()
        cos = cos_scores[idx].numpy()

        rho_gm_l1, p_gm_l1 = spearman_rp(gm, l1)
        rho_gm_bn, p_gm_bn = spearman_rp(gm, bn)
        rho_gm_entropi, p_gm_entropi = spearman_rp(gm, entropi)
        rho_cos_l1, p_cos_l1 = spearman_rp(cos, l1)
        rho_cos_bn, p_cos_bn = spearman_rp(cos, bn)
        rho_cos_entropi, p_cos_entropi = spearman_rp(cos, entropi)
        rho_gm_cos, p_gm_cos = spearman_rp(gm, cos)

        hasil_per_lapisan.append({
            "layer_idx": idx,
            "block_idx_mobilenetv2": block_idx_map.get(idx),
            "jumlah_channel": int(len(l1)),
            "spearman_gm_vs_l1": rho_gm_l1, "spearman_gm_vs_l1_pvalue": p_gm_l1,
            "spearman_gm_vs_bn": rho_gm_bn, "spearman_gm_vs_bn_pvalue": p_gm_bn,
            "spearman_gm_vs_entropi": rho_gm_entropi, "spearman_gm_vs_entropi_pvalue": p_gm_entropi,
            "spearman_cos_vs_l1": rho_cos_l1, "spearman_cos_vs_l1_pvalue": p_cos_l1,
            "spearman_cos_vs_bn": rho_cos_bn, "spearman_cos_vs_bn_pvalue": p_cos_bn,
            "spearman_cos_vs_entropi": rho_cos_entropi, "spearman_cos_vs_entropi_pvalue": p_cos_entropi,
            "spearman_gm_vs_cos": rho_gm_cos, "spearman_gm_vs_cos_pvalue": p_gm_cos,
        })

    # ----- Tabel per lapisan -----
    print(f"\n{'=' * 130}")
    print("TABEL KORELASI SPEARMAN PER LAPISAN")
    print("=" * 130)
    header = f"{'Layer':>5} {'Block':>5} {'#Ch':>5} " + "".join(
        f"{LABEL_KOLOM[k]:>10}" for k in KOLOM_KORELASI
    )
    print(header)
    print("-" * 130)
    for r in hasil_per_lapisan:
        baris = f"{r['layer_idx']:>5} {r['block_idx_mobilenetv2']:>5} {r['jumlah_channel']:>5} "
        baris += "".join(f"{r[k]:>10.4f}" for k in KOLOM_KORELASI)
        print(baris)
    print("=" * 130)

    # ----- Ringkasan lintas lapisan -----
    ringkasan = {}
    for kolom in KOLOM_KORELASI:
        vals = [r[kolom] for r in hasil_per_lapisan]
        n_lemah = sum(1 for v in vals if abs(v) < AMBANG_KORELASI_LEMAH)
        ringkasan[kolom] = {
            "rata_rata": float(np.mean(vals)),
            "min": float(min(vals)),
            "maks": float(max(vals)),
            "rentang": float(max(vals) - min(vals)),
            "jumlah_lapisan_korelasi_lemah": n_lemah,
        }

    print(f"\n{'=' * 100}")
    print(f"RINGKASAN LINTAS {len(hasil_per_lapisan)} LAPISAN "
          f"(\"lemah\" = |korelasi| < {AMBANG_KORELASI_LEMAH})")
    print("=" * 100)
    print(f"{'Pasangan':>10} {'Rata-rata':>10} {'Min':>8} {'Maks':>8} {'Rentang':>9} {'#Lemah':>8}")
    print("-" * 100)
    for kolom in KOLOM_KORELASI:
        r = ringkasan[kolom]
        print(f"{LABEL_KOLOM[kolom]:>10} {r['rata_rata']:>10.4f} {r['min']:>8.4f} "
              f"{r['maks']:>8.4f} {r['rentang']:>9.4f} {r['jumlah_lapisan_korelasi_lemah']:>7}/16")
    print("=" * 100)

    # ----- Simpan -----
    save_data = {
        "checkpoint": CHECKPOINT_NAME,
        "sumber_bobot_redundansi": "conv 1x1 ekspansi (sebelum depthwise), diratakan per channel",
        "metode_gm": "jarak Euclidean ke median geometrik (algoritma Weiszfeld); jarak kecil = "
                     "dekat median = (interpretasi FPGM) redundan; skrip melaporkan jarak mentah",
        "metode_cos": "rata-rata kemiripan kosinus terhadap channel lain di lapisan yang sama; "
                      "tinggi = redundan",
        "entropy_calibration": {"source_split": "train", "samples_per_class": 20, "seed": CFG.SEED},
        "ambang_korelasi_lemah": AMBANG_KORELASI_LEMAH,
        "per_layer": hasil_per_lapisan,
        "ringkasan": ringkasan,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Hasil disimpan: {RESULTS_PATH}")


if __name__ == "__main__":
    main()