"""
30_alokasi_anggaran_entropi.py
Varian pruning baru: ANGGARAN pemangkasan per lapisan ditentukan oleh
entropi rata-rata lapisan (lapisan berentropi rendah = kurang informatif
= dipangkas lebih banyak), sedangkan PEMILIHAN channel di dalam satu
lapisan tetap memakai norma L1 (channel dengan norma L1 terendah yang
dipangkas). Ini berbeda dari I(c) = w1*S_L1 + w2*S_BN + w3*S_H yang
dipakai di seluruh eksperimen lain (04, 07, 11, 12, dst.) -- di sana
entropi ikut menentukan skor PER CHANNEL yang digabung dengan L1 dan BN;
di sini entropi hanya menentukan RASIO PER LAPISAN, lalu L1 murni yang
memutus channel mana yang dipangkas di dalam lapisan itu.

Rumus alokasi MODE "entropi" (per lapisan i, dari 16 lapisan):
  1. e_rel[i]        = entropi_rata2[i] / mean(entropi_rata2 seluruh lapisan)
  2. bobot_balik[i]   = (1 / e_rel[i]) ** alpha
     alpha=0 -> bobot_balik seragam (1.0 semua lapisan) -> alokasi rasio
     seragam adalah KASUS KHUSUS alpha=0, bukan metode terpisah. alpha
     lebih besar -> diferensiasi lebih tajam antara lapisan berentropi
     rendah vs tinggi. Bawaan alpha=1.0, diatur lewat --alpha.
  3. k                = (rasio_global * total_channel) /
                        sum_i(bobot_balik[i] * jumlah_channel[i])
  4. rasio_mentah[i]  = k * bobot_balik[i]
  5. rasio_terklip[i] = clip(rasio_mentah[i], 0.05, 0.85)

  (Versi sebelumnya memakai normalisasi min-maks lalu 1-x, yang membuat
  lapisan berentropi TERTINGGI selalu dapat bobot_balik persis 0 --
  artefak rumus, bukan pilihan metodologis, karena itu diganti.)

Rumus alokasi MODE "acak" (skenario kontrol): 16 rasio per lapisan
dibangkitkan acak dengan seed tetap (CFG.SEED), dipaksa memiliki
simpangan baku yang SETARA dengan alokasi mode "entropi" pada rasio
global dan alpha yang sama (lewat rescaling z-score). Tujuannya
memisahkan apakah yang membantu akurasi itu INFORMASI entropi per
lapisan, atau sekadar KETIDAKSERAGAMAN alokasi (yang bisa "membantu"
murni lewat efek regularisasi tidak-seragam, terlepas dari isi
informasinya).

Total pemangkasan mode "acak" DICOCOKKAN KE JUMLAH PARAMETER akhir
ablation_l1 pada rasio yang sama (bukan sekadar jumlah channel) lewat
pencarian bagi dua (bisection) pada offset seragam mu yang ditambahkan
ke rasio ter-z-score, dievaluasi lewat pemangkasan+penghitungan
parameter SUNGGUHAN (buat_mask_dari_alokasi + apply_pruning +
count_parameters) di setiap iterasi -- bukan rumus tertutup. Ini
diperlukan karena jumlah CHANNEL yang sama tidak menjamin jumlah
PARAMETER yang sama: "harga" parameter per channel yang dipangkas
berbeda antar lapisan (lapisan dengan in/out channel blok yang lebih
lebar kehilangan lebih banyak parameter per channel yang dipangkas
dibanding lapisan yang lebih sempit). Versi sebelumnya mencocokkan
jumlah CHANNEL saja dan menghasilkan model rasio-40%-acak dengan
1.570.364 parameter vs ablation_l1 1.513.575 (selisih 3.75%, melebihi
ambang 1% syarat perbandingan adil) -- persis karena mode acak waktu
itu kebetulan memangkas lebih SEDIKIT di lapisan 960-channel (yang
mahal per channel) dan lebih BANYAK di lapisan kecil (yang murah per
channel), walau total channel yang dipangkas hampir sama dengan mode
entropi. Kalau target_num_params tidak tersedia (data ablation_l1 untuk
rasio itu tidak ditemukan), turun ke pencocokan jumlah channel seperti
versi sebelumnya, dengan peringatan eksplisit di layar dan di hasil.

Simpangan baku target (dari mode entropi) dan aktual (setelah kliping
mode acak) sama-sama dilaporkan supaya kesetaraannya bisa diverifikasi.

Clipping ke [0.05, 0.85] di kedua mode bisa membuat total channel yang
benar-benar dipangkas dan/atau simpangan bakunya sedikit menyimpang dari
target -- skrip ini TIDAK menyeimbangkan ulang secara iteratif untuk
channel (supaya rumusnya tetap sederhana dan mudah diperiksa manual),
tapi target vs aktual SELALU dilaporkan berdampingan. Untuk parameter
(mode acak), pencocokan MEMANG iteratif (bisection), karena itu yang
dipakai sebagai syarat perbandingan adil.

Seed acak (--seed, bawaan CFG.SEED=42) HANYA memengaruhi fine-tuning
(urutan batch, augmentasi acak, dropout) lewat set_seed() yang dipanggil
sekali per rasio tepat sebelum train_model(), MENGIKUTI KONVENSI
15_multiseed_validation.py -- bukan memengaruhi skor L1/entropi atau
alokasi/pemilihan channel, yang selalu dihitung deterministik dari
baseline dan identik di semua seed. Checkpoint dan kunci hasil memakai
akhiran _seed{seed} HANYA kalau seed != CFG.SEED, supaya checkpoint dan
entri hasil yang sudah ada dari seed bawaan tidak tertimpa/berubah nama.

CATATAN: dieksekusi pertama kali (mode entropi & acak, alpha=2.0,
rasio 40%) SEBELUM --seed ditambahkan ke skrip ini -- run itu TIDAK
memanggil set_seed() sama sekali (beda dari skrip lain di proyek ini
yang semuanya memanggil set_seed(CFG.SEED) minimal sekali di awal).
Checkpoint hasil run itu (tanpa akhiran _seed) karena itu bukan run
yang seed-nya benar-benar terkontrol/reproducible, sekalipun secara
konvensi diperlakukan sebagai titik data "seed 42" saat dibandingkan
dengan run --seed 123 dan --seed 2024 yang baru (yang BENAR terkontrol).

Mode operasi:
  - TANPA --jalankan-eksperimen (bawaan): hanya menghitung skor, lalu
    menampilkan + menyimpan tabel pratinjau alokasi mode "entropi" untuk
    rasio 40% pada alpha 0.5, 1.0, dan 2.0, plus SATU tabel mode "acak"
    (dipadankan ke alpha 1.0) sebagai verifikasi kesetaraan simpangan
    baku. Lalu BERHENTI -- tidak memangkas atau melatih apa pun.
  - DENGAN --jalankan-eksperimen: menjalankan pemangkasan + fine-tuning
    30 epoch (CFG.FINETUNE_EPOCHS, CFG.FINETUNE_LR, train_model() yang
    sama persis dipakai skrip 04/12 -- Adam + StepLR) untuk tiap rasio
    di --ratios, memakai --mode (entropi/acak), --alpha, dan --seed yang
    dipilih.

Syarat perbandingan adil: untuk tiap rasio, jumlah parameter akhir
dibandingkan terhadap ablation_l1 pada rasio yang sama (dari
outputs/tabel_hasil_lengkap.json). Kalau selisihnya > 1%, ditandai TIDAK
VALID dan dicatat sebagai catatan eksplisit di hasil -- bukan diam-diam
diabaikan.

Jalankan:
    venv/Scripts/python.exe scripts/30_alokasi_anggaran_entropi.py
        (mode tinjau -- tabel alpha 0.5/1.0/2.0 + acak, tidak melatih apa pun)
    venv/Scripts/python.exe scripts/30_alokasi_anggaran_entropi.py --jalankan-eksperimen --mode entropi --alpha 2.0 --ratios 0.4 --seed 123
    venv/Scripts/python.exe scripts/30_alokasi_anggaran_entropi.py --jalankan-eksperimen --mode acak --ratios 0.4
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

from src.config import CFG
from src.dataset import get_dataloaders
from src.model import load_checkpoint, train_model, evaluate_model, count_parameters
from src.pruning import compute_l1_scores, compute_entropy_scores, apply_pruning
from src.visualize import plot_confusion_matrix, print_results_table

CHECKPOINT_NAME = "baseline_9class.pth"
RASIO_MIN_LAYER = 0.05
RASIO_MAKS_LAYER = 0.85
ALPHA_PRATINJAU = [0.5, 1.0, 2.0]
RESULTS_PATH = CFG.OUTPUT_DIR / "alokasi_anggaran_entropi_results.json"
ALOKASI_PREVIEW_PATH = CFG.OUTPUT_DIR / "alokasi_anggaran_entropi_preview_40pct.json"


def set_seed(seed):
    """Mengikuti konvensi 02/03/04/07/11/12/15/16 -- kontrol reproducibility
    untuk fine-tuning (urutan batch, augmentasi, dropout), BUKAN untuk skor
    kepentingan channel (L1/entropi selalu deterministik dari baseline)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def hitung_entropi_rata_rata_per_layer(entropy_scores):
    return {idx: float(scores.mean()) for idx, scores in entropy_scores.items()}


def hitung_bobot_balik(entropi_rata2, alpha):
    """bobot_balik[i] = (1 / e_rel[i]) ** alpha, e_rel[i] = entropi[i] / mean(entropi).
    alpha=0 -> bobot_balik seragam (1.0 semua lapisan)."""
    mean_entropi = sum(entropi_rata2.values()) / len(entropi_rata2)
    e_rel = {i: v / mean_entropi for i, v in entropi_rata2.items()}
    bobot_balik = {i: (1.0 / e_rel[i]) ** alpha for i in e_rel}
    return e_rel, bobot_balik


def alokasikan_anggaran_entropi(entropi_rata2, jumlah_channel, rasio_global, alpha):
    e_rel, bobot_balik = hitung_bobot_balik(entropi_rata2, alpha)

    total_channel = sum(jumlah_channel.values())
    target_prune = rasio_global * total_channel

    penyebut = sum(bobot_balik[i] * jumlah_channel[i] for i in bobot_balik)
    k = target_prune / penyebut if penyebut > 1e-8 else 0.0

    alokasi = {}
    for i in sorted(bobot_balik):
        rasio_mentah = k * bobot_balik[i]
        rasio_terklip = max(RASIO_MIN_LAYER, min(RASIO_MAKS_LAYER, rasio_mentah))
        alokasi[i] = {
            "entropi_rata_rata": entropi_rata2[i],
            "entropi_relatif": e_rel[i],
            "bobot_balik": bobot_balik[i],
            "jumlah_channel": jumlah_channel[i],
            "rasio_mentah": rasio_mentah,
            "rasio_terklip": rasio_terklip,
            "diklip": bool(abs(rasio_mentah - rasio_terklip) > 1e-9),
        }

    ringkasan = _ringkasan_alokasi(alokasi, total_channel, rasio_global, target_prune)
    ringkasan["mode"] = "entropi"
    ringkasan["alpha"] = alpha
    return alokasi, ringkasan


def buat_mask_dari_alokasi(l1_scores, alokasi):
    """Di dalam tiap lapisan, pangkas channel dengan norma L1 TERENDAH
    sebanyak anggaran (rasio_terklip) lapisan itu."""
    masks = {}
    for idx, skor in l1_scores.items():
        n = len(skor)
        num_prune = round(alokasi[idx]["rasio_terklip"] * n)
        num_prune = min(num_prune, n - 1)  # jaga minimal 1 channel tersisa
        _, urutan = torch.sort(skor)
        prune_idx = urutan[:num_prune]
        mask = torch.ones(n, dtype=torch.bool)
        mask[prune_idx] = False
        masks[idx] = mask
    return masks


def alokasikan_anggaran_acak(alokasi_entropi_acuan, jumlah_channel, rasio_global,
                              l1_scores=None, model_baseline=None, target_num_params=None,
                              seed=None, toleransi_persen_params=0.3, maks_iterasi=40):
    """Skenario kontrol: rasio per lapisan ACAK (seed tetap), simpangan baku
    dipaksa setara alokasi_entropi_acuan (lewat rescaling z-score).

    Kalau target_num_params, l1_scores, dan model_baseline diberikan: total
    pemangkasan dicocokkan ke JUMLAH PARAMETER akhir (bukan jumlah channel)
    lewat bisection pada offset seragam mu, dievaluasi dengan pemangkasan
    sungguhan di tiap iterasi -- supaya sepadan dengan ablation_l1 pada
    rasio yang sama (syarat perbandingan adil). Kalau salah satu tidak
    diberikan: turun ke pencocokan jumlah channel (perilaku versi
    sebelumnya), dengan peringatan eksplisit karena jumlah parameter akhir
    TIDAK dijamin sepadan dalam kasus ini."""
    if seed is None:
        seed = CFG.SEED

    layer_ids = sorted(jumlah_channel.keys())
    n_layer = len(layer_ids)

    rasio_entropi = np.array([alokasi_entropi_acuan[i]["rasio_terklip"] for i in layer_ids])
    std_target = float(rasio_entropi.std())

    n_channel = np.array([jumlah_channel[i] for i in layer_ids], dtype=float)
    total_channel = float(n_channel.sum())
    target_prune_channel = rasio_global * total_channel

    rng = np.random.RandomState(seed)
    mentah = rng.standard_normal(n_layer)
    mentah_std = mentah.std()
    mentah_terskala = mentah / mentah_std * std_target if mentah_std > 1e-8 else np.zeros(n_layer)

    def rasio_untuk_mu(mu):
        rasio_mentah_arr = mu + mentah_terskala
        rasio_terklip_arr = np.clip(rasio_mentah_arr, RASIO_MIN_LAYER, RASIO_MAKS_LAYER)
        return rasio_mentah_arr, rasio_terklip_arr

    def num_params_untuk_mu(mu):
        _, rasio_terklip_arr = rasio_untuk_mu(mu)
        alokasi_sementara = {
            layer_ids[pos]: {"jumlah_channel": jumlah_channel[layer_ids[pos]],
                              "rasio_terklip": float(rasio_terklip_arr[pos])}
            for pos in range(n_layer)
        }
        masks = buat_mask_dari_alokasi(l1_scores, alokasi_sementara)
        pruned = apply_pruning(model_baseline, masks)
        return count_parameters(pruned)

    bisa_cocokkan_parameter = (
        target_num_params is not None and l1_scores is not None and model_baseline is not None
    )

    if bisa_cocokkan_parameter:
        mu_rendah, mu_tinggi = -1.0, 2.0
        params_rendah = num_params_untuk_mu(mu_rendah)
        params_tinggi = num_params_untuk_mu(mu_tinggi)
        if not (params_rendah >= target_num_params >= params_tinggi):
            print(f"[PERINGATAN] Target jumlah parameter ({target_num_params:,}) di luar "
                  f"rentang yang tercapai pencarian mu di [{mu_rendah}, {mu_tinggi}] "
                  f"({params_tinggi:,} s.d. {params_rendah:,}). Hasil terbaik yang "
                  f"ditemukan tetap dilaporkan, tapi mungkin tidak masuk ambang toleransi.")

        mu = 0.0
        params_tercapai = None
        for _ in range(maks_iterasi):
            mu = (mu_rendah + mu_tinggi) / 2
            params_tercapai = num_params_untuk_mu(mu)
            selisih = abs(params_tercapai - target_num_params) / target_num_params * 100
            if selisih <= toleransi_persen_params:
                break
            if params_tercapai > target_num_params:
                mu_rendah = mu
            else:
                mu_tinggi = mu
        metode_pencocokan = "parameter (bisection pada mu, dievaluasi via pemangkasan sungguhan)"
    else:
        mu = (target_prune_channel - float((mentah_terskala * n_channel).sum())) / total_channel
        params_tercapai = None
        metode_pencocokan = "channel (fallback -- target_num_params/l1_scores/model_baseline tidak lengkap)"
        print("[PERINGATAN] Mode acak dicocokkan lewat jumlah CHANNEL saja (bukan parameter) "
              "karena target_num_params/l1_scores/model_baseline tidak lengkap -- jumlah "
              "parameter akhir TIDAK dijamin sepadan dengan ablation_l1.")

    rasio_mentah_arr, rasio_terklip_arr = rasio_untuk_mu(mu)

    alokasi = {}
    for pos, i in enumerate(layer_ids):
        alokasi[i] = {
            "jumlah_channel": jumlah_channel[i],
            "rasio_mentah": float(rasio_mentah_arr[pos]),
            "rasio_terklip": float(rasio_terklip_arr[pos]),
            "diklip": bool(abs(rasio_mentah_arr[pos] - rasio_terklip_arr[pos]) > 1e-9),
        }

    ringkasan = _ringkasan_alokasi(alokasi, total_channel, rasio_global, target_prune_channel)
    ringkasan["mode"] = "acak"
    ringkasan["seed"] = seed
    ringkasan["std_target_setara_entropi"] = std_target
    ringkasan["metode_pencocokan_mu"] = metode_pencocokan
    ringkasan["target_num_params_l1"] = target_num_params
    ringkasan["num_params_saat_pencarian_mu"] = params_tercapai
    return alokasi, ringkasan


def _ringkasan_alokasi(alokasi, total_channel, rasio_global, target_prune):
    total_prune_aktual = sum(
        round(a["rasio_terklip"] * a["jumlah_channel"]) for a in alokasi.values()
    )
    rasio_aktual = total_prune_aktual / total_channel
    std_aktual = float(np.array([a["rasio_terklip"] for a in alokasi.values()]).std())
    return {
        "rasio_target": rasio_global,
        "total_channel": total_channel,
        "total_prune_target": target_prune,
        "total_prune_aktual": total_prune_aktual,
        "rasio_aktual": rasio_aktual,
        "selisih_persen_dari_target": abs(rasio_aktual - rasio_global) / rasio_global * 100,
        "std_rasio_per_lapisan_aktual": std_aktual,
    }


def cetak_tabel_alokasi(alokasi, ringkasan, rasio_global, judul):
    kolom_entropi = "entropi_relatif" in next(iter(alokasi.values()))
    print(f"\n{'=' * 100}")
    print(f"TABEL ALOKASI -- {judul} -- RASIO GLOBAL {rasio_global*100:.0f}%")
    print("=" * 100)
    if kolom_entropi:
        header = (f"{'Layer':>5} {'#Ch':>5} {'Entropi':>9} {'E_rel':>8} "
                  f"{'BobotBalik':>11} {'RasioMentah':>12} {'RasioTerklip':>13} {'#Dipangkas':>11} {'Klip?':>6}")
    else:
        header = (f"{'Layer':>5} {'#Ch':>5} {'RasioMentah':>12} {'RasioTerklip':>13} "
                  f"{'#Dipangkas':>11} {'Klip?':>6}")
    print(header)
    print("-" * 100)
    for idx in sorted(alokasi):
        a = alokasi[idx]
        n_prune = round(a["rasio_terklip"] * a["jumlah_channel"])
        if kolom_entropi:
            print(f"{idx:>5} {a['jumlah_channel']:>5} {a['entropi_rata_rata']:>9.4f} "
                  f"{a['entropi_relatif']:>8.4f} {a['bobot_balik']:>11.4f} "
                  f"{a['rasio_mentah']:>12.4f} {a['rasio_terklip']:>13.4f} {n_prune:>11} "
                  f"{'YA' if a['diklip'] else '':>6}")
        else:
            print(f"{idx:>5} {a['jumlah_channel']:>5} "
                  f"{a['rasio_mentah']:>12.4f} {a['rasio_terklip']:>13.4f} {n_prune:>11} "
                  f"{'YA' if a['diklip'] else '':>6}")
    print("-" * 100)
    print(f"Target  : rasio global {ringkasan['rasio_target']*100:.2f}%  "
          f"({ringkasan['total_prune_target']:.1f} dari {ringkasan['total_channel']:.0f} channel)")
    print(f"Aktual  : rasio global {ringkasan['rasio_aktual']*100:.2f}%  "
          f"({ringkasan['total_prune_aktual']} dari {ringkasan['total_channel']:.0f} channel)  "
          f"-- selisih {ringkasan['selisih_persen_dari_target']:.2f}% dari target")
    if "std_target_setara_entropi" in ringkasan:
        print(f"Simpangan baku rasio per lapisan: target(setara entropi)={ringkasan['std_target_setara_entropi']:.4f}  "
              f"aktual(setelah klip)={ringkasan['std_rasio_per_lapisan_aktual']:.4f}")
        print(f"Pencocokan mu: {ringkasan['metode_pencocokan_mu']}")
        if ringkasan.get("target_num_params_l1") is not None:
            tnp = ringkasan["target_num_params_l1"]
            npar = ringkasan.get("num_params_saat_pencarian_mu")
            if npar is not None:
                selisih_par = abs(npar - tnp) / tnp * 100
                print(f"  target num_params (ablation_l1) = {tnp:,}   "
                      f"tercapai saat pencarian mu = {npar:,}   "
                      f"selisih = {selisih_par:.2f}%")
    else:
        print(f"Simpangan baku rasio per lapisan: {ringkasan['std_rasio_per_lapisan_aktual']:.4f}")
    print("=" * 100)


def muat_pembanding_l1(rasio_persen):
    """Ambil num_params ablation_l1 pada rasio yang sama dari
    outputs/tabel_hasil_lengkap.json (sumber sah tunggal), untuk syarat
    perbandingan adil DAN sebagai target pencocokan parameter mode acak.
    None kalau tidak ditemukan."""
    tabel_path = CFG.OUTPUT_DIR / "tabel_hasil_lengkap.json"
    if not tabel_path.exists():
        return None
    with open(tabel_path, encoding="utf-8") as f:
        tabel = json.load(f)
    kunci = f"{rasio_persen}%"
    return tabel.get("results", {}).get("l1", {}).get(kunci)


def main():
    parser = argparse.ArgumentParser(
        description="Alokasi anggaran pemangkasan berbasis entropi per lapisan + seleksi L1"
    )
    parser.add_argument(
        "--jalankan-eksperimen", action="store_true",
        help="Jalankan pemangkasan + fine-tuning penuh untuk --ratios. Tanpa flag ini, "
             "skrip HANYA menampilkan tabel pratinjau (alpha 0.5/1.0/2.0 + acak) untuk "
             "rasio 40%% lalu berhenti (mode tinjau, tidak melatih apa pun)."
    )
    parser.add_argument("--ratios", type=float, nargs="+", default=[0.3, 0.4, 0.5],
                        help="Rasio yang dijalankan kalau --jalankan-eksperimen dipakai")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Kekuatan alokasi mode entropi (0 = seragam). Bawaan 1.0.")
    parser.add_argument("--mode", choices=["entropi", "acak"], default="entropi",
                        help="Mode alokasi untuk --jalankan-eksperimen: 'entropi' (utama) "
                             "atau 'acak' (kontrol, dipadankan simpangan baku & jumlah "
                             "parameter ke mode entropi/ablation_l1 pada rasio yang sama).")
    parser.add_argument("--seed", type=int, default=CFG.SEED,
                        help="Seed untuk fine-tuning (urutan batch/augmentasi/dropout) -- "
                             "TIDAK memengaruhi skor L1/entropi atau pemilihan channel, yang "
                             "selalu deterministik dan identik di semua seed. Bawaan CFG.SEED. "
                             "Checkpoint/kunci hasil memakai akhiran _seed{seed} hanya kalau "
                             "berbeda dari CFG.SEED, supaya entri seed bawaan yang sudah ada "
                             "tidak tertimpa.")
    args = parser.parse_args()

    device = torch.device("cpu")
    ckpt_path = CFG.CHECKPOINT_DIR / CHECKPOINT_NAME
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint tidak ditemukan: {ckpt_path}")
        return

    print("=" * 70)
    print("TAHAP 1: HITUNG SKOR L1 DAN ENTROPI RATA-RATA PER LAPISAN")
    print("=" * 70)
    model_baseline, _ = load_checkpoint(ckpt_path, device)
    model_baseline.eval()

    SPLIT_DIR = CFG.DATASET_DIR.parent / "dataset_split"
    dataloaders, dataset_sizes = get_dataloaders(SPLIT_DIR)

    l1_scores = compute_l1_scores(model_baseline)
    entropy_scores = compute_entropy_scores(
        model_baseline, dataloaders["train"], device, samples_per_class=20, seed=CFG.SEED
    )
    entropi_rata2 = hitung_entropi_rata_rata_per_layer(entropy_scores)
    jumlah_channel = {idx: len(skor) for idx, skor in l1_scores.items()}

    if not args.jalankan_eksperimen:
        # ----- Mode tinjau: tabel 40% untuk alpha 0.5/1.0/2.0 + satu mode acak -----
        preview_data = {"checkpoint": CHECKPOINT_NAME, "rasio": 0.40, "alpha_diuji": ALPHA_PRATINJAU}
        alokasi_acuan_untuk_acak = None
        for alpha in ALPHA_PRATINJAU:
            alokasi, ringkasan = alokasikan_anggaran_entropi(entropi_rata2, jumlah_channel, 0.40, alpha)
            cetak_tabel_alokasi(alokasi, ringkasan, 0.40, f"MODE ENTROPI, alpha={alpha}")
            preview_data[f"entropi_alpha_{alpha}"] = {"alokasi": alokasi, "ringkasan": ringkasan}
            if alpha == 1.0:
                alokasi_acuan_untuk_acak = alokasi

        if alokasi_acuan_untuk_acak is None:
            alokasi_acuan_untuk_acak = alokasi  # fallback: alpha terakhir yang diuji

        pembanding_l1_40 = muat_pembanding_l1(40)
        target_num_params_40 = pembanding_l1_40["num_params"] if pembanding_l1_40 else None
        alokasi_acak, ringkasan_acak = alokasikan_anggaran_acak(
            alokasi_acuan_untuk_acak, jumlah_channel, 0.40,
            l1_scores=l1_scores, model_baseline=model_baseline,
            target_num_params=target_num_params_40,
        )
        cetak_tabel_alokasi(alokasi_acak, ringkasan_acak, 0.40, "MODE ACAK (dipadankan ke alpha=1.0)")
        preview_data["acak_dipadankan_alpha_1.0"] = {"alokasi": alokasi_acak, "ringkasan": ringkasan_acak}

        with open(ALOKASI_PREVIEW_PATH, "w", encoding="utf-8") as f:
            json.dump(preview_data, f, indent=2, ensure_ascii=False)
        print(f"\n[INFO] Tabel pratinjau disimpan: {ALOKASI_PREVIEW_PATH}")

        print("\n[INFO] Mode TINJAU SAJA (tanpa --jalankan-eksperimen). Berhenti di sini.")
        print("[INFO] Tidak ada pemangkasan atau fine-tuning yang dijalankan.")
        print("[INFO] Jalankan ulang dengan --jalankan-eksperimen --mode {entropi,acak} --alpha X untuk mode penuh.")
        return

    # ----- Tahap 2: jalankan untuk seluruh rasio -----
    print(f"\n{'=' * 70}")
    print(f"TAHAP 2: PEMANGKASAN + FINE-TUNING (mode={args.mode}, alpha={args.alpha}, seed={args.seed}) "
          f"UNTUK RASIO {args.ratios}")
    print("=" * 70)

    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, encoding="utf-8") as f:
            semua_hasil = json.load(f)
    else:
        semua_hasil = {}
    semua_hasil.setdefault(args.mode, {})

    waktu_mulai_total = time.time()

    for rasio in args.ratios:
        rasio_persen = int(round(rasio * 100))
        print(f"\n{'=' * 70}")
        print(f"MODE {args.mode.upper()} -- RASIO {rasio_persen}% -- SEED {args.seed}")
        print("=" * 70)

        pembanding_l1 = muat_pembanding_l1(rasio_persen)

        alokasi_entropi, ringkasan_entropi = alokasikan_anggaran_entropi(
            entropi_rata2, jumlah_channel, rasio, args.alpha
        )
        akhiran_seed = "" if args.seed == CFG.SEED else f"_seed{args.seed}"
        if args.mode == "entropi":
            alokasi, ringkasan = alokasi_entropi, ringkasan_entropi
            cetak_tabel_alokasi(alokasi, ringkasan, rasio, f"MODE ENTROPI, alpha={args.alpha}")
            ckpt_name = f"alokasi_entropi_alpha{args.alpha}_{rasio_persen}pct_30ep{akhiran_seed}.pth"
        else:
            cetak_tabel_alokasi(alokasi_entropi, ringkasan_entropi, rasio,
                                 f"(ACUAN) MODE ENTROPI, alpha={args.alpha}")
            target_num_params = pembanding_l1["num_params"] if pembanding_l1 else None
            alokasi, ringkasan = alokasikan_anggaran_acak(
                alokasi_entropi, jumlah_channel, rasio,
                l1_scores=l1_scores, model_baseline=model_baseline,
                target_num_params=target_num_params,
            )
            cetak_tabel_alokasi(alokasi, ringkasan, rasio, "MODE ACAK")
            ckpt_name = f"alokasi_acak_{rasio_persen}pct_30ep{akhiran_seed}.pth"

        masks = buat_mask_dari_alokasi(l1_scores, alokasi)
        pruned_model = apply_pruning(model_baseline, masks)
        num_params_setelah_pangkas = count_parameters(pruned_model)
        print(f"\n[INFO] Jumlah parameter setelah pemangkasan (sebelum fine-tuning): "
              f"{num_params_setelah_pangkas:,}")

        set_seed(args.seed)
        waktu_mulai = time.time()
        pruned_model, history = train_model(
            model=pruned_model,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            num_epochs=CFG.FINETUNE_EPOCHS,
            lr=CFG.FINETUNE_LR,
            device=device,
            checkpoint_path=CFG.CHECKPOINT_DIR / ckpt_name,
            phase_name=f"Alokasi-{args.mode.capitalize()} {rasio_persen}% seed={args.seed}"
        )
        waktu_selesai = time.time()

        metrics, preds, labels = evaluate_model(pruned_model, dataloaders["test"], device)

        plot_confusion_matrix(
            metrics["confusion_matrix"],
            title=f"CM Alokasi-{args.mode.capitalize()} {rasio_persen}% seed={args.seed} "
                  f"(Acc: {metrics['accuracy']*100:.2f}%)",
            save_path=CFG.OUTPUT_DIR / f"cm_alokasi_{args.mode}_{rasio_persen}pct{akhiran_seed}.png"
        )
        print_results_table(metrics, f"Alokasi-{args.mode.capitalize()} {rasio_persen}% seed={args.seed}")

        # ----- Syarat perbandingan adil -----
        if pembanding_l1 is not None:
            num_params_l1 = pembanding_l1["num_params"]
            selisih_persen = abs(num_params_setelah_pangkas - num_params_l1) / num_params_l1 * 100
            perbandingan_valid = selisih_persen <= 1.0
            if not perbandingan_valid:
                catatan_perbandingan = (
                    f"TIDAK VALID: selisih jumlah parameter {selisih_persen:.2f}% "
                    f"(alokasi-{args.mode}={num_params_setelah_pangkas:,} vs "
                    f"ablation_l1={num_params_l1:,}) melebihi ambang 1%. "
                    f"Perbandingan akurasi pada rasio ini TIDAK ADIL, gunakan dengan hati-hati."
                )
                print(f"\n[PERINGATAN] {catatan_perbandingan}")
            else:
                catatan_perbandingan = (
                    f"VALID: selisih jumlah parameter {selisih_persen:.2f}% (dalam ambang 1%)."
                )
                print(f"\n[INFO] {catatan_perbandingan}")
        else:
            num_params_l1 = None
            perbandingan_valid = None
            catatan_perbandingan = (
                "Tidak bisa dibandingkan -- outputs/tabel_hasil_lengkap.json tidak "
                f"punya entri ablation_l1 untuk rasio {rasio_persen}%."
            )
            print(f"\n[PERINGATAN] {catatan_perbandingan}")

        durasi_detik = waktu_selesai - waktu_mulai
        entri = {
            "checkpoint": ckpt_name,
            "mode": args.mode,
            "alpha": args.alpha if args.mode == "entropi" else None,
            "seed": args.seed,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
            "num_params": num_params_setelah_pangkas,
            "model_size_mb": metrics["model_size_mb"],
            "flops": metrics["flops"],
            "inference_ms": metrics["inference_ms"],
            "durasi_finetuning_detik": durasi_detik,
            "alokasi_per_lapisan": alokasi,
            "ringkasan_alokasi": ringkasan,
            "perbandingan_l1": {
                "num_params_l1": num_params_l1,
                "l1_checkpoint": pembanding_l1["checkpoint"] if pembanding_l1 else None,
                "valid": perbandingan_valid,
                "catatan": catatan_perbandingan,
            },
        }
        if args.mode == "acak":
            entri["alokasi_entropi_acuan"] = alokasi_entropi
            entri["ringkasan_alokasi_entropi_acuan"] = ringkasan_entropi

        kunci_hasil = f"{rasio_persen}%{akhiran_seed}"
        semua_hasil[args.mode][kunci_hasil] = entri

        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(semua_hasil, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Hasil sementara disimpan: {RESULTS_PATH} (kunci: {args.mode}.{kunci_hasil})")
        print(f"[INFO] Durasi fine-tuning rasio {rasio_persen}%: {durasi_detik/60:.1f} menit")

    waktu_selesai_total = time.time()
    print(f"\n{'=' * 70}")
    print(f"[SELESAI] Total waktu tahap 2: {(waktu_selesai_total - waktu_mulai_total)/60:.1f} menit")
    print(f"Hasil lengkap: {RESULTS_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()