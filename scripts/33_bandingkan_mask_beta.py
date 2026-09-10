"""
33_bandingkan_mask_beta.py
Pemeriksaan MURAH (skor + mask saja, TANPA evaluasi model, TANPA
fine-tuning) sebelum mengeluarkan biaya fine-tuning untuk kombinasi
bobot power-law apa pun dari 32_bobot_per_rasio.py.

Untuk beta = 1, 10, dan 20 (bobot per rasio DIBACA dari
outputs/bobot_per_rasio_results.json, bagian power_law -- TIDAK dihitung
ulang di sini, memakai sumber yang sama yang sudah diverifikasi skrip
32), pada SETIAP rasio 10-70%:
  1. Hitung I(c) = w1*S_L1 + w2*S_BN + w3*S_H dengan bobot beta tsb
     (compute_importance_scores(), sama seperti seluruh eksperimen lain).
  2. Bangun mask pemangkasan (get_pruning_mask()) pada rasio itu.
  3. Bandingkan mask beta=10 dan beta=20 terhadap mask beta=1 (bobot
     paling dekat rumus asli/sepertiga-sepertiga) PADA RASIO YANG SAMA.
  4. Laporkan persentase channel yang keputusan pangkas/pertahankannya
     BERBEDA (dari total 7104 channel, dan rincian per lapisan).

Tujuan: memastikan beta=10 dan beta=20 benar-benar mengubah KEPUTUSAN
PEMANGKASAN (bukan cuma mengubah angka bobot tanpa efek praktis)
SEBELUM biaya fine-tuning (puluhan menit per kombinasi) dikeluarkan
untuk kombinasi manapun.

Skrip ini TIDAK memuat/mengevaluasi model yang dipangkas dan TIDAK
melatih apa pun -- murni operasi tensor pada skor dan mask (get_pruning_mask
per rasio menghasilkan jumlah channel dipangkas yang SAMA di setiap
lapisan untuk bobot manapun -- hanya CHANNEL MANA yang bisa berbeda),
seharusnya selesai dalam hitungan detik/menit selain waktu memuat model
dan kalibrasi entropi.

Jalankan:
    venv/Scripts/python.exe scripts/33_bandingkan_mask_beta.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import CFG
from src.dataset import get_dataloaders
from src.model import load_checkpoint
from src.pruning import (
    compute_l1_scores, compute_bn_scores, compute_entropy_scores,
    compute_importance_scores, get_pruning_mask,
)

CHECKPOINT_NAME = "baseline_9class.pth"
BOBOT_PATH = CFG.OUTPUT_DIR / "bobot_per_rasio_results.json"
RESULTS_PATH = CFG.OUTPUT_DIR / "bandingkan_mask_beta_results.json"
BETA_ACUAN = 1
BETA_DIBANDINGKAN = [10, 20]


def muat_bobot_power_law():
    """Baca bobot power-law per rasio dari outputs/bobot_per_rasio_results.json
    (dihasilkan 32_bobot_per_rasio.py) -- TIDAK menghitung ulang bobot apa pun."""
    with open(BOBOT_PATH, encoding="utf-8") as f:
        data = json.load(f)
    hasil = data["power_law"]["hasil"]
    bobot = {}
    for beta_str, entry in hasil.items():
        bobot[beta_str] = {
            b["rasio"]: (b["w1_l1"], b["w2_bn"], b["w3_entropy"]) for b in entry["per_rasio"]
        }
    return bobot


def hitung_persentase_beda(mask_a, mask_b):
    """mask_a, mask_b: dict {layer_idx: tensor boolean (True=pertahankan)}
    dengan layer_idx dan panjang yang sama persis (rasio sama -> jumlah
    channel dipangkas per lapisan identik, hanya isinya yang mungkin beda)."""
    total_channel = 0
    total_beda = 0
    per_layer = []
    for idx in sorted(mask_a):
        a = mask_a[idx]
        b = mask_b[idx]
        beda = int((a != b).sum())
        n = len(a)
        total_channel += n
        total_beda += beda
        per_layer.append({
            "layer_idx": idx, "jumlah_channel": n, "jumlah_beda": beda,
            "persen_beda": (beda / n * 100) if n else 0.0,
        })
    return total_beda, total_channel, per_layer


def main():
    if not BOBOT_PATH.exists():
        print(f"[ERROR] {BOBOT_PATH} tidak ditemukan.")
        print("Jalankan 32_bobot_per_rasio.py terlebih dahulu.")
        return

    device = torch.device("cpu")
    ckpt_path = CFG.CHECKPOINT_DIR / CHECKPOINT_NAME
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint tidak ditemukan: {ckpt_path}")
        return

    print("=" * 70)
    print("PEMBANDINGAN MASK PEMANGKASAN ANTAR BETA (TANPA EVALUASI/FINE-TUNING)")
    print("=" * 70)

    model, _ = load_checkpoint(ckpt_path, device)
    model.eval()

    SPLIT_DIR = CFG.DATASET_DIR.parent / "dataset_split"
    dataloaders, _ = get_dataloaders(SPLIT_DIR)

    print("\n[INFO] Menghitung skor L1, BN gamma, entropi (sekali, dipakai ulang semua beta/rasio)...")
    l1_scores = compute_l1_scores(model)
    bn_scores = compute_bn_scores(model)
    entropy_scores = compute_entropy_scores(
        model, dataloaders["train"], device, samples_per_class=20, seed=CFG.SEED
    )

    bobot = muat_bobot_power_law()
    beta_acuan_str = str(BETA_ACUAN)
    kunci_dibutuhkan = [beta_acuan_str] + [str(b) for b in BETA_DIBANDINGKAN]
    for k in kunci_dibutuhkan:
        if k not in bobot:
            print(f"[ERROR] beta={k} tidak ditemukan di {BOBOT_PATH}.")
            return

    ratios_tersedia = sorted(bobot[beta_acuan_str].keys(), key=lambda k: int(k.rstrip("%")))

    hasil_semua = {
        "checkpoint": CHECKPOINT_NAME,
        "sumber_bobot": str(BOBOT_PATH),
        "catatan": "Bobot power-law dibaca dari outputs/bobot_per_rasio_results.json (hasil "
                   "32_bobot_per_rasio.py), TIDAK dihitung ulang di sini. Skrip ini TIDAK "
                   "mengevaluasi akurasi model apa pun dan TIDAK melatih apa pun -- murni "
                   "membandingkan mask pemangkasan (boolean per channel).",
        "beta_acuan": BETA_ACUAN,
        "beta_dibandingkan": BETA_DIBANDINGKAN,
        "per_rasio": {},
    }

    print(f"\n{'=' * 90}")
    print(f"TABEL PERSENTASE CHANNEL YANG BERBEDA (vs mask beta={BETA_ACUAN}, rasio sama)")
    print("=" * 90)
    header = f"{'Rasio':>6}" + "".join(f"{'beta=' + str(b):>16}" for b in BETA_DIBANDINGKAN)
    print(header)
    print("-" * 90)

    for ratio_key in ratios_tersedia:
        ratio = int(ratio_key.rstrip("%")) / 100.0
        w1a, w2a, w3a = bobot[beta_acuan_str][ratio_key]
        importance_acuan = compute_importance_scores(l1_scores, bn_scores, entropy_scores, w1a, w2a, w3a)
        mask_acuan = get_pruning_mask(importance_acuan, ratio)

        baris = {
            "rasio": ratio_key,
            "bobot_acuan_beta1": {"w1": w1a, "w2": w2a, "w3": w3a},
            "dibandingkan": {},
        }
        kolom_persen = []
        for beta in BETA_DIBANDINGKAN:
            w1b, w2b, w3b = bobot[str(beta)][ratio_key]
            importance_b = compute_importance_scores(l1_scores, bn_scores, entropy_scores, w1b, w2b, w3b)
            mask_b = get_pruning_mask(importance_b, ratio)

            total_beda, total_channel, per_layer = hitung_persentase_beda(mask_acuan, mask_b)
            persen = total_beda / total_channel * 100

            baris["dibandingkan"][str(beta)] = {
                "bobot": {"w1": w1b, "w2": w2b, "w3": w3b},
                "jumlah_channel_beda": total_beda,
                "total_channel": total_channel,
                "persen_beda": persen,
                "per_layer": per_layer,
            }
            kolom_persen.append(persen)

        print(f"{ratio_key:>6}" + "".join(f"{p:>15.2f}%" for p in kolom_persen))
        hasil_semua["per_rasio"][ratio_key] = baris

    print("=" * 90)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(hasil_semua, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Hasil disimpan: {RESULTS_PATH}")


if __name__ == "__main__":
    main()