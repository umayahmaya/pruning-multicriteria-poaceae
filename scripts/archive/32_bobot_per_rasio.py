"""
32_bobot_per_rasio.py
Menghitung bobot w1 (L1), w2 (BN gamma), w3 (Entropi) UNTUK SETIAP rasio
10 sampai 70 persen, memakai rumus yang SAMA PERSIS dengan
07_recompute_weights_val.py -- compute_optimal_weights() di src/pruning.py:

    w_k = Akurasi_k / (Akurasi_L1 + Akurasi_BN + Akurasi_Entropi)

Bedanya dengan skrip 07: skrip 07 hanya menghitung bobot pada SATU rasio
acuan (bawaan 30%) untuk dipakai sebagai bobot tunggal I(c) di seluruh
eksperimen utama. Skrip ini menghitung bobot yang SAMA -- rumus dan
sumber data identik -- untuk KETUJUH rasio, murni untuk melihat apakah
bobotnya benar-benar bervariasi antar rasio atau relatif stabil.

DIPERLUAS dengan dua keluarga rumus alternatif, murni sebagai analisis
sensitivitas terhadap CARA menurunkan bobot dari akurasi -- bukan
pengganti rumus resmi:

  1. POWER-LAW: w_k proporsional (Akurasi_k)^beta, dinormalisasi agar
     w1+w2+w3=1. beta=1 secara matematis IDENTIK dengan rumus rasio
     linear asli (w_k = acc_k / sum(acc)) -- skrip ini memverifikasi
     ini secara eksplisit sebagai pemeriksaan kebenaran (sanity check)
     sebelum mencetak tabel beta lain. beta lebih besar -> perbedaan
     akurasi kecil antar kriteria diperbesar secara tidak proporsional
     (kriteria dengan akurasi tertinggi mendominasi bobot).

  2. SOFTMAX BERSUHU: w_k proporsional exp(Akurasi_k / tau), dinormalisasi.
     tau lebih kecil -> alokasi lebih tajam (mendekati winner-take-all
     pada kriteria berakurasi tertinggi); tau lebih besar -> mendekati
     seragam. Distabilkan secara numerik (kurangi maksimum sebelum exp).

Sumber data: outputs/ablation_val_results.json (akurasi VALIDASI, hasil
07_recompute_weights_val.py -- checkpoint l1/bn dari ablation lama,
checkpoint entropy dari kalibrasi train yang sudah diperbaiki). Skrip
ini TIDAK melatih ulang, TIDAK mengevaluasi checkpoint apa pun -- murni
membaca JSON yang sudah ada dan menghitung rasio/pangkat/softmax akurasi.

Jalankan:
    venv/Scripts/python.exe scripts/32_bobot_per_rasio.py
"""

import sys
import os
import json
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CFG
from src.pruning import compute_optimal_weights

VAL_RESULTS_PATH = CFG.OUTPUT_DIR / "ablation_val_results.json"
RESULTS_PATH = CFG.OUTPUT_DIR / "bobot_per_rasio_results.json"

BETA_VALUES = [1, 5, 10, 20, 50]
TAU_VALUES = [0.05, 0.02, 0.01]


def ambil_akurasi_per_rasio(val_results):
    """Baca akurasi val ketiga kriteria tunggal untuk tiap rasio dari
    outputs/ablation_val_results.json. Tidak menghitung bobot apa pun --
    murni ekstraksi data mentah, dipakai ulang oleh ketiga rumus bobot."""
    baris = []
    for ratio in CFG.PRUNING_RATIOS:
        ratio_key = f"{int(ratio * 100)}%"
        entry = val_results.get(ratio_key)
        if entry is None or not all(c in entry for c in ("l1", "bn", "entropy")):
            print(f"[PERINGATAN] Data val rasio {ratio_key} tidak lengkap, dilewati.")
            continue
        baris.append({
            "rasio": ratio_key,
            "acc_l1": entry["l1"]["accuracy"],
            "acc_bn": entry["bn"]["accuracy"],
            "acc_entropy": entry["entropy"]["accuracy"],
        })
    return baris


def hitung_bobot_power(acc_l1, acc_bn, acc_entropy, beta):
    """w_k proporsional (Akurasi_k)^beta, dinormalisasi agar jumlahnya 1.
    beta=1 -> identik dengan w_k = acc_k / sum(acc) (rumus rasio linear asli)."""
    p1 = acc_l1 ** beta
    p2 = acc_bn ** beta
    p3 = acc_entropy ** beta
    total = p1 + p2 + p3
    return p1 / total, p2 / total, p3 / total


def hitung_bobot_softmax(acc_l1, acc_bn, acc_entropy, tau):
    """w_k proporsional exp(Akurasi_k / tau), dinormalisasi. Distabilkan
    secara numerik dengan mengurangi akurasi maksimum sebelum exp (tidak
    mengubah hasil setelah normalisasi, mencegah overflow/underflow)."""
    accs = (acc_l1, acc_bn, acc_entropy)
    m = max(accs)
    eksp = [math.exp((a - m) / tau) for a in accs]
    total = sum(eksp)
    return tuple(e / total for e in eksp)


def hitung_rentang(nama, vals, baris_ref):
    idx_min = vals.index(min(vals))
    idx_max = vals.index(max(vals))
    return {
        "nama": nama,
        "min": min(vals),
        "min_pada_rasio": baris_ref[idx_min]["rasio"],
        "maks": max(vals),
        "maks_pada_rasio": baris_ref[idx_max]["rasio"],
        "rentang": max(vals) - min(vals),
    }


def cetak_dan_kumpulkan(baris_akurasi, fungsi_bobot, judul):
    """Hitung w1/w2/w3 tiap rasio memakai fungsi_bobot, cetak tabel +
    rentang, dan kembalikan struktur siap-simpan (dict)."""
    baris = []
    for a in baris_akurasi:
        w1, w2, w3 = fungsi_bobot(a["acc_l1"], a["acc_bn"], a["acc_entropy"])
        baris.append({**a, "w1_l1": w1, "w2_bn": w2, "w3_entropy": w3})

    print(f"\n{'=' * 100}")
    print(f"TABEL BOBOT PER RASIO -- {judul}")
    print("=" * 100)
    header = (f"{'Rasio':>6} {'Akurasi L1':>11} {'Akurasi BN':>11} {'Akurasi Entropi':>16} "
              f"{'w1 (L1)':>9} {'w2 (BN)':>9} {'w3 (Entropi)':>13}")
    print(header)
    print("-" * 100)
    for b in baris:
        print(f"{b['rasio']:>6} {b['acc_l1']*100:>10.2f}% {b['acc_bn']*100:>10.2f}% "
              f"{b['acc_entropy']*100:>15.2f}% "
              f"{b['w1_l1']:>9.4f} {b['w2_bn']:>9.4f} {b['w3_entropy']:>13.4f}")
    print("=" * 100)

    w1_vals = [b["w1_l1"] for b in baris]
    w2_vals = [b["w2_bn"] for b in baris]
    w3_vals = [b["w3_entropy"] for b in baris]
    rentang_w1 = hitung_rentang("w1 (L1)", w1_vals, baris)
    rentang_w2 = hitung_rentang("w2 (BN)", w2_vals, baris)
    rentang_w3 = hitung_rentang("w3 (Entropi)", w3_vals, baris)

    print("\nRentang tiap bobot di seluruh rasio (10-70%):")
    print(f"{'Bobot':>14} {'Min':>8} {'(rasio)':>8} {'Maks':>8} {'(rasio)':>8} {'Rentang':>9}")
    print("-" * 70)
    for r in (rentang_w1, rentang_w2, rentang_w3):
        print(f"{r['nama']:>14} {r['min']:>8.4f} {r['min_pada_rasio']:>8} "
              f"{r['maks']:>8.4f} {r['maks_pada_rasio']:>8} {r['rentang']:>9.4f}")

    return {
        "per_rasio": baris,
        "rentang_bobot": {"w1_l1": rentang_w1, "w2_bn": rentang_w2, "w3_entropy": rentang_w3},
    }


def main():
    if not VAL_RESULTS_PATH.exists():
        print(f"[ERROR] {VAL_RESULTS_PATH} tidak ditemukan.")
        print("Jalankan 07_recompute_weights_val.py terlebih dahulu.")
        return

    with open(VAL_RESULTS_PATH, encoding="utf-8") as f:
        data = json.load(f)
    val_results = data.get("ablation_val_results", {})

    baris_akurasi = ambil_akurasi_per_rasio(val_results)
    if not baris_akurasi:
        print("[ERROR] Tidak ada data akurasi val yang lengkap untuk rasio manapun.")
        return

    print("=" * 70)
    print("BOBOT w1/w2/w3 PER RASIO -- RUMUS ASLI, POWER-LAW, DAN SOFTMAX BERSUHU")
    print("=" * 70)

    # ----- Bagian 0: RUMUS ASLI (rasio linear) via compute_optimal_weights() -----
    # Dipakai sebagai acuan sanity check beta=1, sekaligus mempertahankan
    # keluaran/format yang identik dengan versi skrip sebelum diperluas.
    print("\n" + "=" * 70)
    print("RUMUS ASLI (rasio linear, identik 07_recompute_weights_val.py)")
    print("=" * 70)
    baris_asli = []
    for a in baris_akurasi:
        w1, w2, w3 = compute_optimal_weights(a["acc_l1"], a["acc_bn"], a["acc_entropy"])
        baris_asli.append({**a, "w1_l1": w1, "w2_bn": w2, "w3_entropy": w3})

    print(f"\n{'=' * 100}")
    print("TABEL BOBOT PER RASIO -- RUMUS ASLI (rasio linear)")
    print("=" * 100)
    header = (f"{'Rasio':>6} {'Akurasi L1':>11} {'Akurasi BN':>11} {'Akurasi Entropi':>16} "
              f"{'w1 (L1)':>9} {'w2 (BN)':>9} {'w3 (Entropi)':>13}")
    print(header)
    print("-" * 100)
    for b in baris_asli:
        print(f"{b['rasio']:>6} {b['acc_l1']*100:>10.2f}% {b['acc_bn']*100:>10.2f}% "
              f"{b['acc_entropy']*100:>15.2f}% "
              f"{b['w1_l1']:>9.4f} {b['w2_bn']:>9.4f} {b['w3_entropy']:>13.4f}")
    print("=" * 100)

    w1_vals = [b["w1_l1"] for b in baris_asli]
    w2_vals = [b["w2_bn"] for b in baris_asli]
    w3_vals = [b["w3_entropy"] for b in baris_asli]
    rentang_asli = {
        "w1_l1": hitung_rentang("w1 (L1)", w1_vals, baris_asli),
        "w2_bn": hitung_rentang("w2 (BN)", w2_vals, baris_asli),
        "w3_entropy": hitung_rentang("w3 (Entropi)", w3_vals, baris_asli),
    }
    print("\nRentang tiap bobot di seluruh rasio (10-70%):")
    print(f"{'Bobot':>14} {'Min':>8} {'(rasio)':>8} {'Maks':>8} {'(rasio)':>8} {'Rentang':>9}")
    print("-" * 70)
    for r in rentang_asli.values():
        print(f"{r['nama']:>14} {r['min']:>8.4f} {r['min_pada_rasio']:>8} "
              f"{r['maks']:>8.4f} {r['maks_pada_rasio']:>8} {r['rentang']:>9.4f}")

    # ----- Sanity check: beta=1 harus identik dengan rumus asli -----
    print(f"\n{'=' * 70}")
    print("PEMERIKSAAN KEBENARAN: power-law beta=1 vs rumus asli (rasio linear)")
    print("=" * 70)
    selisih_maks_global = 0.0
    for a, b_asli in zip(baris_akurasi, baris_asli):
        w1_b1, w2_b1, w3_b1 = hitung_bobot_power(a["acc_l1"], a["acc_bn"], a["acc_entropy"], beta=1.0)
        selisih = max(abs(w1_b1 - b_asli["w1_l1"]), abs(w2_b1 - b_asli["w2_bn"]),
                      abs(w3_b1 - b_asli["w3_entropy"]))
        selisih_maks_global = max(selisih_maks_global, selisih)
        status = "OK" if selisih < 1e-9 else "BEDA -- PERIKSA RUMUS!"
        print(f"  rasio {a['rasio']:>4}: selisih maks = {selisih:.2e}  [{status}]")
    print(f"  Selisih maksimum di seluruh rasio: {selisih_maks_global:.2e} "
          f"({'lolos, dianggap identik' if selisih_maks_global < 1e-9 else 'GAGAL, tidak identik'})")
    print("=" * 70)

    # ----- Bagian 1: POWER-LAW untuk beta = 1, 5, 10, 20, 50 -----
    hasil_power = {}
    for beta in BETA_VALUES:
        hasil_power[str(beta)] = cetak_dan_kumpulkan(
            baris_akurasi,
            lambda l1, bn, h, beta=beta: hitung_bobot_power(l1, bn, h, beta),
            f"POWER-LAW, beta={beta}" + ("  (harus = rumus asli, lihat pemeriksaan di atas)" if beta == 1 else ""),
        )

    # ----- Bagian 2: SOFTMAX BERSUHU untuk tau = 0.05, 0.02, 0.01 -----
    hasil_softmax = {}
    for tau in TAU_VALUES:
        hasil_softmax[str(tau)] = cetak_dan_kumpulkan(
            baris_akurasi,
            lambda l1, bn, h, tau=tau: hitung_bobot_softmax(l1, bn, h, tau),
            f"SOFTMAX BERSUHU, tau={tau}",
        )

    # ----- Simpan -----
    save_data = {
        "sumber": str(VAL_RESULTS_PATH),
        "catatan": "Skrip ini TIDAK melatih atau mengevaluasi ulang apa pun -- murni menghitung "
                   "bobot dari akurasi validasi yang sudah ada di outputs/ablation_val_results.json, "
                   "untuk SETIAP rasio, dengan tiga rumus: rasio linear asli, power-law (beberapa "
                   "beta), dan softmax bersuhu (beberapa tau).",
        "rumus_asli": {
            "definisi": "w_k = Akurasi_k / (Akurasi_L1 + Akurasi_BN + Akurasi_Entropi) -- identik "
                        "compute_optimal_weights() di src/pruning.py, dipakai 07_recompute_weights_val.py",
            "per_rasio": baris_asli,
            "rentang_bobot": rentang_asli,
        },
        "sanity_check_beta1_vs_asli": {
            "selisih_maks_global": selisih_maks_global,
            "lolos": selisih_maks_global < 1e-9,
        },
        "power_law": {
            "definisi": "w_k proporsional (Akurasi_k)^beta, dinormalisasi agar w1+w2+w3=1. "
                        "beta=1 identik dengan rumus_asli.",
            "beta_diuji": BETA_VALUES,
            "hasil": hasil_power,
        },
        "softmax_bersuhu": {
            "definisi": "w_k proporsional exp(Akurasi_k / tau), dinormalisasi. tau kecil -> "
                        "alokasi tajam (mendekati winner-take-all pada akurasi tertinggi).",
            "tau_diuji": TAU_VALUES,
            "hasil": hasil_softmax,
        },
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"\n[INFO] Hasil disimpan: {RESULTS_PATH}")


if __name__ == "__main__":
    main()