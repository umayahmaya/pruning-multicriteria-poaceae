"""
31_pencarian_bobot.py
Pencarian grid bobot w1+w2+w3=1 (langkah 0,05 -> 231 kombinasi, termasuk
ketiga titik sudut (1,0,0)/(0,1,0)/(0,0,1)) untuk fungsi skoring
I(c) = w1*S_L1 + w2*S_BN + w3*S_H, pada rasio pemangkasan 30% dan 40%.

Untuk TIAP kombinasi bobot dan TIAP rasio:
  1. Hitung I(c) dari skor L1, BN gamma, dan entropi (kalibrasi TRAIN,
     20 gambar/kelas, seed=CFG.SEED -- skor MENTAH dihitung SEKALI di
     awal karena tidak bergantung pada bobot maupun rasio; hanya
     compute_importance_scores() yang dipanggil ulang per kombinasi
     untuk menormalisasi-dan-menggabungkan ulang dengan bobot itu).
  2. Bangun mask pemangkasan dari I(c) lewat get_pruning_mask(ratio)
     (rasio SERAGAM per lapisan -- metodologi WSM standar tesis, BUKAN
     alokasi anggaran per-lapisan seperti 30_alokasi_anggaran_entropi.py
     -- sehingga jumlah parameter akhir pada rasio yang sama otomatis
     identik di seluruh 231 kombinasi, tidak perlu syarat perbandingan
     adil terpisah).
  3. Pangkas model baseline (apply_pruning) -- TANPA FINE-TUNING SAMA
     SEKALI.
  4. Evaluasi akurasi model yang baru dipangkas LANGSUNG pada SUBSET
     VALIDASI (dataset_split/val). Subset TEST TIDAK PERNAH disentuh
     skrip ini.

Tujuan: memetakan lanskap akurasi (proxy zero-shot, tanpa fine-tuning)
di seluruh permukaan bobot w1+w2+w3=1, untuk melihat apakah ada
kombinasi INTERIOR (ketiga bobot > 0) yang mengungguli ketiga titik
SUDUT (satu kriteria tunggal) -- bukti langsung apakah menggabungkan
tiga kriteria memberi manfaat di atas kriteria tunggal, SEBELUM biaya
fine-tuning penuh dikeluarkan untuk kombinasi manapun.

Evaluasi akurasi di sini SENGAJA tidak memakai evaluate_model() di
src/model.py -- fungsi itu juga membenchmark FLOPs dan 300 run waktu
inferensi di setiap panggilan, yang terlalu mahal dikalikan 462
kombinasi (231 bobot x 2 rasio). evaluasi_akurasi_val() di skrip ini
hanya melakukan forward pass + hitung akurasi/F1, tanpa proses lain.

Mode operasi:
  - TANPA --jalankan-eksperimen (bawaan): hitung skor mentah sekali,
    UKUR waktu nyata untuk ketiga titik sudut pada kedua rasio (6
    evaluasi sungguhan -- sekaligus jadi nilai pembanding titik sudut),
    lalu EKSTRAPOLASI total waktu untuk seluruh grid (231 kombinasi x 2
    rasio = 462 evaluasi) dan BERHENTI. Grid penuh TIDAK dijalankan.
  - DENGAN --jalankan-eksperimen: menjalankan seluruh grid, menyimpan
    hasil progresif ke outputs/, lalu melaporkan 10 kombinasi bobot
    dengan akurasi validasi tertinggi per rasio, posisi ketiga titik
    sudut di antara 231 kombinasi, dan apakah ada kombinasi interior
    (ketiga bobot > 0) yang mengungguli ketiga titik sudut.

Jalankan:
    venv/Scripts/python.exe scripts/31_pencarian_bobot.py
        (mode tinjau -- hanya perkiraan durasi + titik sudut, grid penuh TIDAK dijalankan)
    venv/Scripts/python.exe scripts/31_pencarian_bobot.py --jalankan-eksperimen
"""

import sys
import os
import json
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

from src.config import CFG
from src.dataset import get_dataloaders
from src.model import load_checkpoint, count_parameters
from src.pruning import (
    compute_l1_scores, compute_bn_scores, compute_entropy_scores,
    compute_importance_scores, get_pruning_mask, apply_pruning,
)

CHECKPOINT_NAME = "baseline_9class.pth"
RATIOS = [0.30, 0.40]
LANGKAH_BOBOT = 0.05
RESULTS_PATH = CFG.OUTPUT_DIR / "pencarian_bobot_results.json"

SUDUT = [
    (1.0, 0.0, 0.0, "L1 murni"),
    (0.0, 1.0, 0.0, "BN gamma murni"),
    (0.0, 0.0, 1.0, "Entropi murni"),
]


def buat_grid_bobot(langkah=LANGKAH_BOBOT):
    """Semua (w1, w2, w3) kelipatan `langkah` dengan w1+w2+w3=1 -- 231
    kombinasi untuk langkah=0.05 (n=20, C(n+2,2)=231)."""
    n = round(1.0 / langkah)
    if abs(n * langkah - 1.0) > 1e-9:
        raise ValueError(f"langkah={langkah} tidak membagi 1.0 secara genap (n={n})")
    grid = []
    for i in range(n + 1):
        for j in range(n + 1 - i):
            k = n - i - j
            grid.append((round(i * langkah, 10), round(j * langkah, 10), round(k * langkah, 10)))
    return grid


def evaluasi_akurasi_val(model, val_loader, device):
    """Evaluasi CEPAT (akurasi + F1 saja, TANPA FLOPs/waktu inferensi --
    beda dari evaluate_model() di src/model.py yang terlalu mahal untuk
    dipanggil 462 kali)."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = float(accuracy_score(all_labels, all_preds))
    f1 = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))
    return acc, f1


def jalankan_satu_kombinasi(model_baseline, l1_scores, bn_scores, entropy_scores,
                             w1, w2, w3, ratio, val_loader, device):
    importance = compute_importance_scores(l1_scores, bn_scores, entropy_scores, w1, w2, w3)
    masks = get_pruning_mask(importance, ratio)
    pruned_model = apply_pruning(model_baseline, masks)
    num_params = count_parameters(pruned_model)
    acc, f1 = evaluasi_akurasi_val(pruned_model, val_loader, device)
    return acc, f1, num_params


def label_bobot(w1, w2, w3):
    return f"({w1:.2f},{w2:.2f},{w3:.2f})"


def ringkas_ratio(hasil_kombinasi, sudut_hasil):
    """hasil_kombinasi: list of dict {w1,w2,w3,accuracy,f1_score,num_params}.
    sudut_hasil: list sejajar SUDUT, dict yang sama, untuk 3 titik sudut."""
    terurut = sorted(hasil_kombinasi, key=lambda r: r["accuracy"], reverse=True)
    top_10 = terurut[:10]

    akurasi_sudut = [s["accuracy"] for s in sudut_hasil]
    idx_sudut_terbaik = int(np.argmax(akurasi_sudut))
    sudut_terbaik = sudut_hasil[idx_sudut_terbaik]

    interior = [r for r in hasil_kombinasi if r["w1"] > 0 and r["w2"] > 0 and r["w3"] > 0]
    interior_terurut = sorted(interior, key=lambda r: r["accuracy"], reverse=True)
    interior_terbaik = interior_terurut[0] if interior_terurut else None

    # Non-sudut (interior ATAU tepi -- salah satu bobot 0 tapi bukan murni satu kriteria)
    non_sudut = [r for r in hasil_kombinasi
                 if not (r["w1"] in (0.0, 1.0) and r["w2"] in (0.0, 1.0) and r["w3"] in (0.0, 1.0))]
    non_sudut_terurut = sorted(non_sudut, key=lambda r: r["accuracy"], reverse=True)
    non_sudut_terbaik = non_sudut_terurut[0] if non_sudut_terurut else None

    interior_mengungguli = (
        interior_terbaik is not None and interior_terbaik["accuracy"] > sudut_terbaik["accuracy"]
    )
    non_sudut_mengungguli = (
        non_sudut_terbaik is not None and non_sudut_terbaik["accuracy"] > sudut_terbaik["accuracy"]
    )

    # Posisi (rank, 1-based) tiap titik sudut di antara seluruh kombinasi
    posisi_sudut = []
    for s in sudut_hasil:
        rank = 1 + sum(1 for r in hasil_kombinasi if r["accuracy"] > s["accuracy"])
        posisi_sudut.append({"w1": s["w1"], "w2": s["w2"], "w3": s["w3"],
                              "label": s["label"], "accuracy": s["accuracy"], "rank": rank})

    return {
        "top_10": top_10,
        "sudut": posisi_sudut,
        "sudut_terbaik": {"w1": sudut_terbaik["w1"], "w2": sudut_terbaik["w2"],
                           "w3": sudut_terbaik["w3"], "label": sudut_terbaik["label"],
                           "accuracy": sudut_terbaik["accuracy"]},
        "interior_terbaik": interior_terbaik,
        "interior_mengungguli_sudut": interior_mengungguli,
        "margin_interior_vs_sudut_persen_poin": (
            (interior_terbaik["accuracy"] - sudut_terbaik["accuracy"]) * 100
            if interior_terbaik is not None else None
        ),
        "non_sudut_terbaik": non_sudut_terbaik,
        "non_sudut_mengungguli_sudut": non_sudut_mengungguli,
        "margin_non_sudut_vs_sudut_persen_poin": (
            (non_sudut_terbaik["accuracy"] - sudut_terbaik["accuracy"]) * 100
            if non_sudut_terbaik is not None else None
        ),
    }


def cetak_ringkasan_ratio(ratio_persen, ringkasan):
    print(f"\n{'=' * 90}")
    print(f"RINGKASAN RASIO {ratio_persen}%")
    print("=" * 90)
    print("\nTop 10 kombinasi bobot (akurasi validasi tertinggi):")
    print(f"{'Peringkat':>9} {'w1(L1)':>8} {'w2(BN)':>8} {'w3(H)':>8} {'Akurasi Val':>12} {'F1 Val':>8} {'#Param':>10}")
    for i, r in enumerate(ringkasan["top_10"], 1):
        print(f"{i:>9} {r['w1']:>8.2f} {r['w2']:>8.2f} {r['w3']:>8.2f} "
              f"{r['accuracy']*100:>11.2f}% {r['f1_score']:>8.4f} {r['num_params']:>10,}")

    print("\nPosisi ketiga titik sudut (di antara 231 kombinasi):")
    for s in ringkasan["sudut"]:
        print(f"  {label_bobot(s['w1'], s['w2'], s['w3']):>16} {s['label']:<16} "
              f"akurasi={s['accuracy']*100:.2f}%  peringkat={s['rank']}/231")

    ib = ringkasan["interior_terbaik"]
    sb = ringkasan["sudut_terbaik"]
    print(f"\nSudut terbaik    : {label_bobot(sb['w1'], sb['w2'], sb['w3'])} {sb['label']} "
          f"-- akurasi={sb['accuracy']*100:.2f}%")
    if ib is not None:
        print(f"Interior terbaik : {label_bobot(ib['w1'], ib['w2'], ib['w3'])} "
              f"-- akurasi={ib['accuracy']*100:.2f}%  "
              f"(margin={ringkasan['margin_interior_vs_sudut_persen_poin']:+.2f} poin persen vs sudut terbaik)")
        if ringkasan["interior_mengungguli_sudut"]:
            print("  >> ADA kombinasi INTERIOR (ketiga bobot > 0) yang mengungguli seluruh titik sudut.")
        else:
            print("  >> TIDAK ADA kombinasi interior yang mengungguli seluruh titik sudut.")
    else:
        print("Interior terbaik : tidak ada kombinasi dengan ketiga bobot > 0 pada grid ini.")
    print("=" * 90)


def main():
    parser = argparse.ArgumentParser(
        description="Pencarian grid bobot w1+w2+w3=1 untuk I(c), evaluasi akurasi validasi tanpa fine-tuning"
    )
    parser.add_argument(
        "--jalankan-eksperimen", action="store_true",
        help="Jalankan seluruh grid (231 kombinasi x jumlah rasio). Tanpa flag ini, skrip "
             "HANYA mengukur waktu nyata pada ketiga titik sudut (per rasio) lalu "
             "mengekstrapolasi + menampilkan perkiraan durasi grid penuh, tanpa menjalankannya."
    )
    parser.add_argument("--ratios", type=float, nargs="+", default=RATIOS,
                        help="Rasio pemangkasan yang diuji (bawaan 0.30 0.40)")
    parser.add_argument("--step", type=float, default=LANGKAH_BOBOT,
                        help="Langkah grid bobot (bawaan 0.05 -> 231 kombinasi)")
    args = parser.parse_args()

    device = torch.device("cpu")
    ckpt_path = CFG.CHECKPOINT_DIR / CHECKPOINT_NAME
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint tidak ditemukan: {ckpt_path}")
        return

    print("=" * 70)
    print("PENCARIAN GRID BOBOT w1+w2+w3=1 (EVALUASI VALIDASI, TANPA FINE-TUNING)")
    print("=" * 70)
    model_baseline, _ = load_checkpoint(ckpt_path, device)
    model_baseline.eval()

    SPLIT_DIR = CFG.DATASET_DIR.parent / "dataset_split"
    dataloaders, _ = get_dataloaders(SPLIT_DIR)
    val_loader = dataloaders["val"]

    print("\n[INFO] Menghitung skor L1, BN gamma, entropi (SEKALI, dipakai ulang di seluruh grid)...")
    l1_scores = compute_l1_scores(model_baseline)
    bn_scores = compute_bn_scores(model_baseline)
    entropy_scores = compute_entropy_scores(
        model_baseline, dataloaders["train"], device, samples_per_class=20, seed=CFG.SEED
    )

    grid_bobot = buat_grid_bobot(args.step)
    total_kombinasi = len(grid_bobot) * len(args.ratios)
    print(f"\n[INFO] Grid bobot: {len(grid_bobot)} kombinasi (langkah={args.step}) "
          f"x {len(args.ratios)} rasio {args.ratios} = {total_kombinasi} evaluasi total")
    print("[INFO] Evaluasi memakai subset VALIDASI (dataset_split/val). Subset TEST tidak disentuh.")

    if not args.jalankan_eksperimen:
        # ----- Mode tinjau: ukur waktu nyata pada ketiga titik sudut -----
        print(f"\n{'=' * 70}")
        print("MODE TINJAU: kalibrasi waktu pada ketiga titik sudut (per rasio)")
        print("=" * 70)
        waktu_sampel = []
        for ratio in args.ratios:
            print(f"\n-- rasio {ratio*100:.0f}% --")
            for (w1, w2, w3, label) in SUDUT:
                t0 = time.time()
                acc, f1, num_params = jalankan_satu_kombinasi(
                    model_baseline, l1_scores, bn_scores, entropy_scores,
                    w1, w2, w3, ratio, val_loader, device
                )
                dt = time.time() - t0
                waktu_sampel.append(dt)
                print(f"  [{dt:5.2f}s] w={label_bobot(w1, w2, w3)} ({label:<16}) "
                      f"acc_val={acc*100:.2f}%  f1_val={f1:.4f}  params={num_params:,}")

        rata2 = float(np.mean(waktu_sampel))
        std_waktu = float(np.std(waktu_sampel))
        estimasi_detik = rata2 * total_kombinasi

        print(f"\n{'=' * 70}")
        print("PERKIRAAN DURASI GRID PENUH")
        print("=" * 70)
        print(f"  Kombinasi bobot        : {len(grid_bobot)}")
        print(f"  Rasio                  : {args.ratios} ({len(args.ratios)} rasio)")
        print(f"  Total evaluasi         : {total_kombinasi}")
        print(f"  Waktu/kombinasi (rerata): {rata2:.2f} detik (std={std_waktu:.2f}, "
              f"dari {len(waktu_sampel)} sampel kalibrasi titik sudut)")
        print(f"  Estimasi total         : {estimasi_detik/60:.1f} menit (~{estimasi_detik/3600:.2f} jam)")
        print("=" * 70)
        print("\n[INFO] Mode TINJAU SAJA. Grid penuh TIDAK dijalankan.")
        print("[INFO] Jalankan ulang dengan --jalankan-eksperimen untuk menjalankan grid penuh.")
        return

    # ----- Grid penuh -----
    print(f"\n{'=' * 70}")
    print(f"MENJALANKAN GRID PENUH: {total_kombinasi} evaluasi")
    print("=" * 70)

    semua_hasil = {
        "checkpoint": CHECKPOINT_NAME,
        "metodologi": {
            "grid_langkah": args.step,
            "jumlah_kombinasi_bobot": len(grid_bobot),
            "ratios": args.ratios,
            "kalibrasi_entropi": {"source_split": "train", "samples_per_class": 20, "seed": CFG.SEED},
            "evaluasi": "Subset VALIDASI (dataset_split/val), TANPA fine-tuning -- pruning "
                        "struktural langsung dari I(c) baseline (pangkas rasio seragam per "
                        "lapisan via get_pruning_mask), akurasi zero-shot pasca-pangkas.",
            "catatan": "Subset TEST tidak pernah dipakai skrip ini.",
        },
        "per_ratio": {},
        "status": "BERJALAN",
    }

    waktu_mulai_total = time.time()
    i_global = 0

    for ratio in args.ratios:
        ratio_persen = int(round(ratio * 100))
        print(f"\n{'=' * 70}")
        print(f"RASIO {ratio_persen}%")
        print("=" * 70)

        hasil_kombinasi = []
        waktu_mulai_ratio = time.time()
        for (w1, w2, w3) in grid_bobot:
            i_global += 1
            acc, f1, num_params = jalankan_satu_kombinasi(
                model_baseline, l1_scores, bn_scores, entropy_scores,
                w1, w2, w3, ratio, val_loader, device
            )
            hasil_kombinasi.append({
                "w1": w1, "w2": w2, "w3": w3,
                "accuracy": acc, "f1_score": f1, "num_params": num_params,
            })
            if i_global % 10 == 0 or i_global == total_kombinasi:
                elapsed = time.time() - waktu_mulai_total
                print(f"  [{i_global}/{total_kombinasi}] w={label_bobot(w1, w2, w3)} "
                      f"acc_val={acc*100:.2f}%  (waktu berjalan={elapsed/60:.1f} menit)")

        sudut_hasil = []
        for (w1, w2, w3, label) in SUDUT:
            cocok = next(r for r in hasil_kombinasi if r["w1"] == w1 and r["w2"] == w2 and r["w3"] == w3)
            sudut_hasil.append({**cocok, "label": label})

        ringkasan = ringkas_ratio(hasil_kombinasi, sudut_hasil)
        cetak_ringkasan_ratio(ratio_persen, ringkasan)

        semua_hasil["per_ratio"][f"{ratio_persen}%"] = {
            "semua_kombinasi": hasil_kombinasi,
            "ringkasan": ringkasan,
            "durasi_detik": time.time() - waktu_mulai_ratio,
        }

        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(semua_hasil, f, indent=2, ensure_ascii=False)
        print(f"\n[INFO] Hasil sementara disimpan: {RESULTS_PATH}")

    semua_hasil["status"] = "SELESAI"
    semua_hasil["durasi_total_detik"] = time.time() - waktu_mulai_total
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(semua_hasil, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print(f"[SELESAI] Total waktu: {(time.time() - waktu_mulai_total)/60:.1f} menit")
    print(f"Hasil lengkap: {RESULTS_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()