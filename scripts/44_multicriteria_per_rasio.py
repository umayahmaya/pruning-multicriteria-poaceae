"""
44_multicriteria_per_rasio.py
Multi-kriteria pruning dengan BOBOT BERBEDA DI SETIAP RASIO (bukan satu
bobot konstan seperti 12_multicriteria_valweights.py), komposisi L1,
GM-Ekspansi (43_ablation_gm_expansion.py), dan Entropi:

    I(c) = w_L1(r) * S_norm_L1(c) + w_GM(r) * S_norm_GM(c) + w_Ent(r) * S_norm_Ent(c)

Bobot w_k(r) DITURUNKAN per rasio r dari akurasi VALIDASI ketiga
ablation kriteria tunggal pada rasio yang SAMA:

    d_k(r) = Acc_k(r) - min(Acc_L1(r), Acc_GM(r), Acc_Ent(r)) + 0.01
    w_k(r) = d_k(r) / (d_L1(r) + d_GM(r) + d_Ent(r))

(selisih akurasi terhadap kriteria terlemah pada rasio itu, plus 0.01
supaya kriteria terlemah tetap dapat bobot kecil positif -- bukan nol).
Sumber akurasi:
    - L1, Entropi : outputs/ablation_val_results.json (ablation_val_results.{r}%.l1/.entropy)
    - GM-Ekspansi : outputs/ablation_gm_expansion.json (results.{r}%.validasi.accuracy)
Skor L1/GM-Ekspansi/Entropi MENTAH (sebelum normalisasi) sendiri TIDAK
bergantung rasio -- dihitung SEKALI dari checkpoints/baseline_9class.pth
dan dipakai ulang untuk ketujuh rasio; yang beda per rasio hanyalah
BOBOT w_k(r) dan rasio pemangkasan itu sendiri.

Fungsi skor GM-Ekspansi (hitung_skor_gm_ekspansi_semua_layer, beserta
cari_conv_ekspansi yang dipakainya) DIIMPOR LANGSUNG dari
43_ablation_gm_expansion.py (dimuat dinamis lewat importlib, karena nama
file skrip diawali angka sehingga tidak bisa di-`import` biasa) -- TIDAK
disalin ulang, TIDAK mengubah skrip 43 sama sekali. L1 dan Entropi
memakai compute_l1_scores()/compute_entropy_scores() dari src/pruning.py
apa adanya (kalibrasi entropi: train, 20 citra/kelas, seed=CFG.SEED,
sama seperti semua eksperimen lain).

I(c) dibangun lewat compute_importance_scores() di src/pruning.py APA
ADANYA (fungsi generik: w1*norm(arg1)+w2*norm(arg2)+w3*norm(arg3)), tapi
argumen diisi (l1_scores, gm_scores, entropy_scores) dengan
(w1=w_L1, w2=w_GM, w3=w_Ent) -- src/pruning.py TIDAK diubah. Baris log
"[WSM] w2(BN)=.../w3(H)=..." yang tercetak fungsi itu SEBENARNYA berarti
bobot GM-Ekspansi dan Entropi -- diklarifikasi eksplisit sebelum tiap
panggilan.

Prosedur fine-tuning SAMA PERSIS dengan 03_ablation_study.py/
38_multikriteria_gm.py: 30 epoch (CFG.BASELINE_EPOCHS), CFG.FINETUNE_LR,
Adam+StepLR (train_model()), seed=CFG.SEED (42) dipanggil SEKALI sebelum
loop rasio -- keterbatasan metodologis yang sama berlaku (CLAUDE.md
Bagian 9). Evaluasi VALIDASI *dan* TEST terpisah.

Checkpoint memakai nama multicriteria_per_rasio_{rasio}pct_30ep.pth --
baru, tidak menimpa apa pun. Hasil disimpan progresif ke
outputs/multicriteria_per_rasio.json (skip rasio yang checkpoint-nya
sudah ada).

Setelah ketujuh rasio selesai, skrip membaca hasil L1, GM-Ekspansi, dan
Entropi tunggal yang SUDAH ADA (TIDAK dihitung ulang) untuk tabel
perbandingan empat kolom (L1 saja, GM saja, Entropi saja, Multi-Kriteria
per-rasio) x val/test x 7 rasio.

Mode operasi:
  - TANPA --jalankan-eksperimen (bawaan): tampilkan TABEL BOBOT PER
    RASIO (selalu, di kedua mode) dan perkiraan durasi, lalu BERHENTI.
    Tidak melatih apa pun, model baseline TIDAK dimuat.
  - DENGAN --jalankan-eksperimen: menjalankan ketujuh rasio (skip yang
    checkpoint-nya sudah ada), lalu tabel perbandingan empat kolom.

Jalankan:
    venv/Scripts/python.exe scripts/44_multicriteria_per_rasio.py
        (mode tinjau -- tabel bobot + perkiraan durasi, tidak melatih apa pun)
    venv/Scripts/python.exe scripts/44_multicriteria_per_rasio.py --jalankan-eksperimen
"""

import sys
import os
import json
import argparse
import random
import time
import importlib.util

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.config import CFG
from src.dataset import get_dataloaders
from src.model import train_model, evaluate_model, load_checkpoint, count_parameters
from src.pruning import (
    compute_l1_scores, compute_entropy_scores,
    compute_importance_scores, get_pruning_mask, apply_pruning,
)
from src.visualize import plot_confusion_matrix, print_results_table

# ----- Impor dinamis hitung_skor_gm_ekspansi_semua_layer dari skrip 43 -----
# (nama file diawali angka -> tidak bisa `import` biasa; TIDAK mengubah
# skrip 43, hanya memuat definisinya sebagai modul terpisah)
_JALUR_SKRIP_43 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "43_ablation_gm_expansion.py")
_spec_43 = importlib.util.spec_from_file_location("_skrip_43_ablation_gm_expansion", _JALUR_SKRIP_43)
_mod_43 = importlib.util.module_from_spec(_spec_43)
_spec_43.loader.exec_module(_mod_43)
hitung_skor_gm_ekspansi_semua_layer = _mod_43.hitung_skor_gm_ekspansi_semua_layer

CHECKPOINT_NAME = "baseline_9class.pth"
VAL_RESULTS_PATH = CFG.OUTPUT_DIR / "ablation_val_results.json"
GM_EXPANSION_RESULTS_PATH = CFG.OUTPUT_DIR / "ablation_gm_expansion.json"
TABEL_LENGKAP_PATH = CFG.OUTPUT_DIR / "tabel_hasil_lengkap.json"
RESULTS_PATH = CFG.OUTPUT_DIR / "multicriteria_per_rasio.json"
EPOCHS = CFG.BASELINE_EPOCHS  # 30
KONSTANTA_D = 0.01
MENIT_PER_RASIO_HISTORIS = (43.75 + 45.95 + 71.67) / 3


def set_seed(seed=CFG.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def muat_akurasi_val_semua_rasio():
    """{ratio_key: {"l1":.., "gm":.., "entropi":..}} akurasi VALIDASI
    ketiga kriteria tunggal, dari sumber yang sudah ada."""
    hasil = {}
    if VAL_RESULTS_PATH.exists():
        with open(VAL_RESULTS_PATH, encoding="utf-8") as f:
            val_data = json.load(f)
        for ratio_key, entry in val_data.get("ablation_val_results", {}).items():
            hasil.setdefault(ratio_key, {})
            if "l1" in entry:
                hasil[ratio_key]["l1"] = entry["l1"]["accuracy"]
            if "entropy" in entry:
                hasil[ratio_key]["entropi"] = entry["entropy"]["accuracy"]
    if GM_EXPANSION_RESULTS_PATH.exists():
        with open(GM_EXPANSION_RESULTS_PATH, encoding="utf-8") as f:
            gm_data = json.load(f)
        for ratio_key, entry in gm_data.get("results", {}).items():
            hasil.setdefault(ratio_key, {})
            hasil[ratio_key]["gm"] = entry["validasi"]["accuracy"]
    return hasil


def hitung_bobot_per_rasio(akurasi_val):
    """d_k(r) = Acc_k(r) - min(Acc_L1,Acc_GM,Acc_Ent)(r) + 0.01
    w_k(r) = d_k(r) / sum(d)(r)."""
    bobot = {}
    for ratio_key, acc in akurasi_val.items():
        if not all(k in acc for k in ("l1", "gm", "entropi")):
            continue
        acc_l1, acc_gm, acc_ent = acc["l1"], acc["gm"], acc["entropi"]
        acc_min = min(acc_l1, acc_gm, acc_ent)
        d_l1 = acc_l1 - acc_min + KONSTANTA_D
        d_gm = acc_gm - acc_min + KONSTANTA_D
        d_ent = acc_ent - acc_min + KONSTANTA_D
        total = d_l1 + d_gm + d_ent
        bobot[ratio_key] = {
            "acc_l1_val": acc_l1, "acc_gm_val": acc_gm, "acc_entropi_val": acc_ent,
            "d_l1": d_l1, "d_gm": d_gm, "d_entropi": d_ent,
            "w_l1": d_l1 / total, "w_gm": d_gm / total, "w_entropi": d_ent / total,
        }
    return bobot


def cetak_tabel_bobot(bobot_per_rasio):
    print(f"\n{'=' * 110}")
    print("TABEL BOBOT PER RASIO (dari akurasi VALIDASI ablation tunggal)")
    print("d_k(r) = Acc_k(r) - min(Acc_L1,Acc_GM,Acc_Ent)(r) + 0.01 ; w_k(r) = d_k(r) / sum(d)(r)")
    print("=" * 110)
    header = (f"{'Rasio':>6} {'AccL1':>8} {'AccGM':>8} {'AccEnt':>8}   "
              f"{'d_L1':>7} {'d_GM':>7} {'d_Ent':>7}   {'w_L1':>7} {'w_GM':>7} {'w_Ent':>7}")
    print(header)
    print("-" * 110)
    for ratio in CFG.PRUNING_RATIOS:
        ratio_key = f"{int(ratio * 100)}%"
        b = bobot_per_rasio.get(ratio_key)
        if b is None:
            print(f"{ratio_key:>6}   (data akurasi validasi tidak lengkap -- dilewati)")
            continue
        print(f"{ratio_key:>6} {b['acc_l1_val']*100:>7.2f}% {b['acc_gm_val']*100:>7.2f}% "
              f"{b['acc_entropi_val']*100:>7.2f}%   "
              f"{b['d_l1']:>7.4f} {b['d_gm']:>7.4f} {b['d_entropi']:>7.4f}   "
              f"{b['w_l1']:>7.4f} {b['w_gm']:>7.4f} {b['w_entropi']:>7.4f}")
    print("=" * 110)


def muat_num_params_l1(rasio_persen):
    if not TABEL_LENGKAP_PATH.exists():
        return None
    with open(TABEL_LENGKAP_PATH, encoding="utf-8") as f:
        tabel = json.load(f)
    entry = tabel.get("results", {}).get("l1", {}).get(f"{rasio_persen}%")
    return entry["num_params"] if entry else None


def run_single_ratio(model_baseline, l1_scores, gm_scores, entropy_scores,
                      ratio, w, dataloaders, dataset_sizes, device):
    ratio_persen = int(round(ratio * 100))
    ckpt_name = f"multicriteria_per_rasio_{ratio_persen}pct_{EPOCHS}ep.pth"
    ckpt_path = CFG.CHECKPOINT_DIR / ckpt_name
    w_l1, w_gm, w_ent = w["w_l1"], w["w_gm"], w["w_entropi"]

    print(f"\n[INFO] Rasio {ratio_persen}%: compute_importance_scores(l1, gm, entropi, "
          f"w1={w_l1:.4f}, w2={w_gm:.4f}, w3={w_ent:.4f}) -- label 'w2(BN)'/'w3(H)' di baris "
          f"[WSM] berikut SEBENARNYA berarti bobot GM-Ekspansi dan Entropi.")
    importance_scores = compute_importance_scores(l1_scores, gm_scores, entropy_scores,
                                                    w1=w_l1, w2=w_gm, w3=w_ent)
    masks = get_pruning_mask(importance_scores, ratio)

    if ckpt_path.exists():
        print(f"\n{'=' * 60}")
        print(f"SKIP: {ratio_persen}% (checkpoint sudah ada: {ckpt_name})")
        print(f"{'=' * 60}")
        pruned_model, _ = load_checkpoint(ckpt_path, device)
        num_params = count_parameters(pruned_model)
    else:
        print(f"\n{'=' * 60}")
        print(f"MULTI-KRITERIA PER-RASIO: {ratio_persen}% "
              f"w=(L1={w_l1:.4f}, GM={w_gm:.4f}, Ent={w_ent:.4f})")
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
            phase_name=f"Multi-Kriteria-PerRasio {ratio_persen}%"
        )

    print(f"\n[INFO] Evaluasi VALIDASI -- {ratio_persen}%")
    metrics_val, _, _ = evaluate_model(pruned_model, dataloaders["val"], device)
    print_results_table(metrics_val, f"Multi-Kriteria-PerRasio {ratio_persen}% -- VALIDASI")

    print(f"\n[INFO] Evaluasi TEST -- {ratio_persen}%")
    metrics_test, _, _ = evaluate_model(pruned_model, dataloaders["test"], device)
    print_results_table(metrics_test, f"Multi-Kriteria-PerRasio {ratio_persen}% -- TEST")

    plot_confusion_matrix(
        metrics_test["confusion_matrix"],
        title=f"CM Multi-Kriteria-PerRasio {ratio_persen}% (Acc: {metrics_test['accuracy']*100:.2f}%)",
        save_path=CFG.OUTPUT_DIR / f"cm_multicriteria_per_rasio_{ratio_persen}pct.png"
    )

    num_params_l1 = muat_num_params_l1(ratio_persen)
    if num_params_l1 is not None:
        identik = (num_params == num_params_l1)
        catatan = (f"IDENTIK dengan ablation_l1 ({num_params_l1:,})" if identik
                   else f"TIDAK IDENTIK -- {num_params:,} vs ablation_l1={num_params_l1:,}")
    else:
        identik = None
        catatan = f"Tidak bisa dikonfirmasi -- {TABEL_LENGKAP_PATH} tidak punya entri ablation_l1."
    print(f"\n[INFO] Jumlah parameter: {num_params:,} -- {catatan}")

    return {
        "bobot": {"w_l1": w_l1, "w_gm": w_gm, "w_entropi": w_ent},
        "checkpoint": ckpt_name,
        "validasi": {"accuracy": metrics_val["accuracy"], "f1_score": metrics_val["f1_score"]},
        "test": {"accuracy": metrics_test["accuracy"], "f1_score": metrics_test["f1_score"]},
        "num_params": num_params,
        "model_size_mb": metrics_test["model_size_mb"],
        "flops": metrics_test["flops"],
        "inference_ms": metrics_test["inference_ms"],
        "num_params_l1_rasio_sama": num_params_l1,
        "num_params_identik_dengan_l1": identik,
        "catatan_perbandingan_params": catatan,
    }


def muat_kriteria_tunggal_existing():
    """Baca hasil L1, GM-Ekspansi, Entropi (traincal) tunggal yang SUDAH
    ADA -- TIDAK dihitung ulang. Test L1/Entropi dari tabel_hasil_lengkap.json,
    val dari ablation_val_results.json, GM-Ekspansi (val & test) dari
    ablation_gm_expansion.json."""
    hasil = {"l1": {}, "entropi": {}, "gm": {}}

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

    if GM_EXPANSION_RESULTS_PATH.exists():
        with open(GM_EXPANSION_RESULTS_PATH, encoding="utf-8") as f:
            gm_data = json.load(f)
        for ratio_key, entry in gm_data.get("results", {}).items():
            hasil["gm"].setdefault(ratio_key, {})["val"] = entry["validasi"]["accuracy"]
            hasil["gm"].setdefault(ratio_key, {})["test"] = entry["test"]["accuracy"]

    return hasil


def cetak_tabel_perbandingan(hasil_multi, existing):
    print(f"\n{'=' * 130}")
    print("TABEL PERBANDINGAN: L1 SAJA, GM-EKSPANSI SAJA, ENTROPI SAJA, MULTI-KRITERIA PER-RASIO")
    print("=" * 130)
    header = (f"{'Rasio':>6} {'L1 Val':>8} {'L1 Test':>8} {'GM Val':>8} {'GM Test':>8} "
              f"{'Ent Val':>8} {'Ent Test':>9} {'Multi Val':>10} {'Multi Test':>11}")
    print(header)
    print("-" * 130)
    for ratio in CFG.PRUNING_RATIOS:
        ratio_key = f"{int(ratio * 100)}%"

        def fmt(x):
            return f"{x*100:.2f}%" if x is not None else "N/A"

        l1v = existing["l1"].get(ratio_key, {}).get("val")
        l1t = existing["l1"].get(ratio_key, {}).get("test")
        gmv = existing["gm"].get(ratio_key, {}).get("val")
        gmt = existing["gm"].get(ratio_key, {}).get("test")
        env = existing["entropi"].get(ratio_key, {}).get("val")
        ent = existing["entropi"].get(ratio_key, {}).get("test")
        mv = hasil_multi.get(ratio_key, {}).get("validasi", {}).get("accuracy")
        mt = hasil_multi.get(ratio_key, {}).get("test", {}).get("accuracy")

        print(f"{ratio_key:>6} {fmt(l1v):>8} {fmt(l1t):>8} {fmt(gmv):>8} {fmt(gmt):>8} "
              f"{fmt(env):>8} {fmt(ent):>9} {fmt(mv):>10} {fmt(mt):>11}")
    print("=" * 130)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-kriteria (L1, GM-Ekspansi, Entropi) dengan bobot berbeda per rasio"
    )
    parser.add_argument(
        "--jalankan-eksperimen", action="store_true",
        help="Jalankan (atau lanjutkan) ketujuh rasio. Tanpa flag ini, skrip HANYA "
             "menampilkan tabel bobot per rasio dan perkiraan durasi, lalu berhenti."
    )
    args = parser.parse_args()

    print("=" * 70)
    print("MULTI-KRITERIA PER-RASIO: I(c) = w_L1(r)*L1 + w_GM(r)*GM-Ekspansi + w_Ent(r)*Entropi")
    print("=" * 70)

    # ----- Tabel bobot per rasio: SELALU ditampilkan -----
    akurasi_val = muat_akurasi_val_semua_rasio()
    bobot_per_rasio = hitung_bobot_per_rasio(akurasi_val)
    cetak_tabel_bobot(bobot_per_rasio)

    if len(bobot_per_rasio) != len(CFG.PRUNING_RATIOS):
        print(f"\n[PERINGATAN] Bobot lengkap hanya tersedia untuk {len(bobot_per_rasio)} dari "
              f"{len(CFG.PRUNING_RATIOS)} rasio -- periksa sumber data (ablation_val_results.json, "
              f"ablation_gm_expansion.json) untuk rasio yang hilang sebelum melanjutkan.")

    # ----- Rencana + perkiraan durasi -----
    sudah_ada, belum_ada = [], []
    for ratio in CFG.PRUNING_RATIOS:
        ckpt_name = f"multicriteria_per_rasio_{int(ratio*100)}pct_{EPOCHS}ep.pth"
        (sudah_ada if (CFG.CHECKPOINT_DIR / ckpt_name).exists() else belum_ada).append(ratio)

    print(f"\n{'=' * 70}")
    print("RENCANA")
    print("=" * 70)
    print(f"  Checkpoint sudah ada (di-skip dari training): "
          f"{[f'{r*100:.0f}%' for r in sudah_ada] if sudah_ada else '(tidak ada)'}")
    print(f"  Akan dilatih (30 epoch)                     : "
          f"{[f'{r*100:.0f}%' for r in belum_ada] if belum_ada else '(tidak ada)'}")

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
    device = torch.device("cpu")
    ckpt_path = CFG.CHECKPOINT_DIR / CHECKPOINT_NAME
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint tidak ditemukan: {ckpt_path}")
        return

    model_baseline, _ = load_checkpoint(ckpt_path, device)
    model_baseline.eval()

    SPLIT_DIR = CFG.DATASET_DIR.parent / "dataset_split"
    dataloaders, dataset_sizes = get_dataloaders(SPLIT_DIR)

    print("\n[INFO] Menghitung skor L1, GM-Ekspansi, dan Entropi (sekali, dipakai ulang "
          "seluruh rasio -- skor mentah tidak bergantung rasio, hanya bobotnya)...")
    l1_scores = compute_l1_scores(model_baseline)
    gm_scores = hitung_skor_gm_ekspansi_semua_layer(model_baseline)
    entropy_scores = compute_entropy_scores(
        model_baseline, dataloaders["train"], device, samples_per_class=20, seed=CFG.SEED
    )

    set_seed()

    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, encoding="utf-8") as f:
            hasil_multi = json.load(f).get("results", {})
        print(f"\n[INFO] Melanjutkan dari hasil sebagian yang sudah ada: {RESULTS_PATH}")
    else:
        hasil_multi = {}

    start_time = time.time()
    for i, ratio in enumerate(CFG.PRUNING_RATIOS, 1):
        ratio_key = f"{int(ratio * 100)}%"
        w = bobot_per_rasio.get(ratio_key)
        if w is None:
            print(f"\n[PERINGATAN] Bobot untuk rasio {ratio_key} tidak tersedia, rasio ini DILEWATI.")
            continue

        elapsed = time.time() - start_time
        print(f"\n>>> Rasio {i}/{len(CFG.PRUNING_RATIOS)} | {ratio_key} "
              f"| Waktu berjalan: {elapsed/60:.1f} menit")

        hasil = run_single_ratio(
            model_baseline, l1_scores, gm_scores, entropy_scores,
            ratio, w, dataloaders, dataset_sizes, device
        )
        hasil_multi[ratio_key] = hasil

        save_data = {
            "formula": "I(c) = w_L1(r)*S_norm_L1 + w_GM(r)*S_norm_GM-Ekspansi + w_Ent(r)*S_norm_Entropi",
            "rumus_bobot": "d_k(r) = Acc_k(r) - min(Acc_L1,Acc_GM,Acc_Ent)(r) + 0.01 ; "
                           "w_k(r) = d_k(r) / sum(d)(r)",
            "bobot_per_rasio": bobot_per_rasio,
            "seed": CFG.SEED,
            "epochs": EPOCHS,
            "catatan_evaluasi": "accuracy pada 'validasi' untuk keputusan metodologis; 'test' hanya "
                                 "untuk pelaporan, TIDAK dipakai untuk memilih apa pun.",
            "results": hasil_multi,
            "status": f"{i}/{len(CFG.PRUNING_RATIOS)} rasio diproses",
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
    print(f"[SELESAI] Multi-Kriteria Per-Rasio -- total waktu: {total_time/60:.1f} menit "
          f"({total_time/3600:.2f} jam)")
    print(f"Hasil: {RESULTS_PATH}")
    print("=" * 70)

    # ----- Tabel perbandingan empat kolom -----
    existing = muat_kriteria_tunggal_existing()
    cetak_tabel_perbandingan(hasil_multi, existing)


if __name__ == "__main__":
    main()
