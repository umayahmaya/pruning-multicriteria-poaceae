"""
34_multicriteria_beta20.py
Pruning multi-kriteria dengan bobot power-law beta=20 PER RASIO (dari
32_bobot_per_rasio.py / 33_bandingkan_mask_beta.py), HANYA untuk rasio
50% dan 60% -- dua rasio yang menurut 33_bandingkan_mask_beta.py punya
mask pemangkasan paling berbeda dari beta=1 (10,59%/50% dan 8,16%/60%
utk beta=10; 20,95%/50% dan 11,06%/60% utk beta=20 -- 50% adalah beda
TERBESAR dari seluruh 7 rasio).

Bobot DIBACA dari outputs/bobot_per_rasio_results.json bagian
power_law.hasil."20", TIDAK dihitung ulang di sini:
    Rasio 50%: w = (0.5939, 0.2427, 0.1634)
    Rasio 60%: w = (0.4419, 0.0529, 0.5052)
(Bobot BEDA per rasio -- bukan satu bobot konstan seperti
12_multicriteria_valweights.py, karena beta=20 menghasilkan bobot yang
memang dirancang bergantung rasio.)

Prosedur SAMA PERSIS dengan 12_multicriteria_valweights.py: skor L1/BN
gamma/entropi (kalibrasi train, 20 gambar/kelas, seed=CFG.SEED) dihitung
dari checkpoints/baseline_9class.pth, mask dibangun via get_pruning_mask
(rasio SERAGAM per lapisan -- metodologi WSM standar, bukan alokasi
anggaran per-lapisan seperti 30_alokasi_anggaran_entropi.py), fine-tuning
30 epoch (CFG.BASELINE_EPOCHS/FINETUNE_EPOCHS, keduanya 30) dengan
CFG.FINETUNE_LR + Adam + StepLR (train_model()), seed=CFG.SEED (42),
set_seed() dipanggil SEKALI di awal main() sebelum loop rasio --
IDENTIK dengan pola skrip 12 (lihat CLAUDE.md Bagian 9: keterbatasan
metodologis yang sama berlaku di sini, RNG saat fine-tuning rasio 60%
bergantung pada apa yang terjadi selama rasio 50%).

BEDA dari skrip 12: skrip ini mengevaluasi pada VALIDASI *dan* TEST
secara terpisah (skrip 12 hanya mengevaluasi test) -- validasi untuk
keputusan metodologis, test hanya untuk pelaporan, TIDAK dipakai untuk
memilih apa pun.

Karena get_pruning_mask() memangkas rasio SERAGAM per lapisan
(floor(jumlah_channel * rasio) di tiap lapisan) TERLEPAS dari bobot,
jumlah parameter akhir pada rasio yang sama HARUS identik dengan
ablation_l1 (dan dengan multicriteria bobot manapun) pada rasio itu --
skrip ini mengonfirmasi ini secara eksplisit terhadap
outputs/tabel_hasil_lengkap.json.

Checkpoint memakai akhiran _beta20 (multicriteria_beta20_{rasio}pct_30ep.pth),
TIDAK menimpa checkpoint manapun yang sudah ada. Hasil disimpan ke
outputs/multicriteria_beta20_results.json (baru, terpisah).

Jalankan:
    venv/Scripts/python.exe scripts/34_multicriteria_beta20.py
"""

import sys
import os
import json
import random
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CFG
from src.dataset import get_dataloaders
from src.model import train_model, evaluate_model, load_checkpoint, count_parameters
from src.pruning import (
    compute_l1_scores, compute_bn_scores, compute_entropy_scores,
    compute_importance_scores, get_pruning_mask, apply_pruning
)
from src.visualize import plot_confusion_matrix, print_results_table

BOBOT_PATH = CFG.OUTPUT_DIR / "bobot_per_rasio_results.json"
TABEL_LENGKAP_PATH = CFG.OUTPUT_DIR / "tabel_hasil_lengkap.json"
RESULTS_PATH = CFG.OUTPUT_DIR / "multicriteria_beta20_results.json"
BETA = 20
RATIOS_TO_RUN = [0.5, 0.6]
EPOCHS = CFG.BASELINE_EPOCHS  # 30
SUFFIX = "beta20"


def set_seed(seed=CFG.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def muat_bobot_beta20():
    """Baca bobot power-law beta=20 per rasio dari
    outputs/bobot_per_rasio_results.json (hasil 32_bobot_per_rasio.py) --
    TIDAK dihitung ulang di sini."""
    if not BOBOT_PATH.exists():
        print(f"[ERROR] {BOBOT_PATH} tidak ditemukan.")
        print("Jalankan 32_bobot_per_rasio.py terlebih dahulu.")
        return None
    with open(BOBOT_PATH, encoding="utf-8") as f:
        data = json.load(f)
    entry = data.get("power_law", {}).get("hasil", {}).get(str(BETA))
    if entry is None:
        print(f"[ERROR] beta={BETA} tidak ditemukan di {BOBOT_PATH}.")
        return None
    bobot = {b["rasio"]: (b["w1_l1"], b["w2_bn"], b["w3_entropy"]) for b in entry["per_rasio"]}
    return bobot


def muat_num_params_l1(rasio_persen):
    """num_params ablation_l1 pada rasio yang sama, untuk konfirmasi
    jumlah parameter (get_pruning_mask() memangkas rasio SERAGAM per
    lapisan terlepas dari bobot, sehingga jumlah parameter akhir pada
    rasio yang sama seharusnya IDENTIK untuk bobot manapun)."""
    if not TABEL_LENGKAP_PATH.exists():
        return None
    with open(TABEL_LENGKAP_PATH, encoding="utf-8") as f:
        tabel = json.load(f)
    entry = tabel.get("results", {}).get("l1", {}).get(f"{rasio_persen}%")
    return entry["num_params"] if entry else None


def run_single_ratio(model_baseline, l1_scores, bn_scores, entropy_scores,
                      w1, w2, w3, ratio, dataloaders, dataset_sizes, device, epochs):
    ratio_persen = int(round(ratio * 100))
    ratio_key = f"{ratio_persen}%"
    ckpt_name = f"multicriteria_{SUFFIX}_{ratio_persen}pct_{epochs}ep.pth"
    ckpt_path = CFG.CHECKPOINT_DIR / ckpt_name

    importance_scores = compute_importance_scores(l1_scores, bn_scores, entropy_scores, w1=w1, w2=w2, w3=w3)
    masks = get_pruning_mask(importance_scores, ratio)

    if ckpt_path.exists():
        print(f"\n{'=' * 60}")
        print(f"SKIP: {ratio_key} beta={BETA} (checkpoint sudah ada: {ckpt_name})")
        print(f"{'=' * 60}")
        pruned_model, _ = load_checkpoint(ckpt_path, device)
        num_params = count_parameters(pruned_model)
    else:
        print(f"\n{'=' * 60}")
        print(f"PRUNING MULTI-KRITERIA (beta={BETA}): {ratio_key}  w=({w1:.4f},{w2:.4f},{w3:.4f})")
        print(f"{'=' * 60}")

        pruned_model_sebelum_ft = apply_pruning(model_baseline, masks)
        num_params = count_parameters(pruned_model_sebelum_ft)
        print(f"[INFO] Jumlah parameter setelah pemangkasan (sebelum fine-tuning): {num_params:,}")

        pruned_model, history = train_model(
            model=pruned_model_sebelum_ft,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            num_epochs=epochs,
            lr=CFG.FINETUNE_LR,
            device=device,
            checkpoint_path=ckpt_path,
            phase_name=f"Multi-Kriteria (beta={BETA}) {ratio_key}"
        )

    # ----- Evaluasi VALIDASI dan TEST secara terpisah -----
    print(f"\n[INFO] Evaluasi VALIDASI -- {ratio_key} beta={BETA}")
    metrics_val, _, _ = evaluate_model(pruned_model, dataloaders["val"], device)
    print_results_table(metrics_val, f"Multi-Kriteria (beta={BETA}) {ratio_key} -- VALIDASI")

    print(f"\n[INFO] Evaluasi TEST -- {ratio_key} beta={BETA}")
    metrics_test, _, _ = evaluate_model(pruned_model, dataloaders["test"], device)
    print_results_table(metrics_test, f"Multi-Kriteria (beta={BETA}) {ratio_key} -- TEST")

    plot_confusion_matrix(
        metrics_test["confusion_matrix"],
        title=f"CM Multi-Kriteria (beta={BETA}) {ratio_key} TEST (Acc: {metrics_test['accuracy']*100:.2f}%)",
        save_path=CFG.OUTPUT_DIR / f"cm_multicriteria_{SUFFIX}_{ratio_persen}pct.png"
    )

    # ----- Konfirmasi jumlah parameter identik dengan ablation_l1 -----
    num_params_l1 = muat_num_params_l1(ratio_persen)
    if num_params_l1 is not None:
        identik = (num_params == num_params_l1)
        catatan_params = (
            f"IDENTIK dengan ablation_l1 ({num_params_l1:,})" if identik
            else f"TIDAK IDENTIK -- beta20={num_params:,} vs ablation_l1={num_params_l1:,} "
                 f"(selisih {abs(num_params - num_params_l1):,})"
        )
        print(f"\n[INFO] Jumlah parameter: {num_params:,} -- {catatan_params}")
    else:
        identik = None
        catatan_params = f"Tidak bisa dikonfirmasi -- {TABEL_LENGKAP_PATH} tidak punya entri ablation_l1 untuk {ratio_key}."
        print(f"\n[PERINGATAN] {catatan_params}")

    return {
        "bobot": {"w1": w1, "w2": w2, "w3": w3},
        "checkpoint": ckpt_name,
        "validasi": {
            "accuracy": metrics_val["accuracy"],
            "precision": metrics_val["precision"],
            "recall": metrics_val["recall"],
            "f1_score": metrics_val["f1_score"],
        },
        "test": {
            "accuracy": metrics_test["accuracy"],
            "precision": metrics_test["precision"],
            "recall": metrics_test["recall"],
            "f1_score": metrics_test["f1_score"],
        },
        "num_params": num_params,
        "model_size_mb": metrics_test["model_size_mb"],
        "flops": metrics_test["flops"],
        "inference_ms": metrics_test["inference_ms"],
        "num_params_l1_rasio_sama": num_params_l1,
        "num_params_identik_dengan_l1": identik,
        "catatan_perbandingan_params": catatan_params,
    }


def main():
    print("=" * 70)
    print(f"PRUNING MULTI-KRITERIA -- BOBOT POWER-LAW BETA={BETA} PER RASIO")
    print("=" * 70)

    bobot = muat_bobot_beta20()
    if bobot is None:
        return

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"  Device  : {device}")
    print(f"  Rasio   : {[f'{r*100:.0f}%' for r in RATIOS_TO_RUN]}")
    print(f"  Epoch   : {EPOCHS}")
    print(f"  Seed    : {CFG.SEED}")
    for r in RATIOS_TO_RUN:
        rk = f"{int(r*100)}%"
        w1, w2, w3 = bobot[rk]
        print(f"  Bobot {rk}: w=({w1:.4f}, {w2:.4f}, {w3:.4f})")

    SPLIT_DIR = CFG.DATASET_DIR.parent / "dataset_split"
    dataloaders, dataset_sizes = get_dataloaders(SPLIT_DIR)

    BASELINE_CKPT = CFG.CHECKPOINT_DIR / "baseline_9class.pth"
    if not BASELINE_CKPT.exists():
        print("[ERROR] Checkpoint baseline tidak ditemukan!")
        return
    model_baseline, _ = load_checkpoint(BASELINE_CKPT, device)

    print("\n[INFO] Menghitung tiga skor kepentingan channel (sekali, dipakai ulang kedua rasio)...")
    l1_scores = compute_l1_scores(model_baseline)
    bn_scores = compute_bn_scores(model_baseline)
    entropy_scores = compute_entropy_scores(model_baseline, dataloaders["train"], device)

    all_results = {
        "beta": BETA,
        "bobot_sumber": str(BOBOT_PATH),
        "calibration_source": "train",
        "seed": CFG.SEED,
        "epochs": EPOCHS,
        "catatan_evaluasi": "accuracy pada 'validasi' untuk keputusan metodologis; 'test' hanya "
                             "untuk pelaporan, TIDAK dipakai untuk memilih apa pun.",
        "results": {},
        "status": "BERJALAN",
    }

    start_time = time.time()
    for i, ratio in enumerate(RATIOS_TO_RUN, 1):
        ratio_key = f"{int(ratio * 100)}%"
        w1, w2, w3 = bobot[ratio_key]
        elapsed = time.time() - start_time
        print(f"\n>>> Eksperimen {i}/{len(RATIOS_TO_RUN)} | {ratio_key} "
              f"| Waktu berjalan: {elapsed/60:.1f} menit")

        hasil = run_single_ratio(
            model_baseline, l1_scores, bn_scores, entropy_scores,
            w1, w2, w3, ratio, dataloaders, dataset_sizes, device, EPOCHS
        )
        all_results["results"][ratio_key] = hasil
        all_results["status"] = f"{i}/{len(RATIOS_TO_RUN)} selesai"

        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Hasil sementara disimpan: {RESULTS_PATH}")

    total_time = time.time() - start_time
    all_results["status"] = "SELESAI"
    all_results["total_time_seconds"] = total_time
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 90}")
    print(f"RINGKASAN -- MULTI-KRITERIA BETA={BETA} ({EPOCHS} epoch, seed={CFG.SEED})")
    print("=" * 90)
    print(f"{'Rasio':<8} {'Val Acc':<10} {'Test Acc':<10} {'#Param':<12} {'Params=L1?':<12}")
    print("-" * 90)
    for rk, r in all_results["results"].items():
        print(f"{rk:<8} {r['validasi']['accuracy']*100:>7.2f}%  "
              f"{r['test']['accuracy']*100:>7.2f}%  "
              f"{r['num_params']:>10,}  "
              f"{'YA' if r['num_params_identik_dengan_l1'] else 'TIDAK':<12}")
    print("=" * 90)
    print(f"Total waktu: {total_time/60:.1f} menit ({total_time/3600:.2f} jam)")
    print(f"\n[INFO] Hasil disimpan: {RESULTS_PATH}")
    print("[INFO] Tidak ada checkpoint/hasil lama yang ditimpa.")


if __name__ == "__main__":
    main()