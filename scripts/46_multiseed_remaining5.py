"""
46_multiseed_remaining5.py
Melengkapi validasi multi-seed formula I(c) AKTIF (L1 + GM-Ekspansi +
Entropi, bobot w_k(r) per rasio -- lihat CLAUDE.md Bagian 2 dan
44_multicriteria_per_rasio.py) untuk LIMA rasio yang tersisa:
10%, 30%, 40%, 50%, 70%. Baseline, rasio 20%, dan rasio 60% sudah
selesai lewat 45_multiseed_per_rasio.py (TIDAK diulang di sini).

BEDA DARI 45_multiseed_per_rasio.py: di skrip itu seed 42 DIPAKAI ULANG
dari checkpoint 44_multicriteria_per_rasio.py (tidak dilatih ulang).
DI SINI, atas instruksi eksplisit pengguna, seed 42 DILATIH ULANG FRESH
(retrain independen) sama seperti seed 123 dan 2024 -- total 15 proses
training (5 rasio x 3 seed), bukan 10. Checkpoint memakai akhiran
_seed{seed} untuk KETIGA seed (termasuk 42) supaya TIDAK menimpa
checkpoint multicriteria_per_rasio_{rasio}pct_30ep.pth (tanpa akhiran
seed) yang sudah ada dari 44_multicriteria_per_rasio.py.

Bobot w_k(r) dimuat APA ADANYA dari outputs/multicriteria_per_rasio.json
(bobot_per_rasio), TIDAK dihitung ulang -- identik dengan cara
45_multiseed_per_rasio.py memuatnya. compute_l1_scores(),
compute_entropy_scores() (src/pruning.py) dan
hitung_skor_gm_ekspansi_semua_layer() (diimpor dinamis dari
43_ablation_gm_expansion.py) dipakai APA ADANYA, TIDAK diubah.

CATATAN METODOLOGIS (premis sama seperti 15/45): mask pemangkasan per
rasio dihitung SEKALI dari skor I(c) baseline dan IDENTIK di ketiga
seed untuk rasio yang sama -- variasi akurasi antar seed murni variasi
fine-tuning, bukan variasi pemilihan channel.

Hasil DITAMBAHKAN ke outputs/multiseed_per_rasio_results.json yang
SUDAH ADA -- entri baseline/20%/60% dari skrip 45 TIDAK dihapus atau
ditimpa. Disimpan progresif setiap SATU run selesai (15 kali total).

Urutan eksekusi: rasio 10 -> 30 -> 40 -> 50 -> 70, tiap rasio
seed 42 -> 123 -> 2024.

Jalankan:
    venv/Scripts/python.exe scripts/46_multiseed_remaining5.py
"""

import sys
import os
import json
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

# ----- Impor dinamis hitung_skor_gm_ekspansi_semua_layer dari skrip 43 -----
_JALUR_SKRIP_43 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "43_ablation_gm_expansion.py")
_spec_43 = importlib.util.spec_from_file_location("_skrip_43_ablation_gm_expansion", _JALUR_SKRIP_43)
_mod_43 = importlib.util.module_from_spec(_spec_43)
_spec_43.loader.exec_module(_mod_43)
hitung_skor_gm_ekspansi_semua_layer = _mod_43.hitung_skor_gm_ekspansi_semua_layer

SEEDS = [42, 123, 2024]
RATIOS_PCT = [10, 30, 40, 50, 70]
EPOCHS = CFG.BASELINE_EPOCHS  # 30
SPLIT_DIR = CFG.DATASET_DIR.parent / "dataset_split"

PER_RASIO_PATH = CFG.OUTPUT_DIR / "multicriteria_per_rasio.json"
RESULTS_PATH = CFG.OUTPUT_DIR / "multiseed_per_rasio_results.json"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def muat_bobot_per_rasio():
    """Baca bobot_per_rasio yang SUDAH ADA dari
    outputs/multicriteria_per_rasio.json -- tidak dihitung ulang."""
    if not PER_RASIO_PATH.exists():
        print(f"[ERROR] {PER_RASIO_PATH} tidak ditemukan.")
        return None
    with open(PER_RASIO_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("bobot_per_rasio", {})


def muat_hasil_lama():
    if not RESULTS_PATH.exists():
        print(f"[ERROR] {RESULTS_PATH} tidak ditemukan -- jalankan "
              f"45_multiseed_per_rasio.py terlebih dahulu (baseline/20%/60%).")
        return None
    with open(RESULTS_PATH, encoding="utf-8") as f:
        return json.load(f)


def correct_and_total(metrics):
    cm = metrics["confusion_matrix"]
    return int(cm.trace()), int(cm.sum())


def summarize(per_seed_dict):
    accs = [v["accuracy"] for v in per_seed_dict.values()]
    return {
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "n_seeds": len(accs),
    }


def run_pruned_seed(ratio_pct, masks, seed, device, model_baseline):
    ckpt_name = f"multicriteria_per_rasio_{ratio_pct}pct_{EPOCHS}ep_seed{seed}.pth"
    ckpt_path = CFG.CHECKPOINT_DIR / ckpt_name

    set_seed(seed)
    dataloaders, dataset_sizes = get_dataloaders(SPLIT_DIR)

    if ckpt_path.exists():
        print(f"[SKIP] Multi-Kriteria-PerRasio {ratio_pct}% seed={seed}: checkpoint sudah ada ({ckpt_name})")
        model, _ = load_checkpoint(ckpt_path, device)
    else:
        print(f"\n{'=' * 60}")
        print(f"MULTI-KRITERIA-PERRASIO {ratio_pct}% seed={seed} (mask identik di ketiga seed)")
        print(f"{'=' * 60}")
        pruned_model = apply_pruning(model_baseline, masks)
        model, _ = train_model(
            model=pruned_model,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            num_epochs=EPOCHS,
            lr=CFG.FINETUNE_LR,
            device=device,
            checkpoint_path=ckpt_path,
            phase_name=f"Multi-Kriteria-PerRasio {ratio_pct}% seed={seed}",
        )

    metrics_val, _, _ = evaluate_model(model, dataloaders["val"], device)
    metrics_test, _, _ = evaluate_model(model, dataloaders["test"], device)
    return metrics_val, metrics_test, ckpt_name


def main():
    device = torch.device("cpu")

    print("=" * 70)
    print("VALIDASI MULTI-SEED -- LIMA RASIO TERSISA (10/30/40/50/70%)")
    print("=" * 70)
    print(f"  Seed      : {SEEDS} (SEMUA dilatih fresh, termasuk 42 -- atas instruksi eksplisit)")
    print(f"  Rasio     : {RATIOS_PCT}%")
    print(f"  Epoch     : {EPOCHS}")
    print(f"  Device    : {device}")

    bobot_per_rasio = muat_bobot_per_rasio()
    if bobot_per_rasio is None:
        return
    for pct in RATIOS_PCT:
        if f"{pct}%" not in bobot_per_rasio:
            print(f"[ERROR] Bobot untuk rasio {pct}% tidak tersedia.")
            return

    all_results = muat_hasil_lama()
    if all_results is None:
        return

    BASELINE_CKPT = CFG.CHECKPOINT_DIR / "baseline_9class.pth"
    if not BASELINE_CKPT.exists():
        print("[ERROR] Checkpoint baseline (sumber pruning) tidak ditemukan!")
        return

    print("\n[INFO] Memuat baseline & menghitung skor L1, GM-Ekspansi, Entropi sekali "
          "(dipakai untuk kelima rasio di ketiga seed)...")
    model_baseline, _ = load_checkpoint(BASELINE_CKPT, device)
    model_baseline.eval()

    dataloaders_scoring, _ = get_dataloaders(SPLIT_DIR)
    l1_scores = compute_l1_scores(model_baseline)
    gm_scores = hitung_skor_gm_ekspansi_semua_layer(model_baseline)
    entropy_scores = compute_entropy_scores(
        model_baseline, dataloaders_scoring["train"], device, samples_per_class=20, seed=CFG.SEED
    )

    masks_per_ratio = {}
    for pct in RATIOS_PCT:
        w = bobot_per_rasio[f"{pct}%"]
        print(f"\n[INFO] Rasio {pct}%: bobot w_L1={w['w_l1']:.10f}, w_GM={w['w_gm']:.10f}, "
              f"w_Ent={w['w_entropi']:.10f} (dari {PER_RASIO_PATH.name}, tidak dihitung ulang)")
        importance_scores = compute_importance_scores(
            l1_scores, gm_scores, entropy_scores, w1=w["w_l1"], w2=w["w_gm"], w3=w["w_entropi"]
        )
        masks_per_ratio[pct] = get_pruning_mask(importance_scores, pct / 100)

    run_plan = [(pct, seed) for pct in RATIOS_PCT for seed in SEEDS]
    total_runs = len(run_plan)
    start_time = time.time()

    for i, (pct, seed) in enumerate(run_plan, 1):
        elapsed = time.time() - start_time
        print(f"\n>>> Run {i}/{total_runs} | rasio={pct}% seed={seed} "
              f"| Waktu berjalan: {elapsed/60:.1f} menit")

        metrics_val, metrics_test, ckpt_name = run_pruned_seed(
            pct, masks_per_ratio[pct], seed, device, model_baseline
        )
        n_correct, n_total = correct_and_total(metrics_test)

        config_name = f"multicriteria_per_rasio_{pct}pct"
        config_entry = all_results["configs"].setdefault(config_name, {"per_seed": {}})
        config_entry["per_seed"][str(seed)] = {
            "accuracy": metrics_test["accuracy"],
            "f1_score": metrics_test["f1_score"],
            "val_accuracy": metrics_val["accuracy"],
            "correct": n_correct,
            "total": n_total,
            "checkpoint": ckpt_name,
            "catatan": "seed 42 di run ini DILATIH ULANG FRESH, BUKAN dipakai ulang dari "
                       "44_multicriteria_per_rasio.py" if seed == 42 else None,
        }
        config_entry.update(summarize(config_entry["per_seed"]))

        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Hasil sementara disimpan: {RESULTS_PATH}")

    all_results["status"] = "SELESAI (baseline + 7 rasio lengkap 3 seed)"
    all_results["total_time_seconds_lima_rasio_tersisa"] = time.time() - start_time
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print("RINGKASAN -- LIMA RASIO TERSISA")
    print(f"{'=' * 70}")
    for pct in RATIOS_PCT:
        entry = all_results["configs"][f"multicriteria_per_rasio_{pct}pct"]
        print(f"\nrasio {pct}%:")
        for seed_str, r in entry["per_seed"].items():
            print(f"  seed={seed_str:<6} akurasi={r['accuracy']*100:.2f}%")
        print(f"  Rata-rata = {entry['mean_accuracy']*100:.2f}%  "
              f"Std = {entry['std_accuracy']*100:.2f}%  (n={entry['n_seeds']})")
    print(f"{'=' * 70}")
    print(f"\n[INFO] Hasil disimpan: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
