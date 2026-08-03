"""
16_multiseed_remaining_ratios.py
Melengkapi validasi multi-seed (lihat 15_multiseed_validation.py) untuk
rasio multi-kriteria yang belum diuji multi-seed: 10%, 30%, 40%, 50%, 70%.

Hanya seed 123 dan 2024 yang dijalankan di sini -- seed 42 untuk kelima
rasio ini SUDAH ADA dari 12_multicriteria_valweights.py
(checkpoints/multicriteria_{rasio}pct_30ep_valweights.pth, tanpa akhiran
seed) dan TIDAK dijalankan ulang di sini.

Mengikuti pola yang sama persis dengan 15_multiseed_validation.py:
- Mask pemangkasan dihitung SEKALI dari baseline yang sama
  (checkpoints/baseline_9class.pth) dan IDENTIK di semua seed -- termasuk
  seed 42 yang sudah ada, karena baseline dan bobotnya sama persis. Channel
  yang dibuang sama untuk rasio yang sama di ketiga seed. Variasi antar
  seed yang diukur murni variasi fine-tuning (urutan batch, augmentasi
  acak, dropout), BUKAN variasi pemilihan channel.
- Bobot dari outputs/ablation_val_results.json (optimal_weights_val).
- Kalibrasi entropi tetap seed=CFG.SEED (42), tidak ikut berubah dengan
  seed training yang diuji.
- 30 epoch.
- Checkpoint akhiran _seed{seed}, ada logika skip kalau checkpoint sudah ada.

Ringkasan akhir MENGGABUNGKAN hasil seed 123 & 2024 (dari run ini) dengan
hasil seed 42 yang dibaca dari outputs/multicriteria_valweights_results.json
(TIDAK dievaluasi ulang -- jumlah benar dihitung dari round(akurasi x 175)
karena file itu tidak menyimpan confusion matrix), sehingga tiap rasio
akhirnya punya tiga seed lengkap.

Urutan eksekusi: rasio 30% -> 40% -> 50% -> 10% -> 70% (tiap rasio: seed
123 dulu, baru 2024).

Progres disimpan ke outputs/multiseed_remaining_results.json setiap SATU
run selesai.

Jalankan:
    venv/Scripts/python.exe scripts/16_multiseed_remaining_ratios.py
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
from src.model import train_model, evaluate_model, load_checkpoint
from src.pruning import (
    compute_l1_scores, compute_bn_scores, compute_entropy_scores,
    compute_importance_scores, get_pruning_mask, apply_pruning
)

SEEDS = [123, 2024]
EPOCHS = CFG.BASELINE_EPOCHS  # 30
RATIO_ORDER = [30, 40, 50, 10, 70]  # persen, urutan eksekusi sesuai prioritas

VAL_WEIGHTS_PATH = CFG.OUTPUT_DIR / "ablation_val_results.json"
SEED42_RESULTS_PATH = CFG.OUTPUT_DIR / "multicriteria_valweights_results.json"
RESULTS_PATH = CFG.OUTPUT_DIR / "multiseed_remaining_results.json"
SPLIT_DIR = CFG.DATASET_DIR.parent / "dataset_split"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_val_weights():
    """Muat bobot multi-kriteria hasil recompute dari akurasi val."""
    if not VAL_WEIGHTS_PATH.exists():
        print(f"[ERROR] {VAL_WEIGHTS_PATH} tidak ditemukan.")
        print("Jalankan 07_recompute_weights_val.py terlebih dahulu.")
        return None

    with open(VAL_WEIGHTS_PATH) as f:
        data = json.load(f)
    weights = data.get("optimal_weights_val") or {}
    if not all(k in weights for k in ("w1_l1", "w2_bn", "w3_entropy")):
        print(f"[ERROR] Kunci 'optimal_weights_val' tidak lengkap di {VAL_WEIGHTS_PATH}")
        return None

    return weights["w1_l1"], weights["w2_bn"], weights["w3_entropy"]


def load_seed42_results():
    """Muat hasil seed=42 (dari 12_multicriteria_valweights.py) untuk digabung
    ke ringkasan akhir -- TIDAK dievaluasi ulang."""
    if not SEED42_RESULTS_PATH.exists():
        print(f"[PERINGATAN] {SEED42_RESULTS_PATH} tidak ditemukan, "
              f"ringkasan akhir tidak akan menyertakan seed 42.")
        return {}
    with open(SEED42_RESULTS_PATH) as f:
        data = json.load(f)
    return data.get("results", {})


def correct_and_total(metrics):
    """Jumlah prediksi benar dan total sampel dari confusion matrix (bilangan bulat pasti)."""
    cm = metrics["confusion_matrix"]
    return int(cm.trace()), int(cm.sum())


def summarize_config(per_seed):
    accs = [v["accuracy"] for v in per_seed.values()]
    return {
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "n_seeds": len(accs),
    }


def run_pruned(ratio_pct, masks, seed, device, model_baseline_for_pruning):
    """Pangkas (mask tetap, tidak berubah per seed), fine-tune untuk satu seed, evaluasi."""
    ckpt_name = f"multicriteria_{ratio_pct}pct_{EPOCHS}ep_valweights_seed{seed}.pth"
    ckpt_path = CFG.CHECKPOINT_DIR / ckpt_name

    set_seed(seed)
    dataloaders, dataset_sizes = get_dataloaders(SPLIT_DIR)

    if ckpt_path.exists():
        print(f"[SKIP] Multi-kriteria {ratio_pct}% seed={seed}: checkpoint sudah ada ({ckpt_name})")
        model, _ = load_checkpoint(ckpt_path, device)
    else:
        print(f"\n{'=' * 60}")
        print(f"MULTI-KRITERIA {ratio_pct}% seed={seed} (mask identik di semua seed)")
        print(f"{'=' * 60}")
        pruned_model = apply_pruning(model_baseline_for_pruning, masks)
        model, _ = train_model(
            model=pruned_model,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            num_epochs=EPOCHS,
            lr=CFG.FINETUNE_LR,
            device=device,
            checkpoint_path=ckpt_path,
            phase_name=f"Multi-kriteria {ratio_pct}% seed={seed}",
        )

    metrics, _, _ = evaluate_model(model, dataloaders["test"], device)
    return metrics, ckpt_name


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("VALIDASI MULTI-SEED -- RASIO SISA (10/30/40/50/70%)")
    print("=" * 60)
    print(f"  Seed baru   : {SEEDS} (seed 42 sudah ada, tidak diulang)")
    print(f"  Epoch       : {EPOCHS}")
    print(f"  Device      : {device}")
    print(f"  Urutan rasio: {RATIO_ORDER}")

    weights = load_val_weights()
    if weights is None:
        return
    w1, w2, w3 = weights
    print(f"  Bobot       : w1={w1:.6f}, w2={w2:.6f}, w3={w3:.6f}")

    BASELINE_CKPT = CFG.CHECKPOINT_DIR / "baseline_9class.pth"
    if not BASELINE_CKPT.exists():
        print("[ERROR] Checkpoint baseline (sumber pruning) tidak ditemukan!")
        print("Jalankan 02_train_baseline.py terlebih dahulu.")
        return

    print("\n[INFO] Memuat baseline & menghitung skor I(c) sekali "
          "(dipakai untuk semua rasio x semua seed baru)...")
    model_baseline_for_pruning, _ = load_checkpoint(BASELINE_CKPT, device)

    dataloaders_scoring, _ = get_dataloaders(SPLIT_DIR)
    l1_scores = compute_l1_scores(model_baseline_for_pruning)
    bn_scores = compute_bn_scores(model_baseline_for_pruning)
    entropy_scores = compute_entropy_scores(
        model_baseline_for_pruning, dataloaders_scoring["train"], device
    )
    importance_scores = compute_importance_scores(
        l1_scores, bn_scores, entropy_scores, w1=w1, w2=w2, w3=w3
    )

    # Mask dihitung sekali per rasio -- objek yang sama dipakai untuk seed
    # 123 dan 2024, dan (karena baseline & bobot identik) otomatis sama
    # dengan mask yang dipakai seed 42 di 12_multicriteria_valweights.py.
    masks_by_ratio = {
        ratio_pct: get_pruning_mask(importance_scores, ratio_pct / 100)
        for ratio_pct in RATIO_ORDER
    }

    seed42_results = load_seed42_results()

    all_results = {
        "seeds_new": SEEDS,
        "epochs": EPOCHS,
        "weights_used": {"w1": w1, "w2": w2, "w3": w3},
        "pruning_note": (
            "Mask pemangkasan dihitung sekali dari baseline yang sama dan "
            "IDENTIK di semua seed (termasuk seed 42 dari "
            "12_multicriteria_valweights.py) untuk rasio yang sama. Variasi "
            "antar seed murni variasi fine-tuning, bukan pemilihan channel. "
            "Seed 42 TIDAK dijalankan ulang di sini -- hasilnya dibaca dari "
            "outputs/multicriteria_valweights_results.json."
        ),
        "configs": {},
        "status": "BERJALAN",
    }

    run_plan = [
        (ratio_pct, masks_by_ratio[ratio_pct], seed)
        for ratio_pct in RATIO_ORDER
        for seed in SEEDS
    ]

    total_runs = len(run_plan)
    start_time = time.time()

    for i, (ratio_pct, masks, seed) in enumerate(run_plan, 1):
        elapsed = time.time() - start_time
        print(f"\n>>> Run {i}/{total_runs} | {ratio_pct}% seed={seed} "
              f"| Waktu berjalan: {elapsed/60:.1f} menit")

        metrics, ckpt_name = run_pruned(
            ratio_pct, masks, seed, device, model_baseline_for_pruning
        )
        n_correct, n_total = correct_and_total(metrics)

        config_name = f"multicriteria_{ratio_pct}pct"
        config_entry = all_results["configs"].setdefault(config_name, {"per_seed": {}})
        config_entry["per_seed"][str(seed)] = {
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
            "correct": n_correct,
            "total": n_total,
            "checkpoint": ckpt_name,
        }

        # Gabungkan dengan hasil seed 42 yang sudah ada (tidak dievaluasi ulang)
        ratio_key = f"{ratio_pct}%"
        seed42_entry = seed42_results.get(ratio_key)
        if seed42_entry:
            acc42 = seed42_entry["accuracy"]
            config_entry["per_seed"]["42"] = {
                "accuracy": acc42,
                "f1_score": seed42_entry["f1_score"],
                "correct": round(acc42 * 175),
                "total": 175,
                "checkpoint": f"multicriteria_{ratio_pct}pct_{EPOCHS}ep_valweights.pth",
                "source": "12_multicriteria_valweights.py (tidak dijalankan ulang)",
            }

        config_entry.update(summarize_config(config_entry["per_seed"]))

        with open(RESULTS_PATH, "w") as f:
            json.dump(all_results, f, indent=2)

    all_results["status"] = "SELESAI"
    all_results["total_time_seconds"] = time.time() - start_time
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)

    # Ringkasan akhir
    print(f"\n{'=' * 70}")
    print("RINGKASAN VALIDASI MULTI-SEED -- RASIO SISA (tiga seed lengkap)")
    print(f"{'=' * 70}")
    for ratio_pct in RATIO_ORDER:
        config_name = f"multicriteria_{ratio_pct}pct"
        entry = all_results["configs"][config_name]
        print(f"\n{config_name}:")
        for seed_str in ["42", "123", "2024"]:
            if seed_str in entry["per_seed"]:
                r = entry["per_seed"][seed_str]
                print(f"  seed={seed_str:<6} akurasi={r['accuracy']*100:.2f}%  "
                      f"benar={r['correct']}/{r['total']}")
        print(f"  Rata-rata = {entry['mean_accuracy']*100:.2f}%  "
              f"Std = {entry['std_accuracy']*100:.2f}%  (n={entry['n_seeds']})")
    print(f"{'=' * 70}")
    print(f"\n[INFO] Hasil disimpan: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
