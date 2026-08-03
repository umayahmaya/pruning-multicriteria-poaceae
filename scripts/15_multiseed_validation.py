"""
15_multiseed_validation.py
Validasi multi-seed (42, 123, 2024) untuk tiga konfigurasi, masing-masing
30 epoch:
    1. Multi-kriteria rasio 20% (diprioritaskan -- klaim utama tesis)
    2. Multi-kriteria rasio 60%
    3. Baseline (tanpa pruning)

CATATAN METODOLOGIS PENTING (rujukan bab metodologi):
Untuk konfigurasi multi-kriteria (20% dan 60%), mask pemangkasan -- yaitu
channel intermediate mana yang dibuang -- dihitung SEKALI dari skor I(c)
model baseline yang sudah ada (checkpoints/baseline_9class.pth, bobot dari
outputs/ablation_val_results.json) dan IDENTIK di ketiga seed. Ketiga seed
memangkas channel yang SAMA PERSIS; yang berbeda hanya proses fine-tuning
setelahnya (urutan batch, augmentasi data acak, dropout). Dengan kata lain,
variasi akurasi antar seed yang dilaporkan di sini adalah variasi FINE-TUNING
SAJA, BUKAN variasi pemilihan channel/pruning. Konfigurasi baseline berbeda:
model dilatih dari awal (pretrained ImageNet + head baru) secara independen
di tiap seed, jadi variasinya mencakup inisialisasi bobot classifier head
sekaligus dinamika training.

Split dataset TIDAK PERNAH dibuat ulang oleh skrip ini -- hanya membaca
folder dataset_split/{train,val,test} yang sudah ada lewat get_dataloaders(),
persis seperti skrip lain. Seed tidak memengaruhi split sama sekali.

Checkpoint memakai akhiran _seed{seed} sehingga tidak menimpa checkpoint
manapun yang sudah ada. Ada logika skip: kalau checkpoint untuk suatu
(konfigurasi, seed) sudah ada, training dilewati dan langsung dievaluasi
ulang dari checkpoint tsb.

Urutan eksekusi (SENGAJA, bukan urutan 1-2-3 di atas): 20% x 3 seed dulu,
lalu 60% x 3 seed, baru baseline x 3 seed -- supaya kalau proses terhenti
di tengah jalan, angka yang paling dibutuhkan (rasio 20%) sudah lebih dulu
lengkap.

Progres disimpan ke outputs/multiseed_results.json setiap SATU run selesai.

Jalankan:
    venv/Scripts/python.exe scripts/15_multiseed_validation.py
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
from src.model import create_mobilenetv2, train_model, evaluate_model, load_checkpoint
from src.pruning import (
    compute_l1_scores, compute_bn_scores, compute_entropy_scores,
    compute_importance_scores, get_pruning_mask, apply_pruning
)

SEEDS = [42, 123, 2024]
EPOCHS = CFG.BASELINE_EPOCHS  # 30
VAL_WEIGHTS_PATH = CFG.OUTPUT_DIR / "ablation_val_results.json"
RESULTS_PATH = CFG.OUTPUT_DIR / "multiseed_results.json"
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


def run_baseline(seed, device):
    """Latih (atau muat) baseline untuk satu seed, evaluasi pada test."""
    ckpt_name = f"baseline_{EPOCHS}ep_seed{seed}.pth"
    ckpt_path = CFG.CHECKPOINT_DIR / ckpt_name

    set_seed(seed)
    dataloaders, dataset_sizes = get_dataloaders(SPLIT_DIR)

    if ckpt_path.exists():
        print(f"[SKIP] Baseline seed={seed}: checkpoint sudah ada ({ckpt_name})")
        model, _ = load_checkpoint(ckpt_path, device)
    else:
        print(f"\n{'=' * 60}")
        print(f"BASELINE seed={seed}")
        print(f"{'=' * 60}")
        model = create_mobilenetv2(num_classes=CFG.NUM_CLASSES, pretrained=True)
        model, _ = train_model(
            model=model,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            num_epochs=EPOCHS,
            lr=CFG.BASELINE_LR,
            device=device,
            checkpoint_path=ckpt_path,
            phase_name=f"Baseline seed={seed}",
        )

    metrics, _, _ = evaluate_model(model, dataloaders["test"], device)
    return metrics, ckpt_name


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
        print(f"MULTI-KRITERIA {ratio_pct}% seed={seed} (mask identik di ketiga seed)")
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
    print("VALIDASI MULTI-SEED")
    print("=" * 60)
    print(f"  Seed    : {SEEDS}")
    print(f"  Epoch   : {EPOCHS}")
    print(f"  Device  : {device}")
    print(f"  Urutan  : 20% (3 seed) -> 60% (3 seed) -> baseline (3 seed)")

    weights = load_val_weights()
    if weights is None:
        return
    w1, w2, w3 = weights
    print(f"  Bobot   : w1={w1:.6f}, w2={w2:.6f}, w3={w3:.6f}")

    BASELINE_CKPT = CFG.CHECKPOINT_DIR / "baseline_9class.pth"
    if not BASELINE_CKPT.exists():
        print("[ERROR] Checkpoint baseline (sumber pruning) tidak ditemukan!")
        print("Jalankan 02_train_baseline.py terlebih dahulu.")
        return

    print("\n[INFO] Memuat baseline & menghitung skor I(c) sekali "
          "(dipakai untuk 20% dan 60% di ketiga seed)...")
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

    # Mask dihitung sekali per rasio -- objek yang SAMA dipakai untuk ketiga
    # seed, supaya "identik di ketiga seed" terjamin secara konstruksi.
    masks_20 = get_pruning_mask(importance_scores, 0.2)
    masks_60 = get_pruning_mask(importance_scores, 0.6)

    all_results = {
        "seeds": SEEDS,
        "epochs": EPOCHS,
        "weights_used": {"w1": w1, "w2": w2, "w3": w3},
        "pruning_note": (
            "Mask pemangkasan untuk multicriteria_20pct dan multicriteria_60pct "
            "dihitung SEKALI dari baseline yang sudah ada dan IDENTIK di ketiga "
            "seed -- channel yang dibuang sama persis. Variasi antar seed pada "
            "kedua konfigurasi ini murni variasi fine-tuning (urutan batch, "
            "augmentasi acak, dropout), BUKAN variasi pemilihan channel. "
            "Konfigurasi baseline melatih model baru dari awal secara independen "
            "tiap seed."
        ),
        "configs": {},
        "status": "BERJALAN",
    }

    run_plan = (
        [("multicriteria_20pct", 20, masks_20, seed) for seed in SEEDS] +
        [("multicriteria_60pct", 60, masks_60, seed) for seed in SEEDS] +
        [("baseline", None, None, seed) for seed in SEEDS]
    )

    total_runs = len(run_plan)
    start_time = time.time()

    for i, (config_name, ratio_pct, masks, seed) in enumerate(run_plan, 1):
        elapsed = time.time() - start_time
        print(f"\n>>> Run {i}/{total_runs} | {config_name} seed={seed} "
              f"| Waktu berjalan: {elapsed/60:.1f} menit")

        if config_name == "baseline":
            metrics, ckpt_name = run_baseline(seed, device)
        else:
            metrics, ckpt_name = run_pruned(
                ratio_pct, masks, seed, device, model_baseline_for_pruning
            )

        n_correct, n_total = correct_and_total(metrics)

        config_entry = all_results["configs"].setdefault(config_name, {"per_seed": {}})
        config_entry["per_seed"][str(seed)] = {
            "accuracy": metrics["accuracy"],
            "f1_score": metrics["f1_score"],
            "correct": n_correct,
            "total": n_total,
            "checkpoint": ckpt_name,
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
    print("RINGKASAN VALIDASI MULTI-SEED")
    print(f"{'=' * 70}")
    for config_name, entry in all_results["configs"].items():
        print(f"\n{config_name}:")
        for seed_str, r in entry["per_seed"].items():
            print(f"  seed={seed_str:<6} akurasi={r['accuracy']*100:.2f}%  "
                  f"benar={r['correct']}/{r['total']}")
        print(f"  Rata-rata = {entry['mean_accuracy']*100:.2f}%  "
              f"Std = {entry['std_accuracy']*100:.2f}%  (n={entry['n_seeds']})")
    print(f"{'=' * 70}")
    print(f"\n[INFO] Hasil disimpan: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
