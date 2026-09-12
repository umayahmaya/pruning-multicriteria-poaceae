"""
45_multiseed_per_rasio.py
Validasi multi-seed untuk formula I(c) AKTIF -- L1 + GM-Ekspansi +
Entropi, bobot w_k(r) per rasio (lihat CLAUDE.md Bagian 2 dan
44_multicriteria_per_rasio.py) -- rasio 20% (klaim utama tesis) dan 60%
(titik kompresi agresif), seed 123 dan 2024. Seed 42 TIDAK dilatih ulang,
dipakai ulang dari outputs/multicriteria_per_rasio.json (hasil
44_multicriteria_per_rasio.py yang sudah ada).

MENGAPA HANYA 20% DAN 60%, BUKAN KETUJUH RASIO: waktu fine-tuning CPU
~54 menit/rasio (basis historis dipakai 44_multicriteria_per_rasio.py,
rerata 34/37/38_multikriteria_gm.py) -- ketujuh rasio x 2 seed baru akan
makan >12 jam. Rasio 20% dipilih karena klaim akurasi utama tesis (20%
> baseline pada seed 42: 97,14% vs 96,00%, CLAUDE.md Bagian 4 Aturan 5
mensyaratkan validasi multi-seed sebelum klaim itu sah dikutip). Rasio
60% dipilih sebagai titik kompresi agresif kedua, MENIRU CAKUPAN
15_multiseed_validation.py (yang memvalidasi 20% dan 60% untuk formula
LAMA) supaya kedua formula (lama dan aktif) punya titik pembanding yang
sama pada tingkat kompresi yang sama.

CATATAN METODOLOGIS PENTING (premis sama seperti 15_multiseed_validation.py):
Mask pemangkasan -- channel intermediate mana yang dibuang -- dihitung
SEKALI dari skor I(c) model baseline yang sudah ada
(checkpoints/baseline_9class.pth), memakai bobot w_k(r) per rasio yang
SUDAH ADA di outputs/multicriteria_per_rasio.json (bobot_per_rasio,
TIDAK dihitung ulang di sini), dan IDENTIK di ketiga seed (42 dari
skrip 44, 123 dan 2024 dari skrip ini). Variasi akurasi antar seed
murni variasi fine-tuning (urutan batch, augmentasi acak, dropout),
BUKAN variasi pemilihan channel.

Skor GM-Ekspansi memakai hitung_skor_gm_ekspansi_semua_layer, diimpor
dinamis dari 43_ablation_gm_expansion.py (TIDAK diubah), persis seperti
44_multicriteria_per_rasio.py. L1 dan Entropi memakai
compute_l1_scores()/compute_entropy_scores() dari src/pruning.py apa
adanya (kalibrasi entropi: train, 20 citra/kelas, seed=CFG.SEED --
skor mentah TIDAK bergantung seed run, dipakai ulang untuk kedua
rasio dan ketiga seed).

Baseline (tanpa pruning) TIDAK dilatih ulang di sini -- baseline tidak
bergantung pada formula I(c) sama sekali (tidak ada pruning), jadi
hasil multi-seed baseline dari outputs/multiseed_results.json (config
"baseline", seed 42/123/2024, dari 15_multiseed_validation.py) dipakai
ulang apa adanya untuk perbandingan di ringkasan akhir.

Checkpoint memakai akhiran _seed{seed} mengikuti konvensi
15_multiseed_validation.py, tidak menimpa checkpoint manapun. Ada
logika skip-jika-checkpoint-ada untuk resume. Progres disimpan ke
outputs/multiseed_per_rasio_results.json setiap satu run selesai.

Urutan eksekusi (SENGAJA): 20% x 2 seed dulu, lalu 60% x 2 seed --
supaya kalau proses terhenti, angka yang paling dibutuhkan (rasio 20%)
sudah lebih dulu lengkap.

Jalankan:
    venv/Scripts/python.exe scripts/45_multiseed_per_rasio.py
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
# (nama file diawali angka -> tidak bisa `import` biasa; TIDAK mengubah
# skrip 43, hanya memuat definisinya sebagai modul terpisah -- sama
# seperti dilakukan 44_multicriteria_per_rasio.py)
_JALUR_SKRIP_43 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "43_ablation_gm_expansion.py")
_spec_43 = importlib.util.spec_from_file_location("_skrip_43_ablation_gm_expansion", _JALUR_SKRIP_43)
_mod_43 = importlib.util.module_from_spec(_spec_43)
_spec_43.loader.exec_module(_mod_43)
hitung_skor_gm_ekspansi_semua_layer = _mod_43.hitung_skor_gm_ekspansi_semua_layer

SEEDS_BARU = [123, 2024]
RATIOS_PCT = [20, 60]
EPOCHS = CFG.BASELINE_EPOCHS  # 30
SPLIT_DIR = CFG.DATASET_DIR.parent / "dataset_split"

PER_RASIO_PATH = CFG.OUTPUT_DIR / "multicriteria_per_rasio.json"
BASELINE_MULTISEED_PATH = CFG.OUTPUT_DIR / "multiseed_results.json"
RESULTS_PATH = CFG.OUTPUT_DIR / "multiseed_per_rasio_results.json"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def muat_bobot_dan_seed42():
    """Baca bobot_per_rasio dan hasil seed=42 (test/val) yang SUDAH ADA
    dari outputs/multicriteria_per_rasio.json -- tidak dihitung ulang."""
    if not PER_RASIO_PATH.exists():
        print(f"[ERROR] {PER_RASIO_PATH} tidak ditemukan. Jalankan 44_multicriteria_per_rasio.py "
              f"--jalankan-eksperimen terlebih dahulu.")
        return None, None
    with open(PER_RASIO_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("bobot_per_rasio", {}), data.get("results", {})


def muat_baseline_multiseed():
    """Baca ringkasan multi-seed baseline yang SUDAH ADA (tidak bergantung
    formula I(c), jadi dipakai ulang apa adanya)."""
    if not BASELINE_MULTISEED_PATH.exists():
        return None
    with open(BASELINE_MULTISEED_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("configs", {}).get("baseline")


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
    print("VALIDASI MULTI-SEED -- FORMULA AKTIF (GM-EKSPANSI, BOBOT PER-RASIO)")
    print("=" * 70)
    print(f"  Seed baru : {SEEDS_BARU} (seed 42 dipakai ulang dari 44_multicriteria_per_rasio.py)")
    print(f"  Rasio     : {RATIOS_PCT}%")
    print(f"  Epoch     : {EPOCHS}")
    print(f"  Device    : {device}")

    bobot_per_rasio, hasil_seed42 = muat_bobot_dan_seed42()
    if bobot_per_rasio is None:
        return

    for pct in RATIOS_PCT:
        if f"{pct}%" not in bobot_per_rasio:
            print(f"[ERROR] Bobot untuk rasio {pct}% tidak tersedia di {PER_RASIO_PATH}.")
            return

    BASELINE_CKPT = CFG.CHECKPOINT_DIR / "baseline_9class.pth"
    if not BASELINE_CKPT.exists():
        print("[ERROR] Checkpoint baseline (sumber pruning) tidak ditemukan!")
        return

    print("\n[INFO] Memuat baseline & menghitung skor L1, GM-Ekspansi, Entropi sekali "
          "(dipakai untuk 20% dan 60% di ketiga seed)...")
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
        print(f"\n[INFO] Rasio {pct}%: bobot w_L1={w['w_l1']:.4f}, w_GM={w['w_gm']:.4f}, "
              f"w_Ent={w['w_entropi']:.4f} (dari {PER_RASIO_PATH.name}, tidak dihitung ulang)")
        importance_scores = compute_importance_scores(
            l1_scores, gm_scores, entropy_scores, w1=w["w_l1"], w2=w["w_gm"], w3=w["w_entropi"]
        )
        masks_per_ratio[pct] = get_pruning_mask(importance_scores, pct / 100)

    all_results = {
        "seeds_baru": SEEDS_BARU,
        "seed_42_sumber": str(PER_RASIO_PATH),
        "epochs": EPOCHS,
        "formula": "I(c) = w_L1(r)*S_L1 + w_GM(r)*S_GM-Ekspansi + w_Ent(r)*S_Entropi (bobot per rasio)",
        "bobot_per_rasio_dipakai": {f"{pct}%": bobot_per_rasio[f"{pct}%"] for pct in RATIOS_PCT},
        "pruning_note": (
            "Mask pemangkasan dihitung SEKALI dari baseline yang sudah ada dan IDENTIK "
            "di ketiga seed (42 dari 44_multicriteria_per_rasio.py, 123 dan 2024 dari "
            "skrip ini) -- channel yang dibuang sama persis. Variasi antar seed murni "
            "variasi fine-tuning, BUKAN variasi pemilihan channel."
        ),
        "configs": {},
        "status": "BERJALAN",
    }

    # Seed 42: masukkan hasil yang SUDAH ADA (tidak dilatih ulang)
    for pct in RATIOS_PCT:
        entry_42 = hasil_seed42.get(f"{pct}%")
        config_name = f"multicriteria_per_rasio_{pct}pct"
        config_entry = all_results["configs"].setdefault(config_name, {"per_seed": {}})
        if entry_42:
            config_entry["per_seed"]["42"] = {
                "accuracy": entry_42["test"]["accuracy"],
                "f1_score": entry_42["test"]["f1_score"],
                "val_accuracy": entry_42["validasi"]["accuracy"],
                "checkpoint": entry_42["checkpoint"],
                "sumber": "dipakai ulang dari 44_multicriteria_per_rasio.py (tidak dilatih ulang)",
            }
        else:
            print(f"[PERINGATAN] Hasil seed=42 untuk rasio {pct}% tidak ditemukan di {PER_RASIO_PATH.name}.")

    run_plan = [(pct, seed) for pct in RATIOS_PCT for seed in SEEDS_BARU]
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
        }
        config_entry.update(summarize(config_entry["per_seed"]))

        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Hasil sementara disimpan: {RESULTS_PATH}")

    # Pastikan ringkasan (mean/std) tercatat juga untuk config yang semua
    # seed-nya kebetulan sudah ada sejak awal (skip semua run baru).
    for config_name, entry in all_results["configs"].items():
        entry.update(summarize(entry["per_seed"]))

    all_results["status"] = "SELESAI"
    all_results["total_time_seconds"] = time.time() - start_time

    baseline_multiseed = muat_baseline_multiseed()
    if baseline_multiseed:
        all_results["baseline_dipakai_ulang"] = {
            "sumber": str(BASELINE_MULTISEED_PATH),
            "catatan": "Baseline tidak bergantung formula I(c) (tanpa pruning), dipakai ulang apa adanya.",
            "per_seed": baseline_multiseed["per_seed"],
            "mean_accuracy": baseline_multiseed["mean_accuracy"],
            "std_accuracy": baseline_multiseed["std_accuracy"],
            "n_seeds": baseline_multiseed["n_seeds"],
        }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Ringkasan akhir
    print(f"\n{'=' * 70}")
    print("RINGKASAN VALIDASI MULTI-SEED -- FORMULA AKTIF (PER-RASIO)")
    print(f"{'=' * 70}")
    if baseline_multiseed:
        print("\nbaseline (dipakai ulang dari 15_multiseed_validation.py):")
        for seed_str, r in baseline_multiseed["per_seed"].items():
            print(f"  seed={seed_str:<6} akurasi={r['accuracy']*100:.2f}%  benar={r['correct']}/{r['total']}")
        print(f"  Rata-rata = {baseline_multiseed['mean_accuracy']*100:.2f}%  "
              f"Std = {baseline_multiseed['std_accuracy']*100:.2f}%  (n={baseline_multiseed['n_seeds']})")
    for config_name, entry in all_results["configs"].items():
        print(f"\n{config_name}:")
        for seed_str, r in entry["per_seed"].items():
            print(f"  seed={seed_str:<6} akurasi={r['accuracy']*100:.2f}%")
        print(f"  Rata-rata = {entry['mean_accuracy']*100:.2f}%  "
              f"Std = {entry['std_accuracy']*100:.2f}%  (n={entry['n_seeds']})")
    print(f"{'=' * 70}")
    print(f"\n[INFO] Hasil disimpan: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
