"""
48_multiseed_single_criteria.py
Validasi multi-seed (123, 2024; seed 42 dipakai ulang dari checkpoint
ablation kriteria tunggal yang sudah ada) untuk KETIGA kriteria tunggal
yang membentuk formula I(c) aktif -- L1, GM-Ekspansi, Entropi -- di
SELURUH 7 rasio pemangkasan. 21 pasang (kriteria x rasio) x 2 seed baru
= 42 run.

LATAR BELAKANG: grafik kurva_kompresi_akurasi_per_rasio.png menunjukkan
pola non-monoton pada seed 42 tunggal -- L1 turun-naik drastis di
10-20-30%, dan Entropi melonjak di 50-60%. Audit checkpoint (hash SHA256
unik, jumlah channel per block turun monoton sesuai rasio) sudah
membuktikan ini BUKAN bug pemangkasan/mask -- tapi tanpa data multi-seed,
tidak bisa dipastikan pola itu titik yang reproducible atau cuma satu
tarikan beruntung/sial (CLAUDE.md Bagian 9: set_seed() dipanggil sekali
di awal skrip ablation asli, RNG antar-rasio tidak independen).

CAKUPAN PENUH (bukan hanya "kriteria terbaik per rasio" seperti rencana
awal CLAUDE.md Bagian 8 butir 10) dipilih karena: (a) ada 3 rasio dengan
SERI antar kriteria pada seed 42 (20%: GM=Ent; 40%: L1=GM; 70%: GM=Ent),
jadi "terbaik per rasio" tidak selalu tunggal; (b) titik yang dicurigai
di atas (L1@20%, Entropi@50/70%) tidak semuanya masuk kriteria "terbaik"
pada rasio itu, jadi cakupan sempit tidak otomatis menjawabnya.

Skor mentah tiap kriteria (L1, GM-Ekspansi, Entropi) dihitung SEKALI dari
checkpoints/baseline_9class.pth (deterministik, TIDAK bergantung rasio
maupun seed) dan dipakai ulang untuk ketujuh rasio -- mask pemangkasan
per (kriteria, rasio) karena itu IDENTIK di ketiga seed; variasi akurasi
antar seed murni variasi fine-tuning. compute_l1_scores(),
compute_entropy_scores(), normalize_min_max(), get_pruning_mask(),
apply_pruning() (src/pruning.py) dan hitung_skor_gm_ekspansi_semua_layer()
(diimpor dinamis dari 43_ablation_gm_expansion.py) dipakai APA ADANYA,
TIDAK diubah.

Urutan eksekusi (prioritas menjawab kecurigaan spesifik dulu, supaya
kalau proses terhenti, titik yang paling dibutuhkan sudah lebih dulu
lengkap):
    1. L1 @ 10,20,30,40,50%   (bentuk-V dan segmen datar yang dicurigai)
    2. Entropi @ 50,60,70%    (lonjakan di 60% yang dicurigai)
    3. L1 @ 60,70%            (melengkapi L1)
    4. GM-Ekspansi @ 10-70%   (belum disinggung kecurigaan, tapi perlu
                                 untuk cakupan penuh 3 kriteria x 7 rasio)
    5. Entropi @ 10,20,30,40% (melengkapi Entropi)
Tiap pasang (kriteria, rasio): seed 123 lalu 2024 sebelum lanjut ke
pasang berikutnya (supaya mean+std tiap titik lengkap secepat mungkin).

Checkpoint memakai akhiran _seed{123,2024}, TIDAK menimpa checkpoint
seed 42 yang sudah ada (ablation_l1_*, ablation_gmexp_*,
ablation_entropy_*_traincal.pth). Skip-jika-checkpoint-ada untuk resume.
Progres disimpan ke outputs/multiseed_single_criteria_results.json
setiap SATU run selesai.

Jalankan:
    venv/Scripts/python.exe scripts/48_multiseed_single_criteria.py
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
from src.model import train_model, evaluate_model, load_checkpoint
from src.pruning import (
    compute_l1_scores, compute_entropy_scores,
    normalize_min_max, get_pruning_mask, apply_pruning,
)

# ----- Impor dinamis hitung_skor_gm_ekspansi_semua_layer dari skrip 43 -----
_JALUR_SKRIP_43 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "43_ablation_gm_expansion.py")
_spec_43 = importlib.util.spec_from_file_location("_skrip_43_ablation_gm_expansion", _JALUR_SKRIP_43)
_mod_43 = importlib.util.module_from_spec(_spec_43)
_spec_43.loader.exec_module(_mod_43)
hitung_skor_gm_ekspansi_semua_layer = _mod_43.hitung_skor_gm_ekspansi_semua_layer

SEEDS_BARU = [123, 2024]
EPOCHS = CFG.BASELINE_EPOCHS  # 30
SPLIT_DIR = CFG.DATASET_DIR.parent / "dataset_split"

TABEL_LENGKAP_PATH = CFG.OUTPUT_DIR / "tabel_hasil_lengkap.json"
GM_EXPANSION_PATH = CFG.OUTPUT_DIR / "ablation_gm_expansion.json"
RESULTS_PATH = CFG.OUTPUT_DIR / "multiseed_single_criteria_results.json"

# Nama checkpoint seed=42 yang SUDAH ADA (dipakai ulang, TIDAK dilatih ulang)
CKPT_SEED42 = {
    "l1": lambda pct: f"ablation_l1_{pct}pct_30ep.pth",
    "gm": lambda pct: f"ablation_gmexp_{pct}pct_30ep.pth",
    "entropy": lambda pct: f"ablation_entropy_{pct}pct_30ep_traincal.pth",
}
# Nama checkpoint seed baru (123/2024) yang AKAN dilatih
CKPT_SEED_BARU = {
    "l1": lambda pct, seed: f"ablation_l1_{pct}pct_30ep_seed{seed}.pth",
    "gm": lambda pct, seed: f"ablation_gmexp_{pct}pct_30ep_seed{seed}.pth",
    "entropy": lambda pct, seed: f"ablation_entropy_{pct}pct_30ep_traincal_seed{seed}.pth",
}

RATIOS_PCT = [10, 20, 30, 40, 50, 60, 70]

# Urutan prioritas (kriteria, rasio) -- lihat alasan di docstring atas
URUTAN_PRIORITAS = (
    [("l1", r) for r in [10, 20, 30, 40, 50]] +
    [("entropy", r) for r in [50, 60, 70]] +
    [("l1", r) for r in [60, 70]] +
    [("gm", r) for r in RATIOS_PCT] +
    [("entropy", r) for r in [10, 20, 30, 40]]
)
assert len(URUTAN_PRIORITAS) == 21, f"Harus 21 pasang, dapat {len(URUTAN_PRIORITAS)}"
assert len(set(URUTAN_PRIORITAS)) == 21, "Ada pasangan (kriteria, rasio) duplikat!"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def muat_akurasi_seed42(kriteria, pct):
    """Baca akurasi test seed=42 yang SUDAH ADA (tidak dihitung ulang)."""
    key = f"{pct}%"
    if kriteria == "l1":
        e = json.load(open(TABEL_LENGKAP_PATH, encoding="utf-8"))["results"]["l1"][key]
        return e["accuracy"], e["f1_score"]
    elif kriteria == "entropy":
        e = json.load(open(TABEL_LENGKAP_PATH, encoding="utf-8"))["results"]["entropy"][key]
        return e["accuracy"], e["f1_score"]
    elif kriteria == "gm":
        e = json.load(open(GM_EXPANSION_PATH, encoding="utf-8"))["results"][key]["test"]
        return e["accuracy"], e["f1_score"]


def correct_and_total(metrics):
    cm = metrics["confusion_matrix"]
    return int(cm.trace()), int(cm.sum())


def summarize(per_seed_dict):
    accs = [v["accuracy"] for v in per_seed_dict.values()]
    return {
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "min_accuracy": float(np.min(accs)),
        "max_accuracy": float(np.max(accs)),
        "n_seeds": len(accs),
    }


def run_single_seed(kriteria, pct, masks, seed, device, model_baseline):
    ckpt_name = CKPT_SEED_BARU[kriteria](pct, seed)
    ckpt_path = CFG.CHECKPOINT_DIR / ckpt_name

    set_seed(seed)
    dataloaders, dataset_sizes = get_dataloaders(SPLIT_DIR)

    if ckpt_path.exists():
        print(f"[SKIP] {kriteria} {pct}% seed={seed}: checkpoint sudah ada ({ckpt_name})")
        model, _ = load_checkpoint(ckpt_path, device)
    else:
        print(f"\n{'=' * 60}")
        print(f"{kriteria.upper()} {pct}% seed={seed} (mask identik di ketiga seed)")
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
            phase_name=f"{kriteria.upper()} {pct}% seed={seed}",
        )

    metrics_val, _, _ = evaluate_model(model, dataloaders["val"], device)
    metrics_test, _, _ = evaluate_model(model, dataloaders["test"], device)
    return metrics_val, metrics_test, ckpt_name


def main():
    device = torch.device("cpu")

    print("=" * 70)
    print("VALIDASI MULTI-SEED -- KETIGA KRITERIA TUNGGAL, SELURUH 7 RASIO")
    print("=" * 70)
    print(f"  Seed baru : {SEEDS_BARU} (seed 42 dipakai ulang dari checkpoint yang ada)")
    print(f"  Pasang    : {len(URUTAN_PRIORITAS)} (kriteria x rasio) x {len(SEEDS_BARU)} seed "
          f"= {len(URUTAN_PRIORITAS) * len(SEEDS_BARU)} run")
    print(f"  Epoch     : {EPOCHS}")
    print(f"  Device    : {device}")

    BASELINE_CKPT = CFG.CHECKPOINT_DIR / "baseline_9class.pth"
    if not BASELINE_CKPT.exists():
        print("[ERROR] Checkpoint baseline tidak ditemukan!")
        return

    print("\n[INFO] Memuat baseline & menghitung skor L1, GM-Ekspansi, Entropi sekali "
          "(dipakai ulang untuk ketujuh rasio dan kedua seed baru)...")
    model_baseline, _ = load_checkpoint(BASELINE_CKPT, device)
    model_baseline.eval()

    dataloaders_scoring, _ = get_dataloaders(SPLIT_DIR)
    l1_scores = normalize_min_max(compute_l1_scores(model_baseline))
    gm_scores = normalize_min_max(hitung_skor_gm_ekspansi_semua_layer(model_baseline))
    entropy_scores = normalize_min_max(compute_entropy_scores(
        model_baseline, dataloaders_scoring["train"], device, samples_per_class=20, seed=CFG.SEED
    ))
    scores_by_kriteria = {"l1": l1_scores, "gm": gm_scores, "entropy": entropy_scores}

    # Cache mask per (kriteria, rasio) -- dihitung sekali, dipakai kedua seed baru
    mask_cache = {}

    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, encoding="utf-8") as f:
            all_results = json.load(f)
        print(f"\n[INFO] Melanjutkan dari hasil sebagian yang sudah ada: {RESULTS_PATH}")
    else:
        all_results = {"seeds_baru": SEEDS_BARU, "seed_42_sumber": {
            "l1": str(TABEL_LENGKAP_PATH), "entropy": str(TABEL_LENGKAP_PATH), "gm": str(GM_EXPANSION_PATH)
        }, "epochs": EPOCHS, "kriteria": {}, "status": "BERJALAN"}

    run_plan = [(kriteria, pct, seed) for (kriteria, pct) in URUTAN_PRIORITAS for seed in SEEDS_BARU]
    total_runs = len(run_plan)
    start_time = time.time()

    for i, (kriteria, pct, seed) in enumerate(run_plan, 1):
        elapsed = time.time() - start_time
        print(f"\n>>> Run {i}/{total_runs} | {kriteria} {pct}% seed={seed} "
              f"| Waktu berjalan: {elapsed/60:.1f} menit ({elapsed/3600:.2f} jam)")

        cache_key = (kriteria, pct)
        if cache_key not in mask_cache:
            mask_cache[cache_key] = get_pruning_mask(scores_by_kriteria[kriteria], pct / 100)
        masks = mask_cache[cache_key]

        metrics_val, metrics_test, ckpt_name = run_single_seed(
            kriteria, pct, masks, seed, device, model_baseline
        )
        n_correct, n_total = correct_and_total(metrics_test)

        kriteria_entry = all_results["kriteria"].setdefault(kriteria, {})
        rasio_entry = kriteria_entry.setdefault(f"{pct}%", {"per_seed": {}})

        if "42" not in rasio_entry["per_seed"]:
            acc42, f1_42 = muat_akurasi_seed42(kriteria, pct)
            rasio_entry["per_seed"]["42"] = {
                "accuracy": acc42, "f1_score": f1_42,
                "checkpoint": CKPT_SEED42[kriteria](pct),
                "sumber": "dipakai ulang, tidak dilatih ulang",
            }

        rasio_entry["per_seed"][str(seed)] = {
            "accuracy": metrics_test["accuracy"],
            "f1_score": metrics_test["f1_score"],
            "val_accuracy": metrics_val["accuracy"],
            "correct": n_correct,
            "total": n_total,
            "checkpoint": ckpt_name,
        }
        rasio_entry.update(summarize(rasio_entry["per_seed"]))

        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Hasil sementara disimpan: {RESULTS_PATH}")

    all_results["status"] = "SELESAI"
    all_results["total_time_seconds"] = time.time() - start_time
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print("RINGKASAN -- VALIDASI MULTI-SEED KRITERIA TUNGGAL (3 KRITERIA x 7 RASIO)")
    print(f"{'=' * 70}")
    for kriteria in ["l1", "gm", "entropy"]:
        print(f"\n--- {kriteria.upper()} ---")
        for pct in RATIOS_PCT:
            entry = all_results["kriteria"][kriteria][f"{pct}%"]
            print(f"  {pct}%: mean={entry['mean_accuracy']*100:.2f}%  std={entry['std_accuracy']*100:.2f}pp  "
                  f"min={entry['min_accuracy']*100:.2f}%  maks={entry['max_accuracy']*100:.2f}%")
    print(f"{'=' * 70}")
    print(f"\n[INFO] Hasil disimpan: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
