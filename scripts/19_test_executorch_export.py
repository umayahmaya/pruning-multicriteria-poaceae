"""
19_test_executorch_export.py
Menguji ekspor model ke format ExecuTorch (.pte) untuk tiga checkpoint:
baseline_9class.pth, multicriteria_20pct_30ep_valweights.pth, dan
multicriteria_60pct_30ep_valweights.pth.

KETERBATASAN LINGKUNGAN (lihat CLAUDE.md bagian Pekerjaan yang Masih
Terbuka, butir 8 -- SELESAI lewat skrip 20): executorch.runtime.Runtime,
komponen yang menjalankan berkas .pte hasil ekspor, gagal dimuat di
venv/ ini (ImportError: DLL load failed while importing _portable_lib),
diduga cacat pada wheel Windows executorch 1.3.1 itu sendiri. Skrip ini
karena itu HANYA menguji:
  1. Ekspor berhasil tanpa error
  2. Ukuran berkas .pte hasil ekspor (MB)

Perbandingan prediksi PyTorch vs ExecuTorch (poin 3) dan selisih logit
maksimum (poin 4) TIDAK diuji di sini -- lihat scripts/20_test_executorch_
runtime.py, yang dijalankan lewat venv_mobile/ (executorch 1.0.1 + torch
2.9.1+cpu, terbukti berfungsi), untuk pengujian lengkap.

Jalankan:
    venv/Scripts/python.exe scripts/19_test_executorch_export.py
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.export import export
from executorch.exir import to_edge_transform_and_lower

from src.config import CFG
from src.model import load_checkpoint

MODELS_TO_TEST = [
    "baseline_9class.pth",
    "multicriteria_20pct_30ep_valweights.pth",
    "multicriteria_60pct_30ep_valweights.pth",
]

RESULTS_PATH = CFG.OUTPUT_DIR / "executorch_export_test.json"


def export_model(ckpt_name, device):
    ckpt_path = CFG.CHECKPOINT_DIR / ckpt_name
    if not ckpt_path.exists():
        return {"checkpoint": ckpt_name, "status": "ERROR",
                "error": f"Checkpoint tidak ditemukan: {ckpt_path}"}

    model, _ = load_checkpoint(ckpt_path, device)
    model.eval()

    example_inputs = (torch.randn(1, 3, CFG.IMG_SIZE, CFG.IMG_SIZE),)

    try:
        exported_program = export(model, example_inputs)
        edge_program = to_edge_transform_and_lower(exported_program)
        executorch_program = edge_program.to_executorch()
    except Exception as e:
        return {"checkpoint": ckpt_name, "status": "ERROR",
                "error": f"{type(e).__name__}: {e}"}

    pte_path = CFG.OUTPUT_DIR / f"{Path(ckpt_name).stem}.pte"
    with open(pte_path, "wb") as f:
        f.write(executorch_program.buffer)

    size_mb = pte_path.stat().st_size / (1024 * 1024)

    return {
        "checkpoint": ckpt_name,
        "status": "OK",
        "pte_path": str(pte_path),
        "size_mb": round(size_mb, 3),
    }


def main():
    device = torch.device("cpu")
    results = []

    print("=" * 70)
    print("UJI EKSPOR EXECUTORCH (POIN 1-2 SAJA)")
    print("Poin 3 (kesamaan prediksi) dan poin 4 (selisih logit) TIDAK diuji")
    print("di sini -- runtime ExecuTorch tidak bisa dimuat di venv/ ini.")
    print("=" * 70)

    for ckpt_name in MODELS_TO_TEST:
        print(f"\n[INFO] Mengekspor {ckpt_name}...")
        result = export_model(ckpt_name, device)
        results.append(result)
        if result["status"] == "OK":
            print(f"  [OK] {result['pte_path']}")
            print(f"  [OK] Ukuran: {result['size_mb']} MB")
        else:
            print(f"  [ERROR] {result['error']}")

    save_data = {
        "tested_points": ["1_export_success", "2_file_size_mb"],
        "not_tested_points": ["3_prediction_equivalence_20_images", "4_max_logit_diff"],
        "limitation_note": (
            "executorch.runtime.Runtime gagal dimuat di venv/ ini (ImportError: DLL "
            "load failed while importing _portable_lib), diduga ketidakcocokan ABI "
            "C++ antara wheel executorch dan versi torch yang terpasang. Perbandingan "
            "prediksi/logit PyTorch vs ExecuTorch memerlukan lingkungan Python terpisah "
            "(venv_mobile/) dengan versi torch yang kompatibel dengan executorch."
        ),
        "results": results,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print(f"[INFO] Hasil disimpan: {RESULTS_PATH}")
    n_ok = sum(1 for r in results if r["status"] == "OK")
    print(f"[INFO] {n_ok}/{len(results)} model berhasil diekspor.")


if __name__ == "__main__":
    main()
