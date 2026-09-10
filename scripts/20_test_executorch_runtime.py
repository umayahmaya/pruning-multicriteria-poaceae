"""
20_test_executorch_runtime.py
Uji lengkap ekspor ExecuTorch (poin 1-4) untuk tiga checkpoint:
baseline_9class.pth, multicriteria_20pct_30ep_valweights.pth, dan
multicriteria_60pct_30ep_valweights.pth.

PENTING -- jalankan skrip ini dengan venv_mobile/, BUKAN venv/:
    venv_mobile/Scripts/python.exe scripts/20_test_executorch_runtime.py

venv_mobile/ adalah lingkungan Python terpisah (Python 3.11, executorch
1.0.1, torch 2.9.1+cpu, torchvision 0.24.1) khusus untuk pengujian ini.
venv/ (lingkungan penelitian utama, torch 2.13.0+cpu) TIDAK dipakai untuk
uji runtime karena executorch 1.3.1 (versi terbaru, satu-satunya yang
kompatibel dengan torch 2.13 lewat pip) gagal memuat modul native
_portable_lib di Windows (ImportError: DLL load failed, diduga
ketidakcocokan ABI C++). executorch 1.0.1 dengan torch 2.9.1 (versi
yang secara eksplisit disyaratkan paket itu sendiri) terbukti berfungsi.
venv/ TIDAK diubah untuk mengejar kecocokan ini. Lihat CLAUDE.md bagian
Pekerjaan yang Masih Terbuka untuk detail investigasi.

Untuk tiap model, menguji:
  1. Ekspor ke .pte berhasil tanpa error
  2. Ukuran berkas .pte (MB)
  3. N_SAMPLES citra data uji dijalankan lewat model PyTorch asli DAN
     model hasil ekspor ExecuTorch (lewat executorch.runtime.Runtime),
     prediksi kelas (argmax) harus identik untuk seluruh citra
  4. Selisih nilai logit maksimum antara PyTorch dan ExecuTorch

Sampel citra: acak dengan seed CFG.SEED dari dataset_split/test (175
citra total), deterministik lewat random.Random(seed) lokal.

Catatan: log runtime C++ "InternalConsistency verification requested
but not available" yang mungkin muncul saat load_program() bersifat
informasional (karena etdump tidak diaktifkan), bukan error -- hasil
eksekusi tetap benar.
"""

import sys
import os
import json
import random
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# flatc (kompilator FlatBuffers) harus ditemukan lewat FLATC_EXECUTABLE
# saat venv tidak "diaktifkan" (venv_mobile/Scripts tidak ada di PATH).
_FLATC = Path(sys.executable).parent.parent / "Lib" / "site-packages" / \
    "executorch" / "data" / "bin" / "flatc.exe"
if _FLATC.exists():
    os.environ.setdefault("FLATC_EXECUTABLE", str(_FLATC))

import torch
from torch.export import export
from executorch.exir import to_edge_transform_and_lower
from executorch.runtime import Runtime
from torchvision import datasets

from src.config import CFG
from src.dataset import get_transforms
from src.model import load_checkpoint

MODELS_TO_TEST = [
    "baseline_9class.pth",
    "multicriteria_20pct_30ep_valweights.pth",
    "multicriteria_60pct_30ep_valweights.pth",
]
N_SAMPLES = 20
RESULTS_PATH = CFG.OUTPUT_DIR / "executorch_runtime_test.json"


def sample_test_images():
    test_dir = CFG.DATASET_DIR.parent / "dataset_split" / "test"
    transform = get_transforms("test")
    dataset = datasets.ImageFolder(test_dir, transform=transform)
    rng = random.Random(CFG.SEED)
    indices = rng.sample(range(len(dataset)), N_SAMPLES)
    return dataset, indices


def test_model(ckpt_name, device, dataset, indices):
    ckpt_path = CFG.CHECKPOINT_DIR / ckpt_name
    if not ckpt_path.exists():
        return {"checkpoint": ckpt_name, "status": "ERROR",
                "error": f"Checkpoint tidak ditemukan: {ckpt_path}"}

    model, _ = load_checkpoint(ckpt_path, device)
    model.eval()

    example_input = dataset[indices[0]][0].unsqueeze(0)

    try:
        exported_program = export(model, (example_input,))
        edge_program = to_edge_transform_and_lower(exported_program)
        executorch_program = edge_program.to_executorch()
    except Exception as e:
        return {"checkpoint": ckpt_name, "status": "ERROR",
                "error": f"Ekspor gagal: {type(e).__name__}: {e}"}

    pte_path = CFG.OUTPUT_DIR / f"{Path(ckpt_name).stem}.pte"
    with open(pte_path, "wb") as f:
        f.write(executorch_program.buffer)
    size_mb = pte_path.stat().st_size / (1024 * 1024)

    runtime = Runtime.get()
    program = runtime.load_program(str(pte_path))
    method = program.load_method("forward")

    max_logit_diff = 0.0
    mismatches = []
    n_tested = 0

    with torch.no_grad():
        for idx in indices:
            image_tensor, label = dataset[idx]
            image_path = dataset.samples[idx][0]
            input_tensor = image_tensor.unsqueeze(0)

            pt_logits = model(input_tensor)[0]
            et_outputs = method.execute([input_tensor])
            et_logits = et_outputs[0][0]

            diff = (pt_logits - et_logits).abs().max().item()
            max_logit_diff = max(max_logit_diff, diff)

            pt_pred = int(torch.argmax(pt_logits).item())
            et_pred = int(torch.argmax(et_logits).item())
            n_tested += 1

            if pt_pred != et_pred:
                mismatches.append({
                    "filename": os.path.basename(image_path),
                    "true_label": CFG.CLASS_NAMES[label],
                    "pytorch_pred": CFG.CLASS_NAMES[pt_pred],
                    "executorch_pred": CFG.CLASS_NAMES[et_pred],
                })

    return {
        "checkpoint": ckpt_name,
        "status": "OK",
        "pte_path": str(pte_path),
        "size_mb": round(size_mb, 3),
        "n_images_tested": n_tested,
        "n_prediction_mismatches": len(mismatches),
        "predictions_identical": len(mismatches) == 0,
        "mismatches": mismatches,
        "max_logit_diff": max_logit_diff,
    }


def main():
    device = torch.device("cpu")
    dataset, indices = sample_test_images()

    print("=" * 70)
    print("UJI RUNTIME EXECUTORCH LENGKAP (POIN 1-4)")
    print(f"Python     : {sys.executable}")
    print(f"torch      : {torch.__version__}")
    print(f"FLATC_EXECUTABLE : {os.environ.get('FLATC_EXECUTABLE', '(tidak diset)')}")
    print(f"Sampel     : {N_SAMPLES} citra (seed={CFG.SEED}) dari dataset_split/test")
    print("=" * 70)

    results = []
    for ckpt_name in MODELS_TO_TEST:
        print(f"\n[INFO] Menguji {ckpt_name}...")
        result = test_model(ckpt_name, device, dataset, indices)
        results.append(result)
        if result["status"] == "OK":
            print(f"  [OK] Ekspor  : {result['pte_path']}")
            print(f"  [OK] Ukuran  : {result['size_mb']} MB")
            n_match = result["n_images_tested"] - result["n_prediction_mismatches"]
            ok = result["predictions_identical"]
            print(f"  [{'OK' if ok else 'GAGAL'}] Prediksi: "
                  f"{'IDENTIK' if ok else 'BERBEDA'} "
                  f"({n_match}/{result['n_images_tested']} cocok)")
            print(f"  [INFO] Selisih logit maksimum: {result['max_logit_diff']:.3e}")
            for m in result["mismatches"]:
                print(f"    - MISMATCH {m['filename']}: label={m['true_label']} "
                      f"PyTorch={m['pytorch_pred']} ExecuTorch={m['executorch_pred']}")
        else:
            print(f"  [ERROR] {result['error']}")

    save_data = {
        "environment": {
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
        },
        "n_samples": N_SAMPLES,
        "seed": CFG.SEED,
        "tested_points": ["1_export_success", "2_file_size_mb",
                           "3_prediction_equivalence", "4_max_logit_diff"],
        "results": results,
    }
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print(f"[INFO] Hasil disimpan: {RESULTS_PATH}")
    n_ok = sum(1 for r in results if r["status"] == "OK" and r.get("predictions_identical"))
    print(f"[INFO] {n_ok}/{len(results)} model lolos seluruh 4 poin pengujian.")


if __name__ == "__main__":
    main()
