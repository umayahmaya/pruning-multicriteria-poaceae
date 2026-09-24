"""
55_export_verify_pte_per_rasio.py
Ekspor + verifikasi ExecuTorch untuk checkpoint FORMULA AKTIF (L1 +
GM-Ekspansi + Entropi, bobot per rasio -- CLAUDE.md Bagian 2), rasio 20%
dan 60% -- dua rasio yang sama yang dipakai aplikasi Android.

Menjawab CLAUDE.md Bagian 8 butir 11 (bagian ExecuTorch/Android): checkpoint
.pte yang selama ini disalin scripts/21_export_android_assets.py berasal
dari formula LAMA (`multicriteria_{20,60}pct_30ep_valweights.pth`), belum
pernah diekspor ulang untuk formula aktif. Skrip ini mengisi kekosongan
itu -- pola IDENTIK dengan scripts/20_test_executorch_runtime.py (ekspor,
simpan .pte, jalankan lewat executorch.runtime.Runtime, bandingkan
prediksi + selisih logit dengan model PyTorch asli), scripts/20 SENDIRI
TIDAK diubah (checkpoint yang dirujuknya, formula lama, dan hasil
verifikasinya yang sudah ada tetap sah sebagai arsip -- lihat
outputs/README_OUTPUTS.md bagian archive/lain/).

PENTING -- jalankan skrip ini dengan venv_mobile/, BUKAN venv/ (alasan
sama seperti skrip 20 -- executorch 1.3.1 gagal dimuat di Windows dengan
torch yang terpasang di venv/, venv_mobile/ punya executorch 1.0.1 +
torch 2.9.1 yang terbukti berfungsi. Lihat CLAUDE.md Bagian 8 butir 8):
    venv_mobile/Scripts/python.exe scripts/55_export_verify_pte_per_rasio.py

Untuk tiap model, menguji hal yang SAMA seperti skrip 20:
  1. Ekspor ke .pte berhasil tanpa error
  2. Ukuran berkas .pte (MB)
  3. N_SAMPLES citra data uji dijalankan lewat model PyTorch asli DAN
     model hasil ekspor ExecuTorch, prediksi kelas (argmax) harus identik
  4. Selisih nilai logit maksimum antara PyTorch dan ExecuTorch

Sampel citra: acak dengan seed CFG.SEED dari dataset_split/test (175
citra total), deterministik lewat random.Random(seed) lokal -- indeks
sampel yang SAMA persis dengan skrip 20 (seed identik, urutan
ImageFolder identik), supaya hasil kedua skrip bisa dibandingkan
apple-to-apple kalau diperlukan.
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
    "multicriteria_per_rasio_20pct_30ep.pth",
    "multicriteria_per_rasio_60pct_30ep.pth",
]
N_SAMPLES = 20
RESULTS_PATH = CFG.OUTPUT_DIR / "executorch_runtime_test_per_rasio.json"


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
    print("EKSPOR + VERIFIKASI EXECUTORCH -- FORMULA AKTIF, RASIO 20% & 60%")
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
        "catatan": (
            "Formula AKTIF (L1 + GM-Ekspansi + Entropi, bobot per rasio). "
            "Menggantikan outputs/executorch_runtime_test.json (formula lama) "
            "sebagai dasar checkpoint .pte untuk android_assets/ -- lihat "
            "CLAUDE.md Bagian 8 butir 11."
        ),
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
