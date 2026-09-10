"""
21_export_android_assets.py
Menyiapkan seluruh aset untuk aplikasi Android di folder android_assets/.

Isi yang dihasilkan:
  1. Salinan dua checkpoint .pte (rasio 20% dan 60%) dari outputs/ --
     berkas ini WAJIB sudah ada (hasil 19_test_executorch_export.py /
     20_test_executorch_runtime.py). Skrip ini menyalin, BUKAN mengekspor
     ulang, supaya aset Android persis sama dengan berkas yang sudah
     diverifikasi kesetaraan prediksinya di 20_test_executorch_runtime.py.
  2. Salinan outputs/penanganan_penyakit.json
  3. labels.txt -- 9 nama kelas sesuai urutan CFG.CLASS_NAMES, satu per baris
  4. preprocessing_config.json -- parameter praproses PERSIS yang dipakai
     get_transforms("test") di src/dataset.py (resize, interpolasi,
     mean/std normalisasi, urutan channel), diambil langsung dari objek
     transform yang berjalan, tidak diketik ulang sebagai konstanta.
  5. Lima berkas referensi (reference_01.txt .. reference_05.txt): tensor
     input hasil praproses (150.528 angka -- 3x224x224, flatten C-order:
     seluruh piksel channel R, lalu G, lalu B) diikuti 9 logit keluaran
     (urutan CFG.CLASS_NAMES), satu angka per baris, dari model rasio 20%
     (multicriteria_20pct_30ep_valweights.pth). Dipakai memverifikasi
     praproses Android menghasilkan angka yang sama dengan Python.
     Metadata (nama citra asli, kelas benar, jumlah baris) ada di
     reference_manifest.json terpisah -- berkas .txt sendiri murni angka.

Sampel 5 citra: acak deterministik (seed CFG.SEED) dari dataset_split/test.

Jalankan:
    venv/Scripts/python.exe scripts/21_export_android_assets.py
"""

import sys
import os
import json
import random
import shutil
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchvision import datasets

from src.config import CFG
from src.dataset import get_transforms
from src.model import load_checkpoint

ANDROID_ASSETS_DIR = CFG.ROOT_DIR / "android_assets"
PTE_FILENAMES = [
    "multicriteria_20pct_30ep_valweights.pte",
    "multicriteria_60pct_30ep_valweights.pte",
]
REFERENCE_CHECKPOINT = "multicriteria_20pct_30ep_valweights.pth"
N_REFERENCE_IMAGES = 5


def copy_pte_files():
    results = []
    for filename in PTE_FILENAMES:
        src = CFG.OUTPUT_DIR / filename
        if not src.exists():
            raise FileNotFoundError(
                f"{src} tidak ditemukan. Jalankan 19_test_executorch_export.py "
                "(venv/) atau 20_test_executorch_runtime.py (venv_mobile/) dulu "
                "untuk menghasilkan berkas .pte."
            )
        dst = ANDROID_ASSETS_DIR / filename
        shutil.copy2(src, dst)
        results.append({"filename": filename, "size_mb": round(dst.stat().st_size / (1024 * 1024), 3)})
        print(f"  [OK] {filename} -> android_assets/ ({results[-1]['size_mb']} MB)")
    return results


def copy_penanganan_data():
    src = CFG.OUTPUT_DIR / "penanganan_penyakit.json"
    if not src.exists():
        raise FileNotFoundError(f"{src} tidak ditemukan.")
    dst = ANDROID_ASSETS_DIR / "penanganan_penyakit.json"
    shutil.copy2(src, dst)
    print(f"  [OK] penanganan_penyakit.json -> android_assets/")


def write_labels():
    dst = ANDROID_ASSETS_DIR / "labels.txt"
    with open(dst, "w", encoding="utf-8") as f:
        for cls_name in CFG.CLASS_NAMES:
            f.write(cls_name + "\n")
    print(f"  [OK] labels.txt ({len(CFG.CLASS_NAMES)} kelas)")


def write_preprocessing_config(transform):
    """Ambil parameter langsung dari objek transform get_transforms("test")
    yang sesungguhnya berjalan, bukan diketik ulang sebagai konstanta."""
    resize_t = transform.transforms[0]
    normalize_t = transform.transforms[-1]

    config = {
        "resize": {
            "width": resize_t.size[1] if isinstance(resize_t.size, (list, tuple)) else resize_t.size,
            "height": resize_t.size[0] if isinstance(resize_t.size, (list, tuple)) else resize_t.size,
            "interpolation": str(resize_t.interpolation),
            "antialias": bool(resize_t.antialias),
        },
        "channel_order": "RGB",
        "pixel_scale": "PIL uint8 [0, 255] dibagi 255.0 -> float32 [0, 1] (setara torchvision ToTensor())",
        "normalize": {
            "mean": list(normalize_t.mean),
            "std": list(normalize_t.std),
            "formula": "(pixel - mean[c]) / std[c], per channel, setelah pixel_scale di atas",
        },
        "input_tensor_shape_nchw": [1, 3, resize_t.size[0], resize_t.size[1]],
        "catatan_penting": (
            "resize.antialias=true berarti PyTorch memakai bilinear resize DENGAN "
            "antialiasing saat downsampling. Ini BERBEDA secara numerik dari resize "
            "bilinear biasa tanpa antialiasing (mis. Bitmap.createScaledBitmap bawaan "
            "Android). Kalau praproses Android tidak mereplikasi antialiasing ini, "
            "angka di reference_*.txt TIDAK akan cocok persis meski arsitekturnya benar."
        ),
    }

    dst = ANDROID_ASSETS_DIR / "preprocessing_config.json"
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  [OK] preprocessing_config.json")
    return config


def sample_reference_images():
    test_dir = CFG.DATASET_DIR.parent / "dataset_split" / "test"
    transform = get_transforms("test")
    dataset = datasets.ImageFolder(test_dir, transform=transform)
    rng = random.Random(CFG.SEED)
    indices = rng.sample(range(len(dataset)), N_REFERENCE_IMAGES)
    return dataset, indices


def generate_reference_files(dataset, indices):
    ckpt_path = CFG.CHECKPOINT_DIR / REFERENCE_CHECKPOINT
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint referensi tidak ditemukan: {ckpt_path}")

    device = torch.device("cpu")
    model, _ = load_checkpoint(ckpt_path, device)
    model.eval()

    manifest = {
        "reference_checkpoint": REFERENCE_CHECKPOINT,
        "n_input_values": 3 * CFG.IMG_SIZE * CFG.IMG_SIZE,
        "n_output_values": CFG.NUM_CLASSES,
        "file_format": (
            "Satu angka per baris. N_INPUT_VALUES baris pertama adalah tensor input "
            "hasil praproses, flatten C-order (seluruh piksel channel R, lalu G, "
            "lalu B, tiap channel baris demi baris/row-major). "
            "N_OUTPUT_VALUES baris terakhir adalah logit keluaran model mentah "
            "(SEBELUM softmax), urutan sesuai labels.txt / CFG.CLASS_NAMES."
        ),
        "seed": CFG.SEED,
        "images": [],
    }

    with torch.no_grad():
        for i, idx in enumerate(indices, start=1):
            image_tensor, label = dataset[idx]
            image_path = dataset.samples[idx][0]
            input_tensor = image_tensor.unsqueeze(0)

            logits = model(input_tensor)[0]

            ref_filename = f"reference_{i:02d}.txt"
            dst = ANDROID_ASSETS_DIR / ref_filename
            with open(dst, "w", encoding="utf-8") as f:
                for v in image_tensor.flatten().tolist():
                    f.write(repr(float(v)) + "\n")
                for v in logits.tolist():
                    f.write(repr(float(v)) + "\n")

            pred_idx = int(torch.argmax(logits).item())
            manifest["images"].append({
                "reference_file": ref_filename,
                "source_image": os.path.basename(image_path),
                "true_class": CFG.CLASS_NAMES[label],
                "predicted_class": CFG.CLASS_NAMES[pred_idx],
            })
            print(f"  [OK] {ref_filename} <- {os.path.basename(image_path)} "
                  f"(benar={CFG.CLASS_NAMES[label]}, prediksi={CFG.CLASS_NAMES[pred_idx]})")

    manifest_path = ANDROID_ASSETS_DIR / "reference_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  [OK] reference_manifest.json")


def main():
    print("=" * 70)
    print("EKSPOR ASET ANDROID")
    print("=" * 70)

    ANDROID_ASSETS_DIR.mkdir(exist_ok=True)

    print("\n[1/5] Menyalin checkpoint .pte...")
    copy_pte_files()

    print("\n[2/5] Menyalin data penanganan penyakit...")
    copy_penanganan_data()

    print("\n[3/5] Menulis labels.txt...")
    write_labels()

    print("\n[4/5] Menulis preprocessing_config.json...")
    transform = get_transforms("test")
    write_preprocessing_config(transform)

    print("\n[5/5] Membuat berkas referensi praproses + logit...")
    dataset, indices = sample_reference_images()
    generate_reference_files(dataset, indices)

    print(f"\n{'=' * 70}")
    print(f"[SELESAI] Seluruh aset ada di: {ANDROID_ASSETS_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
