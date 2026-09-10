"""
23_export_224_reference.py
Membuat satu berkas referensi tambahan untuk menguji NORMALISASI SAJA,
tanpa resize -- pelengkap reference_01..05 (lihat 21_export_android_assets.py
dan 22_copy_reference_images.py).

Memakai citra sumber yang SAMA dengan reference_01 (supaya bisa dibandingkan
langsung): me-resize-nya ke 224x224 memakai langkah Resize PERSIS dari
get_transforms("test") (interpolasi dan antialias yang sama), lalu
menyimpan hasil resize itu SENDIRI sebagai android_assets/reference_224.png
(PNG lossless, sudah berukuran 224x224).

Karena reference_224.png sudah 224x224, praproses di sisi Android untuk
berkas ini TIDAK PERLU resize sama sekali -- hanya baca piksel + normalisasi.
Kalau angka yang dihasilkan Android dari reference_224.png cocok dengan
reference_224.txt, tapi angka dari reference_01.png (yang perlu di-resize
dulu) TIDAK cocok dengan reference_01.txt, itu mengisolasi sumber selisihnya
ke langkah RESIZE, bukan ke normalisasi atau pembacaan piksel.

reference_224.txt berisi 150.528 angka (3x224x224, flatten C-order: seluruh
piksel channel R, lalu G, lalu B), satu angka per baris -- tensor hasil
praproses PENUH (resize + ToTensor + Normalize) dari citra sumber asli.
Secara numerik identik dengan hasil memuat reference_224.png lalu HANYA
menormalisasi (tanpa resize), karena langkah resize sudah "dibekukan" ke
dalam isi PNG-nya.

Jalankan:
    venv/Scripts/python.exe scripts/23_export_224_reference.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

from src.config import CFG
from src.dataset import get_transforms

ANDROID_ASSETS_DIR = CFG.ROOT_DIR / "android_assets"
MANIFEST_PATH = ANDROID_ASSETS_DIR / "reference_manifest.json"
TEST_DIR = CFG.DATASET_DIR.parent / "dataset_split" / "test"


def main():
    if not MANIFEST_PATH.exists():
        print(f"[ERROR] {MANIFEST_PATH} tidak ditemukan.")
        print("Jalankan 21_export_android_assets.py terlebih dahulu.")
        sys.exit(1)

    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    ref01 = next((e for e in manifest["images"] if e["reference_file"] == "reference_01.txt"), None)
    if ref01 is None:
        print("[ERROR] Entri reference_01.txt tidak ditemukan di reference_manifest.json.")
        print("Jalankan 21_export_android_assets.py terlebih dahulu.")
        sys.exit(1)

    source_image = ref01["source_image"]
    true_class = ref01["true_class"]
    src_path = TEST_DIR / true_class / source_image
    if not src_path.exists():
        print(f"[ERROR] Citra sumber tidak ditemukan: {src_path}")
        sys.exit(1)

    print("=" * 70)
    print("BUAT BERKAS REFERENSI 224x224 (UJI NORMALISASI TANPA RESIZE)")
    print("=" * 70)
    print(f"[INFO] Citra sumber: {src_path}")
    print(f"[INFO] (Citra sama dengan reference_01, untuk perbandingan langsung)")

    transform = get_transforms("test")
    resize_t = transform.transforms[0]

    png_path = ANDROID_ASSETS_DIR / "reference_224.png"
    txt_path = ANDROID_ASSETS_DIR / "reference_224.txt"

    with Image.open(src_path) as raw_img:
        img = raw_img.convert("RGB")
        resized_img = resize_t(img)
        resized_img.save(png_path, "PNG")
        print(f"  [OK] {png_path.name} ({resized_img.width}x{resized_img.height})")

        input_tensor = transform(img)  # resize + ToTensor + Normalize dari citra ASLI

    n_values = input_tensor.numel()
    with open(txt_path, "w", encoding="utf-8") as f:
        for v in input_tensor.flatten().tolist():
            f.write(repr(float(v)) + "\n")
    print(f"  [OK] {txt_path.name} ({n_values} angka)")

    manifest["reference_224"] = {
        "note": (
            "Pelengkap reference_01..05: reference_224.png sudah 224x224 "
            "(hasil langkah resize get_transforms('test') yang dibekukan), "
            "jadi praproses Android untuk berkas ini TIDAK PERLU resize -- "
            "hanya baca piksel + normalisasi. reference_224.txt = tensor "
            "hasil praproses PENUH (resize + ToTensor + Normalize) dari "
            "citra sumber yang sama dengan reference_01."
        ),
        "source_image": source_image,
        "true_class": true_class,
        "image_file": png_path.name,
        "reference_file": txt_path.name,
        "n_input_values": n_values,
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"  [OK] reference_manifest.json diperbarui (kunci reference_224)")

    print(f"\n{'=' * 70}")
    print(f"[SELESAI]")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
