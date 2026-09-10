"""
22_copy_reference_images.py
Menyalin citra sumber untuk kelima berkas referensi (lihat
21_export_android_assets.py) ke android_assets/, dikonversi ke PNG
(lossless) agar seragam, TANPA diubah ukurannya -- dimensi asli
dipertahankan supaya sisi Android tetap menjalankan praproses (termasuk
resize) sendiri saat memverifikasi terhadap reference_NN.txt.

PNG dipakai, bukan JPEG, karena JPEG bersifat lossy bahkan pada kualitas
100 -- piksel yang dibaca Android dari JPEG bisa sedikit berbeda dari
piksel asli yang dipakai menghasilkan reference_NN.txt. PNG lossless
menjamin piksel yang dibaca Android identik dengan yang dibaca PIL di
sini.

Membaca android_assets/reference_manifest.json (dihasilkan skrip 21),
mencari tiap source_image di dataset_split/test/<true_class>/, menyalin
sebagai android_assets/reference_01.png .. reference_05.png (nomor
mengikuti reference_file), lalu menambahkan field "image_file" ke tiap
entri di reference_manifest.json.

Jalankan:
    venv/Scripts/python.exe scripts/22_copy_reference_images.py
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

from src.config import CFG

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

    print("=" * 70)
    print("SALIN CITRA REFERENSI (KONVERSI KE PNG LOSSLESS)")
    print("=" * 70)

    for entry in manifest["images"]:
        source_image = entry["source_image"]
        true_class = entry["true_class"]
        reference_file = entry["reference_file"]

        src_path = TEST_DIR / true_class / source_image
        if not src_path.exists():
            print(f"[ERROR] Citra sumber tidak ditemukan: {src_path}")
            sys.exit(1)

        image_file = f"{Path(reference_file).stem}.png"
        dst_path = ANDROID_ASSETS_DIR / image_file

        with Image.open(src_path) as img:
            img = img.convert("RGB")
            img.save(dst_path, "PNG")

        entry["image_file"] = image_file
        print(f"  [OK] {source_image} ({true_class}) -> android_assets/{image_file} "
              f"({img.width}x{img.height})")

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] reference_manifest.json diperbarui dengan field image_file")

    print(f"\n{'=' * 70}")
    print(f"[SELESAI] {len(manifest['images'])} citra referensi disalin.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
