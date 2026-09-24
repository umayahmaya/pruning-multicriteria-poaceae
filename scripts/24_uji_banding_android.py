"""
24_uji_banding_android.py
Menjalankan kelima citra yang disalin ke emulator Android lewat adb push
(lihat android_assets/, berkas uji_*.{jpg,png,jpeg} di /sdcard/Pictures/)
lewat model PyTorch, mencetak tiga kelas teratas beserta probabilitasnya
untuk tiap citra -- format yang sepadan dengan log "UjiAplikasi" di
aplikasi Android (lihat catatHasilKeLog di MainActivity.kt) supaya bisa
dibandingkan langsung berdampingan.

CATATAN: kelima berkas ini BUKAN reference_01..05 yang dipakai
22_copy_reference_images.py / PreprocessTest.kt -- ini kumpulan citra
terpisah yang khusus disalin untuk pengujian manual lewat antarmuka
aplikasi (kamera/galeri), lihat percakapan "Salin 5 gambar uji ke
emulator Android".

Memuat checkpoints/multicriteria_per_rasio_20pct_30ep.pth (formula AKTIF --
diperbaiki 2026-09-24, sebelumnya checkpoint formula lama _valweights,
lihat CLAUDE.md Bagian 8 butir 11), menjalankan
praproses evaluasi standar get_transforms("test") -- praproses PyTorch
biasa (resize bilinear + antialias), BUKAN tiruan praprosesBertahap
Android -- sehingga angka di sini adalah baseline "PyTorch murni" untuk
citra yang sama, dipakai membandingkan apakah prediksi aplikasi Android
(yang memakai resize bertahapnya sendiri) tetap sama kelas teratasnya.

Skrip ini TIDAK melatih apa pun -- murni muat checkpoint + evaluasi.

Jalankan:
    venv/Scripts/python.exe scripts/24_uji_banding_android.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image

from src.config import CFG
from src.dataset import get_transforms
from src.model import load_checkpoint

CHECKPOINT_NAME = "multicriteria_per_rasio_20pct_30ep.pth"
TEST_DIR = CFG.DATASET_DIR.parent / "dataset_split" / "test"

# Kelima berkas yang disalin ke /sdcard/Pictures/ emulator lewat adb push,
# dengan nama uji_<kelas>.<ekstensi> di Android -- lihat percakapan
# "Salin 5 gambar uji ke emulator Android".
BERKAS_UJI = [
    ("Brown_Spot_Rice", "Brown_spot  (109).jpg", "uji_Brown_Spot_Rice.jpg"),
    ("Common_Rust_Corn", "IMG_20230429_174905.png", "uji_Common_Rust_Corn.png"),
    ("Healthy_Corn", "IMG_20230429_183326.png", "uji_Healthy_Corn.png"),
    ("Mosaic_Sugarcane", "mosaic (11).jpeg", "uji_Mosaic_Sugarcane.jpeg"),
    ("Rust_Sugarcane", "rust (131).jpeg", "uji_Rust_Sugarcane.jpeg"),
]


def main():
    ckpt_path = CFG.CHECKPOINT_DIR / CHECKPOINT_NAME
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint tidak ditemukan: {ckpt_path}")
        return

    device = torch.device("cpu")
    model, _ = load_checkpoint(ckpt_path, device)
    model.eval()

    transform = get_transforms("test")

    print("=" * 70)
    print("UJI BANDING: PREDIKSI PYTORCH UNTUK CITRA YANG SAMA DENGAN ANDROID")
    print(f"Checkpoint: {CHECKPOINT_NAME}")
    print("=" * 70)

    with torch.no_grad():
        for true_class, source_image, nama_android in BERKAS_UJI:
            src_path = TEST_DIR / true_class / source_image

            if not src_path.exists():
                print(f"\n[ERROR] Citra tidak ditemukan: {src_path}")
                continue

            image = Image.open(src_path).convert("RGB")
            tensor = transform(image).unsqueeze(0)

            logits = model(tensor)[0]
            probs = torch.nn.functional.softmax(logits, dim=0)

            tiga_teratas = torch.topk(probs, k=3)
            teks_tiga_teratas = ", ".join(
                f"{CFG.CLASS_NAMES[idx]} ({prob.item() * 100:.2f}%)"
                for prob, idx in zip(tiga_teratas.values, tiga_teratas.indices)
            )

            print(f"\nBerkas={nama_android} (sumber: {source_image}, kelas asli: {true_class})")
            print(f"  Tiga teratas (PyTorch): {teks_tiga_teratas}")

    print(f"\n{'=' * 70}")


if __name__ == "__main__":
    main()