"""
01_prepare_dataset.py
Persiapan dataset: pengecekan struktur, penyeimbangan kelas, dan split 70/15/15

Jalankan:
    python scripts/01_prepare_dataset.py
    python scripts/01_prepare_dataset.py --max_per_class 400
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CFG
from src.dataset import check_dataset_structure, create_split_folders, get_dataloaders


def main():
    parser = argparse.ArgumentParser(description="Persiapan Dataset Multi-Crop Poaceae")
    parser.add_argument("--max_per_class", type=int, default=None,
                        help="Batas maksimum gambar per kelas (None = auto-balance)")
    parser.add_argument("--force_resplit", action="store_true",
                        help="Paksa split ulang meskipun folder sudah ada")
    args = parser.parse_args()

    print("=" * 60)
    print("TAHAP 1: PERSIAPAN DATASET")
    print("=" * 60)

    # 1. Periksa struktur dataset
    print("\n[1/3] Memeriksa struktur dataset...\n")
    class_counts = check_dataset_structure(CFG.DATASET_DIR)

    # Validasi apakah semua kelas ada
    missing = [cls for cls, count in class_counts.items() if count == 0]
    if missing:
        print(f"\n[ERROR] {len(missing)} kelas tidak ditemukan!")
        print(f"Pastikan folder berikut sudah terisi gambar di: {CFG.DATASET_DIR}")
        for m in missing:
            print(f"  - {m}/")
        print("\nProses dihentikan.")
        return

    # 2. Buat split
    SPLIT_DIR = CFG.DATASET_DIR.parent / "dataset_split"

    if SPLIT_DIR.exists() and not args.force_resplit:
        print(f"\n[2/3] Split dataset sudah ada di: {SPLIT_DIR}")
        print("Gunakan --force_resplit untuk membuat ulang.")
    else:
        if SPLIT_DIR.exists():
            import shutil
            shutil.rmtree(SPLIT_DIR)
            print(f"\n[2/3] Folder split lama dihapus.")

        print(f"\n[2/3] Membuat split dataset...")
        split_info = create_split_folders(
            dataset_dir=CFG.DATASET_DIR,
            output_dir=SPLIT_DIR,
            max_per_class=args.max_per_class
        )

    # 3. Verifikasi DataLoader
    print(f"\n[3/3] Memverifikasi DataLoader...")
    dataloaders, dataset_sizes = get_dataloaders(SPLIT_DIR)

    print(f"\n{'=' * 60}")
    print(f"RINGKASAN DATASET")
    print(f"{'=' * 60}")
    print(f"  Train : {dataset_sizes['train']} gambar")
    print(f"  Val   : {dataset_sizes['val']} gambar")
    print(f"  Test  : {dataset_sizes['test']} gambar")
    print(f"  Total : {sum(dataset_sizes.values())} gambar")
    print(f"  Kelas : {CFG.NUM_CLASSES}")
    print(f"{'=' * 60}")
    print(f"\n[SELESAI] Dataset siap. Lanjutkan ke 02_train_baseline.py")


if __name__ == "__main__":
    main()
