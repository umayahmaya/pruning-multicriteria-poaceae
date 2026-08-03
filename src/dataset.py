"""
dataset.py - Pengolahan Dataset Multi-Crop Famili Poaceae

Modul ini menangani:
1. Pemuatan gambar dari tiga sumber dataset (padi, tebu, jagung)
2. Penyeimbangan jumlah sampel antar kelas
3. Praproses dan augmentasi data
4. Pembagian terstratifikasi 70/15/15
5. Pembuatan DataLoader untuk training, validasi, dan testing

Struktur folder dataset yang diharapkan:
    dataset/
    ├── Leaf_Blast/
    ├── Brown_Spot/
    ├── Healthy_Rice/
    ├── Rust_Sugarcane/
    ├── Mosaic_Sugarcane/
    ├── Healthy_Sugarcane/
    ├── Common_Rust_Corn/
    ├── Gray_Leaf_Spot_Corn/
    └── Healthy_Corn/
"""

import os
import random
import shutil
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from src.config import CFG


def get_transforms(phase="train"):
    """
    Mengembalikan pipeline transformasi untuk setiap fase.

    Args:
        phase: "train", "val", atau "test"

    Returns:
        transforms.Compose object
    """
    if phase == "train":
        return transforms.Compose([
            transforms.Resize((CFG.IMG_SIZE + 32, CFG.IMG_SIZE + 32)),
            transforms.RandomCrop(CFG.IMG_SIZE),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2,
                saturation=0.2, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


def check_dataset_structure(dataset_dir):
    """
    Memeriksa apakah struktur folder dataset sesuai dengan CLASS_NAMES di config.

    Args:
        dataset_dir: Path ke direktori dataset

    Returns:
        dict: {nama_kelas: jumlah_gambar}
    """
    dataset_dir = Path(dataset_dir)
    class_counts = {}
    missing = []

    for cls_name in CFG.CLASS_NAMES:
        cls_dir = dataset_dir / cls_name
        if not cls_dir.exists():
            missing.append(cls_name)
            class_counts[cls_name] = 0
        else:
            # Hitung file gambar (jpg, jpeg, png)
            count = len([
                f for f in cls_dir.iterdir()
                if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
            ])
            class_counts[cls_name] = count

    if missing:
        print(f"\n[PERINGATAN] Folder kelas berikut TIDAK DITEMUKAN:")
        for m in missing:
            print(f"  - {dataset_dir / m}")
        print(f"\nPastikan folder dataset sudah sesuai struktur yang diharapkan.")
        print(f"Lihat docstring di dataset.py untuk format yang benar.\n")

    print("=" * 60)
    print(f"{'Kelas':<25} {'Tanaman':<10} {'Jumlah':<10}")
    print("=" * 60)
    total = 0
    for cls_name, count in class_counts.items():
        plant = CFG.PLANT_MAP.get(cls_name, "?")
        status = "OK" if count > 0 else "MISSING"
        print(f"{cls_name:<25} {plant:<10} {count:<10} {status}")
        total += count
    print("-" * 60)
    print(f"{'TOTAL':<35} {total}")
    print("=" * 60)

    return class_counts


def balance_dataset(dataset_dir, max_per_class=None):
    """
    Menyeimbangkan dataset dengan membatasi jumlah sampel per kelas.
    Kalau max_per_class=None, gunakan jumlah kelas terkecil sebagai batas.

    CATATAN: Fungsi ini TIDAK menghapus gambar asli.
    Ia hanya mengembalikan daftar path gambar yang sudah diseimbangkan.

    Args:
        dataset_dir: Path ke direktori dataset
        max_per_class: Batas maksimum gambar per kelas (None = auto)

    Returns:
        list of (path, label_idx) tuples
    """
    dataset_dir = Path(dataset_dir)
    class_counts = check_dataset_structure(dataset_dir)

    if max_per_class is None:
        # Gunakan jumlah kelas terkecil yang > 0
        valid_counts = [c for c in class_counts.values() if c > 0]
        if not valid_counts:
            raise ValueError("Tidak ada gambar ditemukan di dataset!")
        max_per_class = min(valid_counts)
        print(f"\n[INFO] Auto-balance: {max_per_class} gambar per kelas")

    balanced_samples = []
    for idx, cls_name in enumerate(CFG.CLASS_NAMES):
        cls_dir = dataset_dir / cls_name
        if not cls_dir.exists():
            continue

        # Kumpulkan semua file gambar
        images = sorted([
            f for f in cls_dir.iterdir()
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
        ])

        # Random sample jika melebihi batas
        if len(images) > max_per_class:
            random.seed(CFG.SEED)
            images = random.sample(images, max_per_class)

        for img_path in images:
            balanced_samples.append((str(img_path), idx))

    print(f"[INFO] Total gambar setelah balancing: {len(balanced_samples)}")
    return balanced_samples


def create_split_folders(dataset_dir, output_dir=None, max_per_class=None):
    """
    Membuat folder train/val/test dengan split terstratifikasi.
    Menyalin (bukan memindahkan) gambar ke folder baru.

    Args:
        dataset_dir: Path ke direktori dataset asli
        output_dir: Path output (default: dataset_dir/../dataset_split)
        max_per_class: Batas gambar per kelas untuk balancing

    Returns:
        dict dengan info split
    """
    dataset_dir = Path(dataset_dir)
    if output_dir is None:
        output_dir = dataset_dir.parent / "dataset_split"
    output_dir = Path(output_dir)

    # Dapatkan sampel yang sudah diseimbangkan
    samples = balance_dataset(dataset_dir, max_per_class)
    paths = [s[0] for s in samples]
    labels = [s[1] for s in samples]

    # Split pertama: train vs (val + test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels,
        test_size=(CFG.VAL_RATIO + CFG.TEST_RATIO),
        stratify=labels,
        random_state=CFG.SEED
    )

    # Split kedua: val vs test
    relative_test_size = CFG.TEST_RATIO / (CFG.VAL_RATIO + CFG.TEST_RATIO)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=relative_test_size,
        stratify=temp_labels,
        random_state=CFG.SEED
    )

    # Salin gambar ke folder split
    split_info = {}
    for phase, phase_paths, phase_labels in [
        ("train", train_paths, train_labels),
        ("val", val_paths, val_labels),
        ("test", test_paths, test_labels),
    ]:
        phase_dir = output_dir / phase
        counts = Counter()

        for img_path, label in zip(phase_paths, phase_labels):
            cls_name = CFG.CLASS_NAMES[label]
            dest_dir = phase_dir / cls_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            src = Path(img_path)
            dst = dest_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            counts[cls_name] += 1

        split_info[phase] = dict(counts)
        print(f"\n[{phase.upper()}] Total: {len(phase_paths)}")
        for cls_name in CFG.CLASS_NAMES:
            print(f"  {cls_name}: {counts.get(cls_name, 0)}")

    return split_info


def get_dataloaders(split_dir):
    """
    Membuat DataLoader untuk train, val, dan test dari folder yang sudah di-split.

    Args:
        split_dir: Path ke direktori dataset_split yang berisi train/val/test

    Returns:
        dict: {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    """
    split_dir = Path(split_dir)

    dataloaders = {}
    dataset_sizes = {}

    for phase in ["train", "val", "test"]:
        phase_dir = split_dir / phase
        if not phase_dir.exists():
            raise FileNotFoundError(
                f"Folder {phase_dir} tidak ditemukan. "
                f"Jalankan create_split_folders() terlebih dahulu."
            )

        transform = get_transforms(phase)
        dataset = datasets.ImageFolder(phase_dir, transform=transform)

        # Verifikasi kelas sesuai urutan
        if dataset.classes != CFG.CLASS_NAMES:
            print(f"[PERINGATAN] Urutan kelas di folder berbeda dari CFG.CLASS_NAMES!")
            print(f"  Folder : {dataset.classes}")
            print(f"  Config : {CFG.CLASS_NAMES}")
            print(f"  Pastikan nama folder sesuai dan urut alfabet = urut config.")

        shuffle = (phase == "train")
        dataloaders[phase] = DataLoader(
            dataset,
            batch_size=CFG.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=CFG.NUM_WORKERS,
            pin_memory=True,
            drop_last=(phase == "train"),
        )
        dataset_sizes[phase] = len(dataset)
        print(f"[{phase.upper()}] {len(dataset)} gambar, "
              f"{len(dataset.classes)} kelas, "
              f"batch_size={CFG.BATCH_SIZE}")

    return dataloaders, dataset_sizes


# ==================== SEL PEMULIHAN ====================
# Jalankan sel ini kalau kernel restart dan kamu ingin langsung
# membuat dataloader tanpa split ulang
def quick_load(split_dir=None):
    """
    Pemulihan cepat: langsung buat dataloader dari folder split yang sudah ada.
    """
    if split_dir is None:
        split_dir = CFG.DATASET_DIR.parent / "dataset_split"
    return get_dataloaders(split_dir)


if __name__ == "__main__":
    # Jalankan langsung untuk mengecek dataset
    print("Memeriksa struktur dataset...\n")
    check_dataset_structure(CFG.DATASET_DIR)
