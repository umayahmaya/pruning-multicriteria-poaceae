"""
10_smoke_test_entropy.py
Smoke test untuk compute_entropy_scores() (src/pruning.py).

Memanggil fungsi langsung terhadap dataloaders["train"] TANPA training,
untuk memverifikasi:
  1. Jumlah gambar kalibrasi total (harus 180)
  2. Rincian jumlah gambar per kelas (harus 20 x 9 kelas)
  3. Jumlah lapisan yang menghasilkan skor entropi (harus 16)
  4. Rentang nilai entropi (min, max) per lapisan
  5. Determinisme: dua pemanggilan harus menghasilkan skor identik
  6. Guard RuntimeError (pemetaan indeks train vs eval) tidak terpicu

Jalankan:
    python scripts/10_smoke_test_entropy.py
"""

import sys
import os
import random
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import CFG
from src.dataset import get_dataloaders
from src.model import create_mobilenetv2, load_checkpoint
from src.pruning import compute_entropy_scores

GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

all_passed = True


def check(label, condition, detail=""):
    """Cetak satu hasil verifikasi PASS/FAIL."""
    global all_passed
    if condition:
        print(f"  {GREEN}PASS{RESET}  {label}")
    else:
        all_passed = False
        msg = f"  {RED}FAIL{RESET}  {label}"
        if detail:
            msg += f" -> {detail}"
        print(msg)


def expected_calibration_selection(dataset, samples_per_class, seed):
    """
    Reproduksi independen logika pemilihan sampel di compute_entropy_scores
    (src/pruning.py), dipakai di sini hanya untuk memverifikasi hasilnya
    dari luar -- bukan dipanggil oleh fungsi asli.
    """
    indices_by_class = {}
    for idx, label in enumerate(dataset.targets):
        indices_by_class.setdefault(label, []).append(idx)

    rng = random.Random(seed)
    selected_by_class = {}
    for label in sorted(indices_by_class):
        class_indices = sorted(indices_by_class[label])
        rng.shuffle(class_indices)
        selected_by_class[label] = class_indices[:samples_per_class]

    return selected_by_class


def main():
    start_time = time.time()

    print("=" * 60)
    print("SMOKE TEST: compute_entropy_scores()")
    print("=" * 60)

    device = torch.device("cpu")

    SPLIT_DIR = CFG.DATASET_DIR.parent / "dataset_split"
    dataloaders, _ = get_dataloaders(SPLIT_DIR)
    train_loader = dataloaders["train"]

    BASELINE_CKPT = CFG.CHECKPOINT_DIR / "baseline_9class.pth"
    if BASELINE_CKPT.exists():
        print(f"\n[INFO] Memuat checkpoint baseline: {BASELINE_CKPT}")
        model, _ = load_checkpoint(BASELINE_CKPT, device)
    else:
        print(f"\n[PERINGATAN] Checkpoint baseline tidak ditemukan, "
              f"pakai model MobileNetV2 belum-dilatih.")
        model = create_mobilenetv2(num_classes=CFG.NUM_CLASSES, pretrained=False).to(device)

    samples_per_class = 20
    seed = CFG.SEED

    # ================================================================
    # 1-2: Jumlah gambar kalibrasi total dan per kelas
    # ================================================================
    print(f"\n{'-' * 60}")
    print("1-2. Jumlah gambar kalibrasi (total & per kelas)")
    print(f"{'-' * 60}")

    selected_by_class = expected_calibration_selection(
        train_loader.dataset, samples_per_class, seed
    )
    total_selected = sum(len(v) for v in selected_by_class.values())

    check(f"Total gambar kalibrasi = 180 (terdeteksi: {total_selected})",
          total_selected == 180)

    for label in sorted(selected_by_class):
        cls_name = CFG.CLASS_NAMES[label]
        n = len(selected_by_class[label])
        check(f"Kelas {cls_name}: 20 gambar (terdeteksi: {n})", n == 20)

    check(f"Jumlah kelas terwakili = 9 (terdeteksi: {len(selected_by_class)})",
          len(selected_by_class) == 9)

    # ================================================================
    # 3, 4, 6: Panggilan pertama -- jumlah lapisan, rentang nilai, guard
    # ================================================================
    print(f"\n{'-' * 60}")
    print("3-4, 6. Panggilan compute_entropy_scores() pertama")
    print(f"{'-' * 60}")

    guard_triggered = False
    entropy_scores_1 = {}
    try:
        entropy_scores_1 = compute_entropy_scores(
            model, train_loader, device,
            samples_per_class=samples_per_class, seed=seed
        )
    except RuntimeError as e:
        guard_triggered = True
        print(f"  {RED}RuntimeError tertangkap:{RESET} {e}")

    check("Guard RuntimeError tidak terpicu", not guard_triggered)

    if entropy_scores_1:
        check(f"Jumlah lapisan skor entropi = 16 (terdeteksi: {len(entropy_scores_1)})",
              len(entropy_scores_1) == 16)

        print(f"\n  Rentang nilai entropi per lapisan:")
        for idx in sorted(entropy_scores_1):
            scores = entropy_scores_1[idx]
            print(f"    Layer {idx:2d}: min={scores.min().item():.4f}  "
                  f"max={scores.max().item():.4f}")

    # ================================================================
    # 5: Determinisme -- panggilan kedua harus identik dengan yang pertama
    # ================================================================
    print(f"\n{'-' * 60}")
    print("5. Bukti determinisme (panggilan kedua)")
    print(f"{'-' * 60}")

    if entropy_scores_1:
        entropy_scores_2 = compute_entropy_scores(
            model, train_loader, device,
            samples_per_class=samples_per_class, seed=seed
        )

        check(f"Jumlah lapisan sama antar panggilan "
              f"({len(entropy_scores_1)} vs {len(entropy_scores_2)})",
              set(entropy_scores_1.keys()) == set(entropy_scores_2.keys()))

        max_diff = 0.0
        for idx in entropy_scores_1:
            diff = (entropy_scores_1[idx] - entropy_scores_2[idx]).abs().max().item()
            max_diff = max(max_diff, diff)

        print(f"  Selisih maksimum antar dua panggilan: {max_diff}")
        check(f"Selisih maksimum tepat nol (terdeteksi: {max_diff})",
              max_diff == 0.0)
    else:
        print("  Dilewati karena panggilan pertama gagal.")

    # ================================================================
    # Ringkasan
    # ================================================================
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    if all_passed:
        print(f"  {GREEN}SEMUA VERIFIKASI LULUS{RESET}")
    else:
        print(f"  {RED}ADA VERIFIKASI YANG GAGAL{RESET}")
    print(f"  Waktu: {elapsed:.1f} detik")
    print(f"{'=' * 60}")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
