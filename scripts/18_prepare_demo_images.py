"""
18_prepare_demo_images.py
Menjalankan checkpoint default deployment (multicriteria_20pct_30ep_valweights.pth)
pada seluruh 175 citra subset test satu per satu, lalu menyusun daftar citra
demonstrasi: untuk masing-masing dari sembilan kelas, tiga citra dengan
probabilitas prediksi BENAR tertinggi.

Juga mencatat daftar LENGKAP citra yang salah diprediksi (nama berkas, kelas
asli, kelas prediksi, probabilitas) supaya diketahui citra mana yang harus
dihindari saat demonstrasi.

Skrip ini TIDAK melatih apa pun -- murni evaluasi per-citra memakai transform
evaluasi yang sama dengan endpoint /predict di 06_deploy_flask.py.

Jalankan:
    venv/Scripts/python.exe scripts/18_prepare_demo_images.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchvision import datasets

from src.config import CFG
from src.dataset import get_transforms
from src.model import load_checkpoint

DEFAULT_CHECKPOINT = "multicriteria_20pct_30ep_valweights.pth"
RESULTS_PATH = CFG.OUTPUT_DIR / "citra_demo.json"
TOP_N_PER_CLASS = 3


def main():
    device = torch.device("cpu")

    ckpt_path = CFG.CHECKPOINT_DIR / DEFAULT_CHECKPOINT
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint tidak ditemukan: {ckpt_path}")
        return

    model, _ = load_checkpoint(ckpt_path, device)
    model.eval()

    test_dir = CFG.DATASET_DIR.parent / "dataset_split" / "test"
    transform = get_transforms("test")
    dataset = datasets.ImageFolder(test_dir, transform=transform)

    if dataset.classes != CFG.CLASS_NAMES:
        print("[ERROR] Urutan kelas ImageFolder tidak cocok dengan CFG.CLASS_NAMES!")
        print(f"  ImageFolder : {dataset.classes}")
        print(f"  CFG         : {CFG.CLASS_NAMES}")
        return

    print(f"[INFO] Checkpoint : {ckpt_path.name}")
    print(f"[INFO] Mengevaluasi {len(dataset)} citra test satu per satu...")

    per_class_correct = {cls: [] for cls in CFG.CLASS_NAMES}
    misclassified = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            path, label = dataset.samples[idx]
            image_tensor, _ = dataset[idx]
            image_tensor = image_tensor.unsqueeze(0)

            outputs = model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
            pred_idx = int(torch.argmax(probs).item())

            filename = os.path.basename(path)
            true_class = CFG.CLASS_NAMES[label]
            pred_class = CFG.CLASS_NAMES[pred_idx]
            record = {
                "filename": filename,
                "true_class": true_class,
                "predicted_class": pred_class,
                "probability": round(probs[pred_idx].item() * 100, 2),
            }

            if pred_idx == label:
                per_class_correct[true_class].append(record)
            else:
                misclassified.append(record)

    # Ambil TOP_N_PER_CLASS teratas per kelas berdasar probabilitas prediksi benar
    demo_images = {}
    for cls in CFG.CLASS_NAMES:
        sorted_correct = sorted(per_class_correct[cls], key=lambda x: x["probability"], reverse=True)
        demo_images[cls] = sorted_correct[:TOP_N_PER_CLASS]
        if len(sorted_correct) < TOP_N_PER_CLASS:
            print(f"[PERINGATAN] Kelas {cls} hanya punya {len(sorted_correct)} citra "
                  f"benar (diharapkan minimal {TOP_N_PER_CLASS}).")

    total_correct = sum(len(v) for v in per_class_correct.values())

    save_data = {
        "checkpoint": DEFAULT_CHECKPOINT,
        "total_test_images": len(dataset),
        "total_correct": total_correct,
        "total_misclassified": len(misclassified),
        "accuracy": round(total_correct / len(dataset) * 100, 2),
        "demo_images_top3_per_class": demo_images,
        "misclassified_images": misclassified,
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    # Cetak ringkasan
    print(f"\n{'=' * 70}")
    print(f"CITRA DEMONSTRASI ({TOP_N_PER_CLASS} teratas per kelas, probabilitas prediksi benar)")
    print(f"{'=' * 70}")
    for cls in CFG.CLASS_NAMES:
        print(f"\n{cls}:")
        for item in demo_images[cls]:
            print(f"  {item['filename']:<35} {item['probability']:.2f}%")

    print(f"\n{'=' * 70}")
    print(f"CITRA SALAH PREDIKSI ({len(misclassified)} dari {len(dataset)})")
    print(f"{'=' * 70}")
    for item in misclassified:
        print(f"  {item['filename']:<35} asli={item['true_class']:<22} "
              f"prediksi={item['predicted_class']:<22} ({item['probability']:.2f}%)")

    target_file = "rust (131).jpeg"
    is_misclassified = any(item["filename"] == target_file for item in misclassified)
    print(f"\n[KONFIRMASI] '{target_file}' termasuk citra salah prediksi? "
          f"{'YA' if is_misclassified else 'TIDAK'}")

    print(f"\n[INFO] Akurasi test: {save_data['accuracy']:.2f}% "
          f"({total_correct}/{len(dataset)})")
    print(f"[INFO] Hasil disimpan: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
