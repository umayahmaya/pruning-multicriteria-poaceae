"""
26_banding_demo_sidang.py
Menjalankan kedelapan berkas di demo_sidang/ (hasil
25_siapkan_demo_proporsional.py) langsung dari lokasinya (BUKAN dari
dataset_split/test/) lewat model PyTorch memakai praproses evaluasi
standar get_transforms("test"), mencetak tiga kelas teratas beserta
persentasenya untuk tiap berkas.

Latar belakang: ditemukan ketidakcocokan antara prediksi PyTorch
(25_siapkan_demo_proporsional.py, mengevaluasi dari dataset_split/test/)
dan prediksi aplikasi Android untuk demo_01 (Brown_Spot_Rice) dan demo_03
(Leaf_Blast_Rice) -- keduanya BENAR menurut skrip 25, tapi SALAH dan
tertukar satu sama lain menurut Android. Skrip ini menjalankan ulang
PyTorch pada BERKAS DI demo_sidang/ ITU SENDIRI (hasil copy2 dari
dataset_split/test/, seharusnya identik byte-per-byte) supaya:
  1. Memastikan proses penyalinan ke demo_sidang/ tidak mengubah isi
     berkas (copy2 seharusnya lossless, tapi diverifikasi di sini, bukan
     diasumsikan).
  2. Punya angka PyTorch yang sumbernya PERSIS sama dengan berkas yang
     dipakai Android (disalin dari demo_sidang/ ke /sdcard/Pictures/),
     menghilangkan kemungkinan "berkas beda" sebagai penyebab.
Kalau prediksi PyTorch di sini SAMA dengan skrip 25 (Brown_Spot_Rice dan
Leaf_Blast_Rice tetap benar), ketidakcocokan dengan Android murni berasal
dari perbedaan praproses (resize) atau sisi Android -- bukan dari berkas
sumber yang berbeda.

Skrip ini TIDAK melatih apa pun -- murni muat checkpoint + evaluasi.

Jalankan:
    venv/Scripts/python.exe scripts/26_banding_demo_sidang.py
"""

import sys
import os
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image

from src.config import CFG
from src.dataset import get_transforms
from src.model import load_checkpoint

CHECKPOINT_NAME = "multicriteria_per_rasio_20pct_30ep.pth"  # formula AKTIF, harus sama dengan skrip 25 -- lihat CLAUDE.md Bagian 8 butir 11
DEMO_DIR = CFG.ROOT_DIR / "demo_sidang"

POLA_NAMA = re.compile(r"^demo_(\d+)_(benar|salah)_(.+)$")


def uraikan_nama_berkas(nama_tanpa_ekstensi):
    """Uraikan demo_NN_status_KelasAsli -> (nomor, status, kelas_asli)."""
    cocok = POLA_NAMA.match(nama_tanpa_ekstensi)
    if not cocok:
        return None, None, None
    return cocok.group(1), cocok.group(2), cocok.group(3)


def main():
    ckpt_path = CFG.CHECKPOINT_DIR / CHECKPOINT_NAME
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint tidak ditemukan: {ckpt_path}")
        return

    if not DEMO_DIR.exists():
        print(f"[ERROR] {DEMO_DIR} tidak ditemukan.")
        print("Jalankan 25_siapkan_demo_proporsional.py terlebih dahulu.")
        return

    berkas_demo = sorted(
        p for p in DEMO_DIR.iterdir()
        if p.is_file() and p.name.startswith("demo_") and p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not berkas_demo:
        print(f"[ERROR] Tidak ada berkas demo_* di {DEMO_DIR}")
        return

    device = torch.device("cpu")
    model, _ = load_checkpoint(ckpt_path, device)
    model.eval()

    transform = get_transforms("test")

    print("=" * 70)
    print("UJI BANDING: PREDIKSI PYTORCH LANGSUNG DARI demo_sidang/")
    print(f"Checkpoint: {CHECKPOINT_NAME}")
    print("=" * 70)

    n_cocok_dengan_status = 0
    n_total = 0

    with torch.no_grad():
        for src_path in berkas_demo:
            nomor, status, kelas_asli = uraikan_nama_berkas(src_path.stem)
            if kelas_asli is None:
                print(f"\n[PERINGATAN] Nama berkas tidak sesuai pola demo_NN_status_Kelas: {src_path.name}")
                kelas_asli = "?"
                status = "?"

            image = Image.open(src_path).convert("RGB")
            tensor = transform(image).unsqueeze(0)

            logits = model(tensor)[0]
            probs = torch.nn.functional.softmax(logits, dim=0)

            tiga_teratas = torch.topk(probs, k=3)
            teks_tiga_teratas = ", ".join(
                f"{CFG.CLASS_NAMES[idx]} ({prob.item() * 100:.2f}%)"
                for prob, idx in zip(tiga_teratas.values, tiga_teratas.indices)
            )

            kelas_prediksi = CFG.CLASS_NAMES[tiga_teratas.indices[0].item()]
            benar_sekarang = kelas_prediksi == kelas_asli
            status_sekarang = "benar" if benar_sekarang else "salah"
            cocok_dengan_skrip25 = (status_sekarang == status)

            n_total += 1
            if cocok_dengan_skrip25:
                n_cocok_dengan_status += 1

            tanda = "" if cocok_dengan_skrip25 else "  <-- BEDA DARI SKRIP 25!"

            print(f"\nBerkas={src_path.name} (kelas asli: {kelas_asli}, "
                  f"status skrip 25: {status.upper()}){tanda}")
            print(f"  Tiga teratas (PyTorch, dari demo_sidang/): {teks_tiga_teratas}")
            print(f"  Status sekarang: {status_sekarang.upper()}")

    print(f"\n{'=' * 70}")
    print(f"[RINGKASAN] {n_cocok_dengan_status}/{n_total} status BENAR/SALAH "
          f"cocok dengan hasil 25_siapkan_demo_proporsional.py")
    if n_cocok_dengan_status == n_total:
        print("[KESIMPULAN] Berkas di demo_sidang/ menghasilkan prediksi PyTorch")
        print("YANG SAMA dengan skrip 25 -- penyalinan berkas tidak mengubah isi.")
        print("Ketidakcocokan dengan Android BUKAN berasal dari berkas sumber yang")
        print("berbeda, kemungkinan besar dari perbedaan praproses (resize) atau")
        print("sisi lain pipeline Android.")
    else:
        print("[KESIMPULAN] Ditemukan perbedaan antara prediksi di sini dan skrip 25")
        print("meskipun sumbernya seharusnya identik -- perlu investigasi lebih lanjut")
        print("(mis. apakah shutil.copy2 benar-benar lossless, atau ada perbedaan lain).")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()