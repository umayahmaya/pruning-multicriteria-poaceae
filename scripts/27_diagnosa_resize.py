"""
27_diagnosa_resize.py
Menyelidiki penyebab pembalikan prediksi Brown_Spot_Rice <-> Leaf_Blast_Rice
di Android (demo_01, demo_03) dengan MENIRU pipeline praproses Android
secara langkah-demi-langkah di Python, lalu membandingkan tensor DAN hasil
prediksinya terhadap praproses PyTorch standar get_transforms("test").

Untuk kedelapan berkas di demo_sidang/:
  1. Cetak dimensi asli (piksel)
  2. Hitung tensor PyTorch standar: get_transforms("test") (resize
     bilinear DENGAN antialias, sekali langsung ke 224x224)
  3. Hitung tensor tiruan Android:
     a. Batasi sisi terpanjang ke SISI_MAKSIMUM (1024) dengan SUBSAMPLE
        tiap n piksel (meniru BitmapFactory.Options.inSampleSize di
        MainActivity.muatBitmapDibatasi) -- n dihitung dengan algoritma
        yang SAMA PERSIS dengan hitungInSampleSize() di Kotlin.
     b. Pengecilan bertahap separuh demi separuh selama hasilnya masih
        di atas dua kali 224 (meniru Klasifikasi.praprosesBertahap).
     c. Resize akhir ke 224x224.
     b dan c memakai bilinear TANPA antialias -- PENTING: torchvision
     hanya menghormati antialias=False pada input Tensor, BUKAN pada
     input PIL Image (selalu antialias untuk PIL, diverifikasi manual
     sebelum menulis skrip ini). Karena itu seluruh tahap resize di sini
     dilakukan pada Tensor uint8, bukan PIL Image.
  4. Bandingkan kedua tensor (selisih maksimum dan rata-rata)
  5. Jalankan model pada KEDUA tensor, cetak tiga kelas teratas masing-masing

Kalau tiruan pipeline Android di sini menghasilkan pembalikan yang sama
(Brown_Spot_Rice <-> Leaf_Blast_Rice) seperti di perangkat, itu membuktikan
penyebabnya adalah kombinasi inSampleSize + resize bertahap tanpa
antialias -- bukan sesuatu yang lain di sisi Android (mis. bug ExecuTorch,
urutan channel, dll).

Skrip ini TIDAK melatih apa pun -- murni muat checkpoint + evaluasi.

Jalankan:
    venv/Scripts/python.exe scripts/27_diagnosa_resize.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from src.config import CFG
from src.dataset import get_transforms
from src.model import load_checkpoint

CHECKPOINT_NAME = "multicriteria_20pct_30ep_valweights.pth"
DEMO_DIR = CFG.ROOT_DIR / "demo_sidang"
IMG_SIZE = 224
SISI_MAKSIMUM = 1024
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def hitung_in_sample_size(lebar, tinggi, sisi_maksimum):
    """Identik dengan hitungInSampleSize() di MainActivity.kt."""
    n = 1
    sisi_terpanjang = max(lebar, tinggi)
    while sisi_terpanjang // 2 >= sisi_maksimum:
        n *= 2
        sisi_terpanjang //= 2
    return n


def praproses_tiruan_android(image_pil):
    """Tiru pipeline Android: inSampleSize -> resize bertahap -> resize
    akhir 224x224 -> normalisasi. Semua resize pada Tensor uint8 (BUKAN
    PIL Image) supaya antialias=False benar-benar berlaku."""
    catatan = {}
    lebar0, tinggi0 = image_pil.size
    catatan["dimensi_asli"] = (lebar0, tinggi0)

    # 1. inSampleSize: subsample tiap n piksel
    n = hitung_in_sample_size(lebar0, tinggi0, SISI_MAKSIMUM)
    catatan["in_sample_size"] = n
    arr = np.array(image_pil)
    if n > 1:
        arr = arr[::n, ::n]
    tensor_u8 = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).contiguous()
    catatan["dimensi_setelah_insamplesize"] = (tensor_u8.shape[2], tensor_u8.shape[1])

    # 2. Pengecilan bertahap separuh demi separuh, bilinear TANPA antialias
    tinggi_now, lebar_now = int(tensor_u8.shape[1]), int(tensor_u8.shape[2])
    tahapan = []
    while lebar_now // 2 >= IMG_SIZE and tinggi_now // 2 >= IMG_SIZE:
        lebar_now //= 2
        tinggi_now //= 2
        tensor_u8 = TF.resize(
            tensor_u8, [tinggi_now, lebar_now],
            interpolation=InterpolationMode.BILINEAR, antialias=False,
        )
        tahapan.append((lebar_now, tinggi_now))
    catatan["tahapan_bertahap"] = tahapan

    # 3. Resize akhir ke 224x224, bilinear TANPA antialias
    tensor_u8 = TF.resize(
        tensor_u8, [IMG_SIZE, IMG_SIZE],
        interpolation=InterpolationMode.BILINEAR, antialias=False,
    )

    # 4. Normalisasi -- identik get_transforms("test")
    tensor_float = tensor_u8.float() / 255.0
    tensor_norm = TF.normalize(tensor_float, mean=MEAN, std=STD)

    return tensor_norm, catatan


def tiga_teratas(model, tensor_chw):
    with torch.no_grad():
        logits = model(tensor_chw.unsqueeze(0))[0]
        probs = torch.nn.functional.softmax(logits, dim=0)
    topk = torch.topk(probs, k=3)
    return [
        (CFG.CLASS_NAMES[idx], prob.item() * 100)
        for prob, idx in zip(topk.values, topk.indices)
    ]


def uraikan_nama_berkas(nama_tanpa_ekstensi):
    import re
    cocok = re.match(r"^demo_(\d+)_(benar|salah)_(.+)$", nama_tanpa_ekstensi)
    if not cocok:
        return "?"
    return cocok.group(3)


def main():
    ckpt_path = CFG.CHECKPOINT_DIR / CHECKPOINT_NAME
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint tidak ditemukan: {ckpt_path}")
        return
    if not DEMO_DIR.exists():
        print(f"[ERROR] {DEMO_DIR} tidak ditemukan.")
        return

    berkas_demo = sorted(
        p for p in DEMO_DIR.iterdir()
        if p.is_file() and p.name.startswith("demo_") and p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    device = torch.device("cpu")
    model, _ = load_checkpoint(ckpt_path, device)
    model.eval()

    transform_pytorch = get_transforms("test")

    print("=" * 70)
    print("DIAGNOSA PRAPROSES: TIRUAN PIPELINE ANDROID vs PYTORCH STANDAR")
    print(f"Checkpoint: {CHECKPOINT_NAME}")
    print(f"SISI_MAKSIMUM (inSampleSize): {SISI_MAKSIMUM}  |  IMG_SIZE: {IMG_SIZE}")
    print("=" * 70)

    n_pembalikan = 0

    for src_path in berkas_demo:
        kelas_asli = uraikan_nama_berkas(src_path.stem)
        image = Image.open(src_path).convert("RGB")

        # --- Tensor PyTorch standar ---
        tensor_pytorch = transform_pytorch(image)

        # --- Tensor tiruan Android ---
        tensor_android, catatan = praproses_tiruan_android(image)

        # --- Bandingkan tensor ---
        selisih = (tensor_pytorch - tensor_android).abs()
        selisih_maks = selisih.max().item()
        selisih_rata = selisih.mean().item()

        # --- Prediksi kedua tensor ---
        top3_pytorch = tiga_teratas(model, tensor_pytorch)
        top3_android = tiga_teratas(model, tensor_android)

        pred_pytorch = top3_pytorch[0][0]
        pred_android = top3_android[0][0]
        pembalikan = pred_pytorch != pred_android
        if pembalikan:
            n_pembalikan += 1

        print(f"\n{'-' * 70}")
        print(f"[{src_path.name}] kelas asli: {kelas_asli}")
        print(f"  1. Dimensi asli            : {catatan['dimensi_asli'][0]}x{catatan['dimensi_asli'][1]}")
        print(f"  1b. inSampleSize (n)       : {catatan['in_sample_size']}  -> "
              f"{catatan['dimensi_setelah_insamplesize'][0]}x{catatan['dimensi_setelah_insamplesize'][1]}")
        if catatan["tahapan_bertahap"]:
            teks_tahapan = " -> ".join(f"{w}x{h}" for w, h in catatan["tahapan_bertahap"])
            print(f"  1c. Tahapan bertahap       : {teks_tahapan} -> {IMG_SIZE}x{IMG_SIZE}")
        else:
            print(f"  1c. Tahapan bertahap       : (tidak ada, langsung ke {IMG_SIZE}x{IMG_SIZE})")

        print(f"  2/3. Selisih tensor        : maksimum={selisih_maks:.6f}  rata-rata={selisih_rata:.6f}")

        teks_pytorch = ", ".join(f"{k} ({p:.2f}%)" for k, p in top3_pytorch)
        teks_android = ", ".join(f"{k} ({p:.2f}%)" for k, p in top3_android)
        print(f"  5. Tiga teratas (PyTorch standar)   : {teks_pytorch}")
        print(f"  5. Tiga teratas (tiruan Android)     : {teks_android}")

        if pembalikan:
            print(f"  >>> PEMBALIKAN KELAS TERATAS: {pred_pytorch} -> {pred_android} <<<")
        else:
            print(f"  Kelas teratas tetap sama: {pred_pytorch}")

    print(f"\n{'=' * 70}")
    print(f"[RINGKASAN] {n_pembalikan}/{len(berkas_demo)} berkas mengalami pembalikan kelas "
          f"teratas antara praproses PyTorch standar dan tiruan Android.")
    if n_pembalikan > 0:
        print("[KESIMPULAN] Tiruan pipeline Android di Python TERBUKTI bisa menghasilkan")
        print("pembalikan kelas -- penyebabnya konsisten dengan kombinasi inSampleSize +")
        print("resize bertahap tanpa antialias, BUKAN faktor lain di sisi Android (mis.")
        print("ExecuTorch, urutan channel). Ini bisa diperbaiki/diuji lebih lanjut di")
        print("Python tanpa perlu menebak-nebak langsung di perangkat.")
    else:
        print("[KESIMPULAN] Tiruan pipeline Android di sini TIDAK mereproduksi pembalikan")
        print("yang diamati di Android -- penyebabnya kemungkinan bukan (hanya) resize,")
        print("perlu diselidiki faktor lain (mis. detail implementasi ExecuTorch/JNI,")
        print("perbedaan decoder gambar BitmapFactory vs PIL, dll).")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()