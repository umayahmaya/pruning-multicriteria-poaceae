"""
28_audit_exif.py
Memindai seluruh berkas gambar di dataset_split/{train,val,test} untuk
memeriksa sebaran tag EXIF Orientation -- menindaklanjuti temuan
27_diagnosa_resize.py bahwa demo_01/demo_03 (Orientation=6) diputar oleh
aplikasi Android (MainActivity.putarBitmapSesuaiOrientasi) tapi TIDAK
oleh pipeline Python manapun (training, validasi, testing selalu memakai
piksel mentah dari PIL.Image.open(), tanpa koreksi EXIF -- dikonfirmasi
lewat grep, nol referensi EXIF di seluruh src/ dan scripts/).

Melaporkan (TIDAK mengubah berkas apa pun -- murni pemindaian baca-saja):
1. Jumlah gambar yang punya tag Orientation vs yang tidak.
2. Sebaran nilai orientasi (1, 6, 8, dst.).
3. Rincian per split dan per kelas: berapa gambar yang BUTUH ROTASI --
   didefinisikan PERSIS seperti cabang putarBitmapSesuaiOrientasi() di
   MainActivity.kt: nilai 2, 3, 4, 5, 6, 7, 8 memicu transformasi;
   nilai 1 (normal), 0 (undefined), nilai lain yang tidak dikenal, atau
   tag yang hilang sama-sama TIDAK memicu rotasi (jatuh ke cabang
   `else -> return bitmap` di Kotlin).
4. Persentase gambar terpengaruh terhadap total keseluruhan.

Jalankan:
    venv/Scripts/python.exe scripts/28_audit_exif.py
"""

import sys
import os
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image

from src.config import CFG

SPLITS = ["train", "val", "test"]
EKSTENSI_VALID = (".jpg", ".jpeg", ".png")

# Nilai EXIF Orientation yang memicu transformasi di
# MainActivity.putarBitmapSesuaiOrientasi() -- selain ini (termasuk 1, 0,
# nilai tak dikenal, atau tag hilang) jatuh ke cabang else (tanpa rotasi).
NILAI_MEMICU_ROTASI = {2, 3, 4, 5, 6, 7, 8}

TAG_ORIENTATION = 0x0112  # 274, standar EXIF


def baca_orientasi(path):
    """Kembalikan nilai tag Orientation (int), None kalau tidak ada tag,
    atau "ERROR" kalau berkas gagal dibaca sama sekali."""
    try:
        with Image.open(path) as img:
            exif = img.getexif()
            return exif.get(TAG_ORIENTATION)
    except Exception:
        return "ERROR"


def main():
    split_dir_root = CFG.DATASET_DIR.parent / "dataset_split"

    total_gambar = 0
    total_ada_tag = 0
    total_tanpa_tag = 0
    total_butuh_rotasi = 0
    total_error = 0

    sebaran_orientasi = defaultdict(int)
    rincian_split_kelas = defaultdict(lambda: {"total": 0, "butuh_rotasi": 0})

    print("=" * 70)
    print("AUDIT EXIF ORIENTATION: dataset_split/{train,val,test}")
    print("=" * 70)

    for split in SPLITS:
        split_dir = split_dir_root / split
        if not split_dir.exists():
            print(f"[PERINGATAN] {split_dir} tidak ditemukan, dilewati.")
            continue

        kelas_list = sorted(p.name for p in split_dir.iterdir() if p.is_dir())
        for kelas in kelas_list:
            kelas_dir = split_dir / kelas
            berkas_list = sorted(
                p for p in kelas_dir.iterdir()
                if p.is_file() and p.suffix.lower() in EKSTENSI_VALID
            )

            kunci = (split, kelas)
            for path in berkas_list:
                total_gambar += 1
                rincian_split_kelas[kunci]["total"] += 1

                nilai = baca_orientasi(path)

                if nilai == "ERROR":
                    total_error += 1
                    sebaran_orientasi["ERROR (gagal dibaca)"] += 1
                    continue

                if nilai is None:
                    total_tanpa_tag += 1
                    sebaran_orientasi["Tidak ada tag"] += 1
                    continue

                total_ada_tag += 1
                sebaran_orientasi[nilai] += 1

                if nilai in NILAI_MEMICU_ROTASI:
                    total_butuh_rotasi += 1
                    rincian_split_kelas[kunci]["butuh_rotasi"] += 1

    # ----- 1. Punya tag vs tidak -----
    print(f"\nTotal gambar dipindai: {total_gambar}")
    print(f"  Punya tag Orientation : {total_ada_tag}")
    print(f"  Tidak punya tag       : {total_tanpa_tag}")
    if total_error:
        print(f"  Gagal dibaca (error)  : {total_error}")

    # ----- 2. Sebaran nilai orientasi -----
    print(f"\n{'=' * 70}")
    print("SEBARAN NILAI ORIENTASI")
    print("=" * 70)

    def kunci_urut(item):
        k, _ = item
        if isinstance(k, int):
            return (0, k)
        return (1, str(k))

    for nilai, jumlah in sorted(sebaran_orientasi.items(), key=kunci_urut):
        keterangan = ""
        if nilai == 1:
            keterangan = " (normal, tidak perlu rotasi)"
        elif nilai == 0:
            keterangan = " (undefined -- TIDAK memicu rotasi di Android, lihat catatan berkas)"
        elif isinstance(nilai, int) and nilai in NILAI_MEMICU_ROTASI:
            keterangan = " (MEMICU ROTASI di Android)"
        print(f"  {nilai!s:<28}: {jumlah:>4} gambar{keterangan}")

    # ----- 3. Rincian per split dan per kelas -----
    print(f"\n{'=' * 70}")
    print("RINCIAN PER SPLIT DAN PER KELAS (butuh rotasi = nilai 2,3,4,5,6,7,8)")
    print("=" * 70)
    print(f"  {'Split':<6} {'Kelas':<22} {'Total':>6} {'Butuh Rotasi':>13} {'Persentase':>11}")
    for split in SPLITS:
        for kelas in CFG.CLASS_NAMES:
            kunci = (split, kelas)
            if kunci not in rincian_split_kelas:
                continue
            data = rincian_split_kelas[kunci]
            persen = data["butuh_rotasi"] / data["total"] * 100 if data["total"] else 0.0
            tanda = "  <--" if data["butuh_rotasi"] > 0 else ""
            print(f"  {split:<6} {kelas:<22} {data['total']:>6} {data['butuh_rotasi']:>13} "
                  f"{persen:>10.2f}%{tanda}")

    # ----- 4. Persentase total -----
    print(f"\n{'=' * 70}")
    print("RINGKASAN")
    print("=" * 70)
    persen_total = total_butuh_rotasi / total_gambar * 100 if total_gambar else 0.0
    print(f"  Total gambar (seluruh split)                : {total_gambar}")
    print(f"  Butuh rotasi (nilai memicu transformasi)     : {total_butuh_rotasi}")
    print(f"  Persentase terpengaruh dari total keseluruhan: {persen_total:.2f}%")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()