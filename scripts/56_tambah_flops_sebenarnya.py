"""
56_tambah_flops_sebenarnya.py
Menambahkan field "flops_true_2x_macs" (FLOPs sebenarnya = 2 x MACs) di
SEBELAH field "flops" yang sudah ada -- TANPA mengubah atau menghapus
field "flops" itu sendiri -- pada seluruh outputs/*.json yang punya key
"flops" (yang, per temuan CLAUDE.md Bagian 8 butir 12 dan
scripts/51_hitung_flops.py, sebenarnya berisi MACs dari thop.profile(),
bukan FLOPs).

Kenapa TIDAK menimpa field "flops" langsung: banyak skrip lain (Flask
06_deploy_flask.py via load_efficiency_metrics(), skrip laporan lain)
membaca field "flops" dan mengasumsikan satuannya konsisten dengan
seluruh entri lain di file yang sama (perbandingan persentase antar
entri tetap valid walau labelnya salah, karena semua entri di file yang
sama sama-sama MACs). Mengganti isinya diam-diam berisiko mematahkan
konsumen itu tanpa terdeteksi. Field baru "flops_true_2x_macs" murni
aditif -- pembaca lama tidak terpengaruh, pembaca baru yang butuh FLOPs
sebenarnya tinggal pakai field baru ini.

Berkas yang diproses (satu-satunya di outputs/ yang punya key "flops",
diverifikasi lewat grep sebelum menulis skrip ini):
  - tabel_hasil_lengkap.json
  - ablation_val_results.json
  - multicriteria_per_rasio.json
  - ablation_gm_expansion.json
  - ablation_l1_expansion.json

Skrip ini TIDAK melatih atau menghitung ulang apa pun -- murni transformasi
JSON aditif (walk rekursif, tambah field sibling di tiap dict yang punya
key "flops" bertipe angka).

Jalankan:
    venv/Scripts/python.exe scripts/56_tambah_flops_sebenarnya.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CFG

BERKAS_TARGET = [
    "tabel_hasil_lengkap.json",
    "ablation_val_results.json",
    "multicriteria_per_rasio.json",
    "ablation_gm_expansion.json",
    "ablation_l1_expansion.json",
]

CATATAN_KEY = "_catatan_macs_vs_flops"
CATATAN_TEXT = (
    "Field 'flops' di seluruh berkas ini sebenarnya MACs (multiply-accumulate) "
    "dari thop.profile() via measure_flops() di src/model.py -- thop TIDAK "
    "mengalikan faktor 2 per MAC. Field 'flops_true_2x_macs' (ditambahkan "
    "2026-09-24, scripts/56_tambah_flops_sebenarnya.py) adalah FLOPs "
    "sebenarnya = 2 x nilai 'flops'. Field 'flops' asli TIDAK diubah supaya "
    "konsumen lama (skrip/kode yang sudah membaca field ini) tidak terdampak "
    "-- lihat CLAUDE.md Bagian 8 butir 12."
)


def tambah_flops_sebenarnya(obj):
    """Rekursif: di tiap dict yang punya key 'flops' bertipe angka, tambah
    sibling 'flops_true_2x_macs' = 2 x nilai itu (kalau belum ada). Jalan
    ke seluruh nested dict/list tanpa asumsi struktur spesifik per berkas,
    supaya satu fungsi ini valid untuk kelima berkas yang strukturnya
    berbeda-beda (tabel_hasil_lengkap punya 'results.baseline.flops',
    multicriteria_per_rasio punya 'results.10%.flops', dst)."""
    n_ditambahkan = 0
    if isinstance(obj, dict):
        if "flops" in obj and isinstance(obj["flops"], (int, float)) \
                and "flops_true_2x_macs" not in obj:
            obj["flops_true_2x_macs"] = obj["flops"] * 2
            n_ditambahkan += 1
        for v in obj.values():
            n_ditambahkan += tambah_flops_sebenarnya(v)
    elif isinstance(obj, list):
        for item in obj:
            n_ditambahkan += tambah_flops_sebenarnya(item)
    return n_ditambahkan


def main():
    print("=" * 70)
    print("TAMBAH FIELD FLOPS SEBENARNYA (2 x MACs) -- ADITIF, TIDAK MENGUBAH 'flops'")
    print("=" * 70)

    for nama_berkas in BERKAS_TARGET:
        path = CFG.OUTPUT_DIR / nama_berkas
        if not path.exists():
            print(f"  [LEWATI] {nama_berkas} tidak ditemukan.")
            continue

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        n = tambah_flops_sebenarnya(data)

        if CATATAN_KEY not in data:
            data[CATATAN_KEY] = CATATAN_TEXT

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"  [OK] {nama_berkas}: {n} field 'flops_true_2x_macs' ditambahkan")

    print(f"\n{'=' * 70}")
    print("[SELESAI]")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
