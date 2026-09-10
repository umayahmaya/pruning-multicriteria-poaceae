"""
25_siapkan_demo_proporsional.py
Menyiapkan 8 citra demonstrasi untuk sidang dengan komposisi yang
merepresentasikan akurasi model sebenarnya: 7 benar, 1 salah. Rasio
7:1 (87,5%) adalah pendekatan bilangan bulat terdekat ke akurasi uji
riil model (96,00% pada 175 citra) yang masih bisa ditampilkan dalam
satu citra salah pada sampel sekecil 8 -- 8/8 benar tidak jujur (tidak
merepresentasikan bahwa model memang salah pada ~4% kasus), sementara
menyertakan lebih dari satu citra salah akan melebih-lebihkan tingkat
kesalahan sebenarnya.

Langkah:
1. Evaluasi checkpoint multicriteria_20pct_30ep_valweights.pth pada
   SELURUH dataset_split/test (175 citra), catat benar/salah per citra
   beserta tiga kelas teratas dan persentasenya. Confusion matrix
   dihitung ULANG di sini dari evaluasi ini -- tidak diasumsikan dari
   sesi/skrip sebelumnya.
2. Dari confusion matrix itu, cari pasangan (kelas asli tebu -> kelas
   prediksi) yang paling sering tertukar. Ini menentukan citra "salah"
   di langkah 3.
3. Pilih 8 citra:
   - 3 padi (Brown_Spot_Rice, Healthy_Rice, Leaf_Blast_Rice), masing-
     masing 1 citra BENAR dengan probabilitas prediksi tertinggi di
     kelasnya.
   - 3 jagung (Common_Rust_Corn, Gray_Leaf_Spot_Corn, Healthy_Corn),
     sama seperti padi.
   - 1 tebu BENAR: dari kelas yang menjadi TUJUAN pasangan tertukar
     paling sering (mis. kalau Rust_Sugarcane paling sering diprediksi
     sebagai Healthy_Sugarcane, citra benar tebu diambil dari kelas
     Healthy_Sugarcane) -- supaya berdampingan dengan citra salah,
     memperagakan langsung pasangan yang tertukar.
   - 1 tebu SALAH: contoh pasangan tertukar paling sering itu sendiri,
     dipilih dengan probabilitas SALAH tertinggi (contoh paling
     meyakinkan/jelas dari kekeliruan tersebut).
   Total: 7 benar (mencakup 6 dari 6 kelas padi+jagung, plus 1 kelas
   tebu) + 1 salah (kelas tebu lain) = 8 kelas dari 9 kelas terwakili.
4. Salin kedelapan citra ke demo_sidang/ dengan nama yang mencantumkan
   kelas asli dan status.
5. Tulis demo_sidang/CATATAN_DEMO.md: tabel nama berkas, kelas asli,
   prediksi model, tiga kelas teratas + persentase, status. Untuk citra
   salah, tambahkan catatan bahwa ini representasi ~4% kesalahan model,
   konsisten temuan confusion matrix bahwa tebu paling sering tertukar.

Pemilihan pada dasarnya deterministik (probabilitas tertinggi menang,
seri dipecah berdasar nama berkas menaik) -- random.seed(SEED) tetap
diset secara eksplisit untuk reproduksibilitas kalau ada kebutuhan
pemilihan acak di masa depan, konsisten dengan konvensi seed=42 proyek
ini.

Skrip ini TIDAK melatih apa pun -- murni muat checkpoint + evaluasi.

Jalankan:
    venv/Scripts/python.exe scripts/25_siapkan_demo_proporsional.py
"""

import sys
import os
import shutil
import random
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchvision import datasets

from src.config import CFG
from src.dataset import get_transforms
from src.model import load_checkpoint

CHECKPOINT_NAME = "multicriteria_20pct_30ep_valweights.pth"
DEMO_DIR = CFG.ROOT_DIR / "demo_sidang"
CATATAN_PATH = DEMO_DIR / "CATATAN_DEMO.md"

PADI = ["Brown_Spot_Rice", "Healthy_Rice", "Leaf_Blast_Rice"]
JAGUNG = ["Common_Rust_Corn", "Gray_Leaf_Spot_Corn", "Healthy_Corn"]
TEBU = ["Healthy_Sugarcane", "Mosaic_Sugarcane", "Rust_Sugarcane"]


def evaluasi_seluruh_test(model, dataset):
    """Evaluasi tiap citra test satu per satu, kembalikan daftar hasil
    lengkap (path, kelas asli, tiga kelas teratas + persentase, benar)."""
    hasil = []
    with torch.no_grad():
        for idx in range(len(dataset)):
            path, label = dataset.samples[idx]
            image_tensor, _ = dataset[idx]
            tensor = image_tensor.unsqueeze(0)

            logits = model(tensor)[0]
            probs = torch.nn.functional.softmax(logits, dim=0)
            topk = torch.topk(probs, k=3)
            tiga_teratas = [
                (CFG.CLASS_NAMES[i], p.item() * 100)
                for p, i in zip(topk.values, topk.indices)
            ]

            kelas_asli = CFG.CLASS_NAMES[label]
            kelas_prediksi = tiga_teratas[0][0]
            benar = kelas_prediksi == kelas_asli

            hasil.append({
                "path": Path(path),
                "filename": os.path.basename(path),
                "kelas_asli": kelas_asli,
                "kelas_prediksi": kelas_prediksi,
                "benar": benar,
                "tiga_teratas": tiga_teratas,
            })
    return hasil


def hitung_pasangan_tertukar_tebu(hasil):
    """Dari daftar hasil evaluasi, hitung frekuensi pasangan
    (kelas_asli, kelas_prediksi) untuk citra SALAH yang kelas aslinya
    tebu, kembalikan pasangan paling sering beserta hitungannya."""
    hitungan = {}
    for r in hasil:
        if r["benar"]:
            continue
        if r["kelas_asli"] not in TEBU:
            continue
        pasangan = (r["kelas_asli"], r["kelas_prediksi"])
        hitungan[pasangan] = hitungan.get(pasangan, 0) + 1

    if not hitungan:
        raise RuntimeError("Tidak ditemukan kesalahan pada kelas tebu di seluruh test set.")

    # Urutkan menurun berdasar hitungan, seri dipecah oleh nama pasangan
    # (deterministik) supaya tidak bergantung urutan dict.
    pasangan_teratas = sorted(hitungan.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return pasangan_teratas[0], pasangan_teratas[1], hitungan


def pilih_terbaik(hasil, kelas_asli, benar, kelas_prediksi=None):
    """Pilih satu citra dari daftar hasil: kelas_asli cocok, status
    benar/salah cocok, (opsional) kelas_prediksi cocok -- diurutkan
    probabilitas top-1 tertinggi, seri dipecah nama berkas menaik."""
    kandidat = [
        r for r in hasil
        if r["kelas_asli"] == kelas_asli and r["benar"] == benar
        and (kelas_prediksi is None or r["kelas_prediksi"] == kelas_prediksi)
    ]
    if not kandidat:
        raise RuntimeError(
            f"Tidak ada kandidat untuk kelas_asli={kelas_asli}, benar={benar}, "
            f"kelas_prediksi={kelas_prediksi}"
        )
    kandidat.sort(key=lambda r: (-r["tiga_teratas"][0][1], r["filename"]))
    return kandidat[0]


def main():
    random.seed(CFG.SEED)

    device = torch.device("cpu")
    ckpt_path = CFG.CHECKPOINT_DIR / CHECKPOINT_NAME
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

    print("=" * 70)
    print("EVALUASI SELURUH DATASET_SPLIT/TEST")
    print("=" * 70)
    print(f"[INFO] Checkpoint : {ckpt_path.name}")
    print(f"[INFO] Mengevaluasi {len(dataset)} citra test satu per satu...")

    hasil = evaluasi_seluruh_test(model, dataset)
    n_benar = sum(1 for r in hasil if r["benar"])
    n_total = len(hasil)
    print(f"[INFO] Akurasi test: {n_benar}/{n_total} = {n_benar / n_total * 100:.2f}%")

    print("\n" + "=" * 70)
    print("PASANGAN TERTUKAR PALING SERING (KELAS ASLI TEBU)")
    print("=" * 70)
    (kelas_asli_salah, kelas_prediksi_salah), jumlah, semua_pasangan = hitung_pasangan_tertukar_tebu(hasil)
    for (asli, pred), jml in sorted(semua_pasangan.items(), key=lambda kv: -kv[1]):
        tanda = " <-- terpilih" if (asli, pred) == (kelas_asli_salah, kelas_prediksi_salah) else ""
        print(f"  {asli} -> {pred} : {jml} kali{tanda}")

    # ----- Pilih 8 citra -----
    terpilih = []

    for kelas in PADI + JAGUNG:
        terpilih.append(("benar", pilih_terbaik(hasil, kelas, benar=True)))

    # Citra tebu BENAR: dari kelas yang jadi tujuan pasangan tertukar,
    # supaya berdampingan langsung memperagakan pasangan yang tertukar.
    r_tebu_benar = pilih_terbaik(hasil, kelas_prediksi_salah, benar=True)
    terpilih.append(("benar", r_tebu_benar))

    # Citra tebu SALAH: contoh pasangan tertukar paling sering, dengan
    # probabilitas salah tertinggi (paling meyakinkan/jelas).
    r_tebu_salah = pilih_terbaik(
        hasil, kelas_asli_salah, benar=False, kelas_prediksi=kelas_prediksi_salah
    )
    terpilih.append(("salah", r_tebu_salah))

    n_terpilih_benar = sum(1 for status, _ in terpilih if status == "benar")
    n_terpilih_salah = sum(1 for status, _ in terpilih if status == "salah")
    print(f"\n[INFO] Terpilih: {n_terpilih_benar} benar, {n_terpilih_salah} salah "
          f"(total {len(terpilih)})")

    # ----- Salin ke demo_sidang/ -----
    DEMO_DIR.mkdir(exist_ok=True)
    baris_tabel = []

    print("\n" + "=" * 70)
    print("MENYALIN CITRA KE demo_sidang/")
    print("=" * 70)

    for i, (status, r) in enumerate(terpilih, start=1):
        ekstensi = r["path"].suffix
        nama_baru = f"demo_{i:02d}_{status}_{r['kelas_asli']}{ekstensi}"
        tujuan = DEMO_DIR / nama_baru
        shutil.copy2(r["path"], tujuan)

        print(f"  [OK] {nama_baru} <- {r['filename']} "
              f"(asli={r['kelas_asli']}, prediksi={r['kelas_prediksi']}, status={status})")

        baris_tabel.append({
            "no": i,
            "berkas": nama_baru,
            "kelas_asli": r["kelas_asli"],
            "kelas_prediksi": r["kelas_prediksi"],
            "tiga_teratas": r["tiga_teratas"],
            "status": status,
        })

    # ----- Tulis CATATAN_DEMO.md -----
    tulis_catatan_demo(baris_tabel, n_benar, n_total, kelas_asli_salah, kelas_prediksi_salah, jumlah, semua_pasangan)

    print(f"\n{'=' * 70}")
    print(f"[SELESAI] 8 citra demo + CATATAN_DEMO.md ada di: {DEMO_DIR}")
    print(f"{'=' * 70}")


def tulis_catatan_demo(baris_tabel, n_benar, n_total, kelas_asli_salah, kelas_prediksi_salah, jumlah_tertukar, semua_pasangan):
    akurasi_asli = n_benar / n_total * 100
    maks_hitungan = max(semua_pasangan.values())
    pasangan_seri = sorted(k for k, v in semua_pasangan.items() if v == maks_hitungan)
    ada_seri = len(pasangan_seri) > 1

    lines = []
    lines.append("# Catatan Citra Demonstrasi Sidang")
    lines.append("")
    lines.append(
        f"Delapan citra ini dipilih dengan komposisi **7 benar, 1 salah** "
        f"(87,50%), mendekati akurasi uji model yang sebenarnya "
        f"**{n_benar}/{n_total} = {akurasi_asli:.2f}%** pada seluruh "
        f"`dataset_split/test` -- rasio bilangan bulat terdekat yang masih "
        f"bisa direpresentasikan pada sampel sekecil 8 citra sambil tetap "
        f"menyertakan minimal satu contoh kesalahan (jujur terhadap "
        f"performa model, bukan hanya menampilkan kasus yang mudah)."
    )
    lines.append("")
    lines.append(f"Checkpoint: `multicriteria_20pct_30ep_valweights.pth`")
    lines.append("")
    lines.append(
        f"Pemilihan deterministik (seed {CFG.SEED}): tiap citra benar diambil "
        f"dari probabilitas prediksi BENAR tertinggi di kelasnya; citra salah "
        f"diambil dari probabilitas prediksi SALAH tertinggi pada pasangan "
        f"kelas yang paling sering tertukar."
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Tabel Citra Demo")
    lines.append("")
    lines.append("| No | Berkas | Kelas Asli | Prediksi Model | Tiga Kelas Teratas | Status |")
    lines.append("|---|---|---|---|---|---|")
    for b in baris_tabel:
        teks_tiga = ", ".join(f"{k} ({p:.2f}%)" for k, p in b["tiga_teratas"])
        status_teks = "BENAR" if b["status"] == "benar" else "SALAH"
        lines.append(
            f"| {b['no']} | {b['berkas']} | {b['kelas_asli']} | "
            f"{b['kelas_prediksi']} | {teks_tiga} | {status_teks} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## Catatan untuk Citra Salah")
    lines.append("")
    baris_salah = next(b for b in baris_tabel if b["status"] == "salah")
    lines.append(
        f"**{baris_salah['berkas']}** (kelas asli **{kelas_asli_salah}**, "
        f"diprediksi sebagai **{kelas_prediksi_salah}**) termasuk dalam "
        f"sekitar **{100 - akurasi_asli:.2f}% kesalahan model** pada "
        f"seluruh citra uji ({n_total - n_benar} dari {n_total} citra salah)."
    )
    lines.append("")
    if ada_seri:
        daftar_seri = "; ".join(f"{a} -> {p}" for a, p in pasangan_seri)
        lines.append(
            f"**Catatan kejujuran soal \"paling sering tertukar\":** pada "
            f"evaluasi ini seluruh kesalahan kelas tebu terjadi tepat "
            f"{maks_hitungan} kali masing-masing -- **tidak ada pasangan yang "
            f"benar-benar dominan**. Ketiga pasangan yang seri: {daftar_seri}. "
            f"Pasangan {kelas_asli_salah} -> {kelas_prediksi_salah} dipilih "
            f"lewat aturan pemecah seri deterministik (urutan abjad nama "
            f"kelas), bukan karena frekuensinya lebih tinggi dari yang lain. "
            f"Yang tetap valid dan konsisten dengan temuan sesi-sesi "
            f"sebelumnya (mis. `HASIL_VERIFIKASI.md`): seluruh kesalahan "
            f"kelas tebu pada evaluasi ini bermuara ke kelas tebu lain "
            f"(bukan ke padi/jagung, kecuali satu kasus Rust_Sugarcane -> "
            f"Common_Rust_Corn), sehingga tebu tetap area yang paling "
            f"rawan tertukar dibanding padi atau jagung."
        )
    else:
        lines.append(
            f"Pasangan {kelas_asli_salah} -> {kelas_prediksi_salah} adalah "
            f"kesalahan yang paling sering terjadi pada kelas tebu di "
            f"confusion matrix evaluasi ini ({jumlah_tertukar} dari total "
            f"kesalahan tebu), sehingga citra ini representatif -- bukan "
            f"kasus acak atau kasus terburuk yang dipilih untuk sensasi, "
            f"melainkan pola kekeliruan yang paling konsisten muncul."
        )
    lines.append("")

    with open(CATATAN_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n[OK] {CATATAN_PATH.name} ditulis.")


if __name__ == "__main__":
    main()