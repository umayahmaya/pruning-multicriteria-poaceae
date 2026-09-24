"""
51_hitung_flops.py
Tabel beban komputasi (MACs dan FLOPs) untuk baseline dan ketujuh rasio
formula AKTIF (L1 + GM-Ekspansi + Entropi, bobot per rasio), melengkapi
outputs/tabel_parameter_per_blok.csv (skrip 49).

CATATAN PENOMORAN: nomor "50" yang diminta di permintaan tugas sudah
dipakai (50_visualisasi_peta_fitur.py, sesi sebelumnya) -- skrip ini
diberi nomor 51 (nomor kosong berikutnya) supaya tidak menimpa apa pun.

REKONSTRUKSI MODEL: pola IDENTIK dengan scripts/49_tabel_parameter_per_blok.py
-- skor L1/GM-Ekspansi/Entropi dihitung sekali dari baseline (fungsi
compute_l1_scores/compute_entropy_scores/normalize_min_max/
compute_importance_scores/get_pruning_mask/apply_pruning diimpor APA
ADANYA dari src/pruning.py, GM-Ekspansi diimpor dinamis dari
scripts/43_ablation_gm_expansion.py -- src/pruning.py TIDAK diubah),
bobot w_k(r) dibaca APA ADANYA dari outputs/multicriteria_per_rasio.json.
TIDAK ADA estimasi -- setiap model benar-benar direkonstruksi lewat
apply_pruning() dengan mask sungguhan sebelum diprofil.

UKURAN INPUT: diambil dari CFG.IMG_SIZE (src/config.py), DICETAK
eksplisit di output -- TIDAK diasumsikan 224x224 (kebetulan memang 224
di proyek ini, tapi dikonfirmasi dari konfigurasi, bukan ditulis
hardcode di sini).

LIBRARY: thop 0.1.1 (SUDAH terpasang di venv/, dipakai measure_flops()
di src/model.py -- LIBRARY YANG SAMA dipakai supaya angka di tabel ini
sebanding dengan angka FLOPs yang sudah dipublikasikan di
outputs/tabel_hasil_lengkap.json dan outputs/multicriteria_per_rasio.json).

=====================================================================
CATATAN WAJIB -- MACs vs FLOPs (dampak ke SELURUH proyek, bukan cuma
skrip ini):
=====================================================================
thop.profile() TIDAK mengeluarkan FLOPs sungguhan, melainkan MACs
(multiply-accumulate operations) -- diverifikasi LANGSUNG dari source
code thop terpasang di lingkungan ini:
    venv/Lib/site-packages/thop/vision/calc_func.py, fungsi
    calculate_conv2d_flops():
        return l_prod(output_size) * (in_c // g) * l_prod(kernel_size[2:])
    Ini rumus N x Cout x Hout x Wout x (Cin/groups) x Kh x Kw -- TIDAK
    ADA faktor 2 di manapun. Ini definisi MAC baku (thop menamainya
    "flops" di kode internalnya, tapi nilainya MAC). Satu MAC = satu
    operasi kali + satu operasi tambah = 2 FLOPs (konvensi umum
    literatur, mis. Sandler et al. 2018 melaporkan MobileNetV2 1.0x@224
    ~300 juta MACs / ~600 juta FLOPs -- baseline proyek ini melaporkan
    326.218.240 di kolom "flops" tabel_hasil_lengkap.json, jauh lebih
    dekat ke ~300 juta MACs daripada ~600 juta FLOPs, mengonfirmasi
    silang temuan pembacaan source code di atas).

AKIBATNYA: seluruh kolom "flops" yang SUDAH dipublikasikan proyek ini
sejauh ini (outputs/tabel_hasil_lengkap.json, outputs/multicriteria_per_rasio.json,
outputs/ablation_gm_expansion.json, dst. -- semuanya lewat measure_flops()
di src/model.py yang memanggil thop.profile() apa adanya dan melabeli
hasilnya "flops" tanpa konversi) SEBENARNYA adalah MACs, bukan FLOPs.
Skrip ini TIDAK mengubah pelaporan lama itu (di luar cakupan tugas ini,
src/model.py tidak disentuh) -- HANYA melaporkan KEDUA angka secara
eksplisit dan berlabel benar untuk tabel BARU ini:
    MACs = nilai mentah thop.profile() apa adanya
    FLOPs = 2 x MACs (konversi eksplisit, dicetak di setiap baris)

Jalankan:
    venv/Scripts/python.exe scripts/51_hitung_flops.py
"""

import sys
import os
import csv
import json
import copy
import importlib.util

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import thop

from src.config import CFG
from src.model import load_checkpoint
from src.pruning import (
    compute_l1_scores, compute_entropy_scores,
    compute_importance_scores, get_pruning_mask, apply_pruning,
)
from src.dataset import get_dataloaders

# ----- Impor dinamis hitung_skor_gm_ekspansi_semua_layer dari skrip 43 -----
# (pola identik dengan scripts/44/45/46/48/49/50, src/pruning.py TIDAK diubah)
_JALUR_SKRIP_43 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "43_ablation_gm_expansion.py")
_spec_43 = importlib.util.spec_from_file_location("_skrip_43_ablation_gm_expansion", _JALUR_SKRIP_43)
_mod_43 = importlib.util.module_from_spec(_spec_43)
_spec_43.loader.exec_module(_mod_43)
hitung_skor_gm_ekspansi_semua_layer = _mod_43.hitung_skor_gm_ekspansi_semua_layer

PER_RASIO_PATH = CFG.OUTPUT_DIR / "multicriteria_per_rasio.json"
PARAM_CSV_PATH = CFG.OUTPUT_DIR / "tabel_parameter_per_blok.csv"
CSV_PATH = CFG.OUTPUT_DIR / "tabel_flops_per_rasio.csv"
SPLIT_DIR = CFG.DATASET_DIR.parent / "dataset_split"

AMBANG_SELISIH_POLA = 3.0  # poin persentase -- lihat TUGAS 6 (pemeriksaan kewajaran)


def hitung_macs(model, img_size, device):
    """MACs mentah dari thop.profile(), TANPA konversi. Model diprofil
    dalam bentuk SALINAN (bukan objek asli) karena thop.profile() memasang
    buffer total_ops/total_params permanen pada tiap submodule yang
    diprofil -- pola yang SAMA dipakai measure_flops() di src/model.py
    supaya model asli/tersimpan tidak tercemar buffer thop."""
    profile_model = copy.deepcopy(model).to(device)
    profile_model.eval()
    dummy = torch.randn(1, 3, img_size, img_size).to(device)
    macs, params_thop = thop.profile(profile_model, inputs=(dummy,), verbose=False)
    del profile_model
    return macs, params_thop


def main():
    device = torch.device("cpu")

    print("=" * 70)
    print("PERHITUNGAN MACs DAN FLOPs -- FORMULA AKTIF, KETUJUH RASIO")
    print("=" * 70)
    print(f"[INFO] Library: thop versi {thop.__version__} "
          f"({os.path.dirname(thop.__file__)}) -- library YANG SAMA dipakai "
          f"measure_flops() di src/model.py untuk seluruh angka 'flops' yang "
          f"sudah dipublikasikan proyek ini.")
    print(f"[INFO] Ukuran input: {CFG.IMG_SIZE}x{CFG.IMG_SIZE} (dari CFG.IMG_SIZE, src/config.py -- "
          f"BUKAN diasumsikan, dikonfirmasi dari konfigurasi pipeline).")
    print(f"[INFO] Konvensi: thop.profile() mengeluarkan MACs (dibuktikan dari source code thop, "
          f"lihat docstring skrip ini). FLOPs dilaporkan = 2 x MACs, konversi eksplisit.")

    baseline_path = CFG.CHECKPOINT_DIR / "baseline_9class.pth"
    if not baseline_path.exists():
        print(f"[ERROR] {baseline_path} tidak ditemukan.")
        return
    if not PER_RASIO_PATH.exists():
        print(f"[ERROR] {PER_RASIO_PATH} tidak ditemukan.")
        return

    model_baseline, _ = load_checkpoint(baseline_path, device)
    model_baseline.eval()

    with open(PER_RASIO_PATH, encoding="utf-8") as f:
        bobot_per_rasio = json.load(f)["bobot_per_rasio"]

    print("\n[INFO] Menghitung skor L1, GM-Ekspansi, Entropi sekali (tidak bergantung rasio)...")
    dataloaders, _ = get_dataloaders(SPLIT_DIR)
    l1_scores = compute_l1_scores(model_baseline)
    gm_scores = hitung_skor_gm_ekspansi_semua_layer(model_baseline)
    entropy_scores = compute_entropy_scores(
        model_baseline, dataloaders["train"], device, samples_per_class=20, seed=CFG.SEED
    )

    # ----- Baseline -----
    macs_baseline, _ = hitung_macs(model_baseline, CFG.IMG_SIZE, device)
    flops_baseline = macs_baseline * 2

    hasil = [{
        "rasio": "0% (baseline)",
        "macs": macs_baseline,
        "flops": flops_baseline,
    }]

    for ratio in CFG.PRUNING_RATIOS:
        pct = int(round(ratio * 100))
        ratio_key = f"{pct}%"
        if ratio_key not in bobot_per_rasio:
            print(f"[PERINGATAN] Bobot untuk rasio {ratio_key} tidak ditemukan, dilewati.")
            continue
        w = bobot_per_rasio[ratio_key]

        importance = compute_importance_scores(
            l1_scores, gm_scores, entropy_scores,
            w1=w["w_l1"], w2=w["w_gm"], w3=w["w_entropi"]
        )
        masks = get_pruning_mask(importance, ratio)
        pruned_model = apply_pruning(model_baseline, masks)

        macs, _ = hitung_macs(pruned_model, CFG.IMG_SIZE, device)
        flops = macs * 2
        hasil.append({"rasio": ratio_key, "macs": macs, "flops": flops})

    # ----- Bandingkan dengan penghematan parameter yang sudah ada (TUGAS 6) -----
    param_sebelum = param_sesudah = None
    param_per_rasio = {}
    if PARAM_CSV_PATH.exists():
        # total parameter per rasio = num_params di outputs/multicriteria_per_rasio.json
        # (dipakai apa adanya, TIDAK dihitung ulang -- sudah diverifikasi skrip 49)
        with open(PER_RASIO_PATH, encoding="utf-8") as f:
            per_rasio_json = json.load(f)
        for ratio_key, entry in per_rasio_json.get("results", {}).items():
            param_per_rasio[ratio_key] = entry["num_params"]
    else:
        print(f"[PERINGATAN] {PARAM_CSV_PATH} tidak ditemukan -- pemeriksaan kewajaran (TUGAS 6) dilewati.")

    total_params_baseline = sum(p.numel() for p in model_baseline.parameters())

    # ----- Cetak tabel markdown -----
    print(f"\n### Tabel MACs dan FLOPs per Rasio (input {CFG.IMG_SIZE}x{CFG.IMG_SIZE})\n")
    header = ("| Rasio | MACs (M) | FLOPs (M) = 2xMACs | Hemat MACs % | Hemat FLOPs % | "
               "Hemat Params % | Selisih pola (FLOPs% - Params%) |")
    print(header)
    print("|---|---|---|---|---|---|---|")

    baris_csv = []
    for h in hasil:
        macs_m = h["macs"] / 1e6
        flops_m = h["flops"] / 1e6
        hemat_macs = (1 - h["macs"] / macs_baseline) * 100
        hemat_flops = (1 - h["flops"] / flops_baseline) * 100

        if h["rasio"] == "0% (baseline)":
            hemat_params = 0.0
            selisih_pola = 0.0
        else:
            n_param = param_per_rasio.get(h["rasio"])
            if n_param is not None:
                hemat_params = (1 - n_param / total_params_baseline) * 100
                selisih_pola = hemat_flops - hemat_params
            else:
                hemat_params = None
                selisih_pola = None

        row = {
            "rasio": h["rasio"],
            "macs_M": round(macs_m, 2),
            "flops_M": round(flops_m, 2),
            "hemat_macs_persen": round(hemat_macs, 2),
            "hemat_flops_persen": round(hemat_flops, 2),
            "hemat_params_persen": round(hemat_params, 2) if hemat_params is not None else None,
            "selisih_pola_pp": round(selisih_pola, 2) if selisih_pola is not None else None,
        }
        baris_csv.append(row)

        hp_str = f"{row['hemat_params_persen']:.2f}%" if row["hemat_params_persen"] is not None else "N/A"
        sp_str = f"{row['selisih_pola_pp']:+.2f}pp" if row["selisih_pola_pp"] is not None else "N/A"
        print(f"| {row['rasio']} | {row['macs_M']:.2f} | {row['flops_M']:.2f} | "
              f"{row['hemat_macs_persen']:.2f}% | {row['hemat_flops_persen']:.2f}% | {hp_str} | {sp_str} |")

    # ----- TUGAS 6: pemeriksaan kewajaran, dilaporkan apa adanya -----
    print(f"\n[INFO] Pemeriksaan kewajaran (ambang selisih pola: {AMBANG_SELISIH_POLA} poin persentase):")
    ada_yang_lewat_ambang = False
    for row in baris_csv:
        if row["selisih_pola_pp"] is not None and abs(row["selisih_pola_pp"]) > AMBANG_SELISIH_POLA:
            ada_yang_lewat_ambang = True
            print(f"  Rasio {row['rasio']}: selisih {row['selisih_pola_pp']:+.2f}pp "
                  f"MELEBIHI ambang {AMBANG_SELISIH_POLA}pp "
                  f"(hemat FLOPs={row['hemat_flops_persen']:.2f}%, hemat params={row['hemat_params_persen']:.2f}%).")
    if not ada_yang_lewat_ambang:
        print(f"  Tidak ada rasio dengan selisih pola melebihi {AMBANG_SELISIH_POLA}pp -- "
              f"penurunan FLOPs dan penurunan parameter bergerak dalam pola yang berdekatan di seluruh 7 rasio.")
    print(f"  Penjelasan mekanis (BUKAN evaluasi baik/buruk hasil, murni struktural): parameter conv "
          f"1x1 (expansion/projection) TIDAK bergantung resolusi spasial (hanya Cout x Cin), sedangkan "
          f"FLOPs-nya bergantung resolusi spasial (Cout x Cin x Hout x Wout) -- untuk conv depthwise pun "
          f"serupa (param = C x Kh x Kw, FLOPs = C x Hout x Wout x Kh x Kw). Karena resolusi spasial TIDAK "
          f"berubah oleh pemangkasan channel (hanya jumlah channel yang berubah), rasio FLOPs-per-parameter "
          f"pada BAGIAN YANG DIPANGKAS pada dasarnya konstan -- sumber selisih pola (kalau ada) paling mungkin "
          f"berasal dari bagian yang TIDAK dipangkas (stem, block t=1, conv akhir, classifier -- 425.513 "
          f"parameter konstan di semua rasio per outputs/tabel_parameter_per_blok.csv) yang proporsi "
          f"kontribusi FLOPs-nya terhadap total berbeda dari proporsi kontribusi parameternya terhadap total.")

    # ----- Simpan CSV -----
    fieldnames = ["rasio", "macs_M", "flops_M", "hemat_macs_persen", "hemat_flops_persen",
                  "hemat_params_persen", "selisih_pola_pp"]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(baris_csv)
    print(f"\n[INFO] CSV disimpan: {CSV_PATH}")


if __name__ == "__main__":
    main()
