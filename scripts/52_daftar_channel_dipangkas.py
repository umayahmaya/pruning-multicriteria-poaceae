"""
52_daftar_channel_dipangkas.py
Daftar rinci indeks channel yang dipangkas/dipertahankan per block, untuk
formula AKTIF (L1 + GM-Ekspansi + Entropi, bobot per rasio), ketujuh rasio.

Rekonstruksi model: pola IDENTIK dengan scripts/49_tabel_parameter_per_blok.py
-- _get_prunable_layers(), compute_l1_scores(), compute_entropy_scores(),
normalize_min_max(), compute_importance_scores(), get_pruning_mask(),
apply_pruning() diimpor APA ADANYA dari src/pruning.py (TIDAK diubah).
hitung_skor_gm_ekspansi_semua_layer() diimpor dinamis dari
scripts/43_ablation_gm_expansion.py (pola sama dipakai skrip 44/45/46/48/49/51).
Bobot w_k(r) dibaca APA ADANYA dari outputs/multicriteria_per_rasio.json.

Skor L1/GM/Entropi ternormalisasi di CSV: normalize_min_max() (fungsi
publik yang SAMA dipakai compute_importance_scores() secara internal)
dipanggil terpisah pada masing-masing skor mentah untuk mendapatkan tiga
komponen ternormalisasi secara eksplisit -- BUKAN menulis ulang rumus
normalisasi, hanya memanggil fungsi yang sudah ada untuk tujuan
pelaporan. Skor I(c) gabungan tetap dari compute_importance_scores()
resminya (bukan dihitung ulang manual dari tiga komponen itu), supaya
konsisten persis dengan yang dipakai get_pruning_mask().

Jalankan:
    venv/Scripts/python.exe scripts/52_daftar_channel_dipangkas.py
"""

import sys
import os
import csv
import json
import importlib.util

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import CFG
from src.model import load_checkpoint
from src.pruning import (
    compute_l1_scores, compute_entropy_scores, normalize_min_max,
    compute_importance_scores, get_pruning_mask, apply_pruning, _get_prunable_layers,
)
from src.dataset import get_dataloaders

# ----- Impor dinamis hitung_skor_gm_ekspansi_semua_layer dari skrip 43 -----
# (pola identik dengan scripts/44/45/46/48/49/51, src/pruning.py TIDAK diubah)
_JALUR_SKRIP_43 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "43_ablation_gm_expansion.py")
_spec_43 = importlib.util.spec_from_file_location("_skrip_43_ablation_gm_expansion", _JALUR_SKRIP_43)
_mod_43 = importlib.util.module_from_spec(_spec_43)
_spec_43.loader.exec_module(_mod_43)
hitung_skor_gm_ekspansi_semua_layer = _mod_43.hitung_skor_gm_ekspansi_semua_layer

PER_RASIO_PATH = CFG.OUTPUT_DIR / "multicriteria_per_rasio.json"
PARAM_CSV_PATH = CFG.OUTPUT_DIR / "tabel_parameter_per_blok.csv"
CSV_PATH = CFG.OUTPUT_DIR / "daftar_channel_dipangkas.csv"
SPLIT_DIR = CFG.DATASET_DIR.parent / "dataset_split"

RASIO_UNTUK_RINGKASAN = ["20%", "30%"]  # KELUARAN 2 hanya untuk dua rasio ini
N_TAMPIL = 5  # 5 channel skor terendah + 5 tertinggi per block di ringkasan


def ringkas_rentang(daftar_indeks):
    """[3,4,5,6,10,15,16] -> '3-6, 10, 15-16'. Murni pemformatan tampilan,
    daftar lengkapnya tetap tersimpan apa adanya di CSV."""
    if not daftar_indeks:
        return "(tidak ada)"
    daftar_indeks = sorted(daftar_indeks)
    rentang = []
    awal = akhir = daftar_indeks[0]
    for x in daftar_indeks[1:]:
        if x == akhir + 1:
            akhir = x
        else:
            rentang.append(f"{awal}-{akhir}" if awal != akhir else f"{awal}")
            awal = akhir = x
    rentang.append(f"{awal}-{akhir}" if awal != akhir else f"{awal}")
    return ", ".join(rentang)


def muat_n_channel_sesudah_dari_csv():
    """{(rasio, block_idx): n_channel_sesudah} dari
    outputs/tabel_parameter_per_blok.csv -- dipakai APA ADANYA untuk
    verifikasi wajib, TIDAK dihitung ulang di sini."""
    hasil = {}
    if not PARAM_CSV_PATH.exists():
        return hasil
    with open(PARAM_CSV_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            hasil[(row["rasio"], int(row["block_idx"]))] = int(row["n_channel_sesudah"])
    return hasil


def main():
    device = torch.device("cpu")

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

    print("[INFO] Menghitung skor L1, GM-Ekspansi, Entropi sekali (tidak bergantung rasio)...")
    dataloaders, _ = get_dataloaders(SPLIT_DIR)
    l1_scores = compute_l1_scores(model_baseline)
    gm_scores = hitung_skor_gm_ekspansi_semua_layer(model_baseline)
    entropy_scores = compute_entropy_scores(
        model_baseline, dataloaders["train"], device, samples_per_class=20, seed=CFG.SEED
    )
    # Komponen ternormalisasi terpisah -- fungsi publik yang SAMA dipakai
    # compute_importance_scores() secara internal, dipanggil di sini
    # murni untuk pelaporan, bukan mengubah cara I(c) dihitung.
    l1_norm = normalize_min_max(l1_scores)
    gm_norm = normalize_min_max(gm_scores)
    entropy_norm = normalize_min_max(entropy_scores)

    prunable = _get_prunable_layers(model_baseline)
    layer_to_block = {i: b for i, (b, _) in enumerate(prunable)}

    n_channel_sesudah_ref = muat_n_channel_sesudah_dari_csv()
    if not n_channel_sesudah_ref:
        print(f"[PERINGATAN] {PARAM_CSV_PATH} tidak ditemukan -- verifikasi wajib dilewati "
              f"(jalankan scripts/49_tabel_parameter_per_blok.py dulu untuk itu).")

    semua_baris_csv = []
    ringkasan_untuk_cetak = {}  # rasio -> {block_idx: {...}}
    semua_verifikasi_lulus = True

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

        ringkasan_rasio_ini = {}

        for layer_idx in sorted(masks.keys()):
            block_idx = layer_to_block[layer_idx]
            mask = masks[layer_idx]
            skor = importance[layer_idx]
            n_channel = len(skor)

            urutan_naik = torch.argsort(skor)  # ascending -- peringkat 1 = skor terendah
            peringkat = torch.empty(n_channel, dtype=torch.long)
            peringkat[urutan_naik] = torch.arange(1, n_channel + 1)

            channel_dipangkas = []
            for c in range(n_channel):
                dipangkas = not bool(mask[c].item())
                if dipangkas:
                    channel_dipangkas.append(c)
                semua_baris_csv.append({
                    "rasio": ratio_key,
                    "block_idx": block_idx,
                    "channel_idx": c,
                    "skor_Ic": round(skor[c].item(), 6),
                    "skor_L1_norm": round(l1_norm[layer_idx][c].item(), 6),
                    "skor_GM_norm": round(gm_norm[layer_idx][c].item(), 6),
                    "skor_entropi_norm": round(entropy_norm[layer_idx][c].item(), 6),
                    "status": "dipangkas" if dipangkas else "dipertahankan",
                    "peringkat_di_layer": int(peringkat[c].item()),
                })

            n_dipertahankan = int(mask.sum().item())

            # ----- Verifikasi wajib -----
            ref = n_channel_sesudah_ref.get((ratio_key, block_idx))
            if ref is not None and ref != n_dipertahankan:
                semua_verifikasi_lulus = False
                print(f"[VERIFIKASI GAGAL] rasio={ratio_key} block_idx={block_idx}: "
                      f"n_dipertahankan skrip ini={n_dipertahankan} vs "
                      f"n_channel_sesudah di tabel_parameter_per_blok.csv={ref} -- SELISIH {n_dipertahankan - ref}")

            if ratio_key in RASIO_UNTUK_RINGKASAN:
                urutan_skor_naik = sorted(range(n_channel), key=lambda c: skor[c].item())
                ringkasan_rasio_ini[block_idx] = {
                    "n_channel": n_channel,
                    "n_dipangkas": len(channel_dipangkas),
                    "n_dipertahankan": n_dipertahankan,
                    "channel_dipangkas": channel_dipangkas,
                    "terendah_5": [(c, skor[c].item()) for c in urutan_skor_naik[:N_TAMPIL]],
                    "tertinggi_5": [(c, skor[c].item()) for c in urutan_skor_naik[-N_TAMPIL:][::-1]],
                }

        if ratio_key in RASIO_UNTUK_RINGKASAN:
            ringkasan_untuk_cetak[ratio_key] = ringkasan_rasio_ini

    # ----- Simpan CSV -----
    fieldnames = ["rasio", "block_idx", "channel_idx", "skor_Ic", "skor_L1_norm",
                  "skor_GM_norm", "skor_entropi_norm", "status", "peringkat_di_layer"]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(semua_baris_csv)
    print(f"\n[INFO] CSV rinci disimpan: {CSV_PATH}  ({len(semua_baris_csv):,} baris)")

    print(f"\n[VERIFIKASI WAJIB] n_dipertahankan per block cocok dengan "
          f"tabel_parameter_per_blok.csv di SEMUA rasio dan block? {semua_verifikasi_lulus}")

    # ----- KELUARAN 2: ringkasan markdown untuk paparan -----
    for ratio_key in RASIO_UNTUK_RINGKASAN:
        if ratio_key not in ringkasan_untuk_cetak:
            continue
        print(f"\n\n{'=' * 78}")
        print(f"RINGKASAN RASIO {ratio_key}")
        print(f"{'=' * 78}")
        for block_idx in sorted(ringkasan_untuk_cetak[ratio_key].keys()):
            r = ringkasan_untuk_cetak[ratio_key][block_idx]
            print(f"\n### Block {block_idx}")
            print(f"- Channel dipertahankan: {r['n_dipertahankan']} / {r['n_channel']}  "
                  f"(dipangkas: {r['n_dipangkas']})")
            print(f"- Indeks channel DIPANGKAS: {ringkas_rentang(r['channel_dipangkas'])}")
            print(f"- 5 channel skor TERENDAH: " +
                  ", ".join(f"#{c} (I(c)={s:.3f})" for c, s in r["terendah_5"]))
            print(f"- 5 channel skor TERTINGGI: " +
                  ", ".join(f"#{c} (I(c)={s:.3f})" for c, s in r["tertinggi_5"]))


if __name__ == "__main__":
    main()
