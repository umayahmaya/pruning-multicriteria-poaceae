"""
49_tabel_parameter_per_blok.py
Tabel penghematan parameter per inverted residual block, untuk formula
AKTIF (L1 + GM-Ekspansi + Entropi, bobot per rasio -- CLAUDE.md Bagian 2),
ketujuh rasio yang sudah dipakai di eksperimen (CFG.PRUNING_RATIOS).

CATATAN PENOMORAN: nomor "45" yang diminta di permintaan tugas sudah
dipakai (45_multiseed_per_rasio.py, sesi sebelumnya) -- skrip ini diberi
nomor 49 (nomor berikutnya yang belum dipakai, setelah 48) supaya TIDAK
menimpa skrip yang sudah ada.

Traversal block TIDAK ditulis ulang -- _get_prunable_layers(),
compute_l1_scores(), compute_entropy_scores(), normalize_min_max(),
compute_importance_scores(), get_pruning_mask(), apply_pruning() semua
diimpor APA ADANYA dari src/pruning.py. hitung_skor_gm_ekspansi_semua_layer()
diimpor dinamis dari scripts/43_ablation_gm_expansion.py (pola yang sama
dipakai scripts/44/45/46/48). Bobot w_k(r) per rasio dibaca APA ADANYA
dari outputs/multicriteria_per_rasio.json (bobot_per_rasio), TIDAK
dihitung ulang.

JUMLAH CHANNEL SESUDAH pemangkasan diambil dari mask SUNGGUHAN
(get_pruning_mask() atas skor I(c) gabungan yang sama persis dipakai
pipeline produksi), BUKAN estimasi rasio x lebar -- lalu model benar-benar
direkonstruksi lewat apply_pruning() dan parameter dihitung dari MODUL
NYATA hasil rekonstruksi itu (bukan rumus manual per tensor), supaya
angka yang dilaporkan pasti cocok dengan model yang sungguh-sungguh
dihasilkan pipeline.

Semua Conv2d di MobileNetV2 torchvision punya bias=False (diverifikasi
sebelum menulis skrip ini) -- jadi "jumlah bobot" conv = weight.numel()
saja. BatchNorm2d selalu affine=True (gamma+beta, 2 parameter per
channel) -- ikut dihitung dalam "total bobot blok" (bukan di kolom
"bobot conv" tersendiri, sesuai yang diminta) supaya total block+residual
persis sama dengan sum(p.numel() for p in model.parameters()) pada model
penuh.

Struktur model.features (torchvision MobileNetV2, diverifikasi):
    features[0]      = stem Conv2dNormActivation (TIDAK dipangkas)
    features[1]      = InvertedResidual t=1 (TIDAK dipangkas, di-skip
                        _get_prunable_layers karena tidak py expansion)
    features[2..17]  = 16 InvertedResidual block t>1 (DIPANGKAS -- ini
                        yang dilaporkan tabel ini)
    features[18]     = Conv2dNormActivation akhir 320->1280 (TIDAK dipangkas)
    classifier        = Dropout + Linear(1280, N_KELAS) (TIDAK dipangkas)
Bagian yang TIDAK dipangkas ("residual") dihitung terpisah dan independen
dari total blok (bukan sekadar total-dikurangi-blok) sebagai bagian dari
verifikasi silang di TUGAS akhir skrip ini.

Jalankan:
    venv/Scripts/python.exe scripts/49_tabel_parameter_per_blok.py
"""

import sys
import os
import csv
import json
import importlib.util

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from src.config import CFG
from src.model import load_checkpoint
from src.pruning import (
    compute_l1_scores, compute_entropy_scores, normalize_min_max,
    compute_importance_scores, get_pruning_mask, apply_pruning, _get_prunable_layers,
)
from src.dataset import get_dataloaders

# ----- Impor dinamis hitung_skor_gm_ekspansi_semua_layer dari skrip 43 -----
# (pola identik dengan 44/45/46/48_..., src/pruning.py TIDAK diubah)
_JALUR_SKRIP_43 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "43_ablation_gm_expansion.py")
_spec_43 = importlib.util.spec_from_file_location("_skrip_43_ablation_gm_expansion", _JALUR_SKRIP_43)
_mod_43 = importlib.util.module_from_spec(_spec_43)
_spec_43.loader.exec_module(_mod_43)
hitung_skor_gm_ekspansi_semua_layer = _mod_43.hitung_skor_gm_ekspansi_semua_layer

PER_RASIO_PATH = CFG.OUTPUT_DIR / "multicriteria_per_rasio.json"
CSV_PATH = CFG.OUTPUT_DIR / "tabel_parameter_per_blok.csv"
SPLIT_DIR = CFG.DATASET_DIR.parent / "dataset_split"


def cari_modul_blok(block):
    """Ekstrak enam modul (exp_conv, exp_bn, dw_conv, dw_bn, proj_conv,
    proj_bn) dari SATU InvertedResidual block t>1. Tidak menulis ulang
    logika identifikasi expansion/depthwise -- kriterianya disalin dari
    pola yang sama dipakai _get_prunable_layers()/apply_pruning() di
    src/pruning.py (groups==in_channels untuk depthwise; kernel 1x1
    groups==1 out>in untuk expansion), hanya diterapkan di sini untuk
    MENGEKSTRAK modul individual, bukan untuk menentukan block mana yang
    prunable (itu tugas _get_prunable_layers, dipakai apa adanya)."""
    exp_conv = exp_bn = dw_conv = dw_bn = proj_conv = proj_bn = None
    for layer in block.conv:
        if hasattr(layer, "__getitem__") and hasattr(layer, "__len__"):
            convs = [s for s in layer if isinstance(s, nn.Conv2d)]
            bns = [s for s in layer if isinstance(s, nn.BatchNorm2d)]
            if convs:
                c = convs[0]
                if c.groups == c.in_channels and c.groups > 1:
                    dw_conv = c
                    dw_bn = bns[0] if bns else None
                elif c.kernel_size == (1, 1) and c.groups == 1 and c.out_channels > c.in_channels:
                    exp_conv = c
                    exp_bn = bns[0] if bns else None
        elif isinstance(layer, nn.Conv2d):
            if layer.kernel_size == (1, 1) and layer.groups == 1:
                proj_conv = layer
        elif isinstance(layer, nn.BatchNorm2d):
            proj_bn = layer
    return exp_conv, exp_bn, dw_conv, dw_bn, proj_conv, proj_bn


def n_param(modul):
    """sum(p.numel() for p in modul.parameters()) -- 0 kalau modul None.
    Semua Conv2d di model ini bias=False (diverifikasi), jadi untuk Conv2d
    ini setara weight.numel() saja; untuk BatchNorm2d mencakup weight
    (gamma) + bias (beta)."""
    if modul is None:
        return 0
    return sum(p.numel() for p in modul.parameters())


def rincian_blok(block):
    exp_conv, exp_bn, dw_conv, dw_bn, proj_conv, proj_bn = cari_modul_blok(block)
    return {
        "n_channel": exp_conv.out_channels if exp_conv is not None else None,
        "expansion_conv": n_param(exp_conv),
        "depthwise_conv": n_param(dw_conv),
        "projection_conv": n_param(proj_conv),
        "total_blok": (n_param(exp_conv) + n_param(exp_bn) + n_param(dw_conv)
                       + n_param(dw_bn) + n_param(proj_conv) + n_param(proj_bn)),
    }


def hitung_residual(model):
    """Parameter DI LUAR block yang dipangkas: stem (features[0]),
    block t=1 (features[1]), conv akhir (features[-1]), classifier.
    Dihitung independen (bukan total-dikurangi-blok) supaya jadi
    verifikasi silang yang sah, bukan tautologi."""
    total = 0
    total += sum(p.numel() for p in model.features[0].parameters())
    total += sum(p.numel() for p in model.features[1].parameters())
    total += sum(p.numel() for p in model.features[-1].parameters())
    total += sum(p.numel() for p in model.classifier.parameters())
    return total


def cetak_markdown(rows, judul):
    print(f"\n### {judul}\n")
    header = ("| Block | Channel sebelum | Channel sesudah | Channel dipangkas | "
               "Bobot expansion (sebelum->sesudah) | Bobot depthwise (sebelum->sesudah) | "
               "Bobot projection (sebelum->sesudah) | Total blok sebelum | Total blok sesudah | Hemat % |")
    sep = "|---" * 9 + "|"
    print(header)
    print(sep)
    for r in rows:
        print(f"| {r['block_idx']} | {r['n_channel_sebelum']} | {r['n_channel_sesudah']} | "
              f"{r['n_channel_dipangkas']} | {r['expansion_sebelum']:,} -> {r['expansion_sesudah']:,} | "
              f"{r['depthwise_sebelum']:,} -> {r['depthwise_sesudah']:,} | "
              f"{r['projection_sebelum']:,} -> {r['projection_sesudah']:,} | "
              f"{r['total_blok_sebelum']:,} | {r['total_blok_sesudah']:,} | {r['hemat_persen']:.2f}% |")


def main():
    device = torch.device("cpu")

    baseline_path = CFG.CHECKPOINT_DIR / "baseline_9class.pth"
    if not baseline_path.exists():
        print(f"[ERROR] {baseline_path} tidak ditemukan.")
        return
    if not PER_RASIO_PATH.exists():
        print(f"[ERROR] {PER_RASIO_PATH} tidak ditemukan -- jalankan "
              f"44_multicriteria_per_rasio.py --jalankan-eksperimen terlebih dahulu.")
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

    prunable_baseline = _get_prunable_layers(model_baseline)
    block_idx_list = [b for b, _ in prunable_baseline]

    # Rincian block SEBELUM (sama untuk semua rasio -- baseline tidak berubah)
    rincian_sebelum = {bidx: rincian_blok(model_baseline.features[bidx]) for bidx in block_idx_list}
    residual_sebelum = hitung_residual(model_baseline)
    total_ground_truth_sebelum = sum(p.numel() for p in model_baseline.parameters())

    semua_baris_csv = []
    ringkasan_total = []
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
        pruned_model = apply_pruning(model_baseline, masks)

        rows = []
        for bidx in block_idx_list:
            sebelum = rincian_sebelum[bidx]
            sesudah = rincian_blok(pruned_model.features[bidx])
            n_sebelum = sebelum["n_channel"]
            n_sesudah = sesudah["n_channel"]
            row = {
                "rasio": ratio_key,
                "block_idx": bidx,
                "n_channel_sebelum": n_sebelum,
                "n_channel_sesudah": n_sesudah,
                "n_channel_dipangkas": n_sebelum - n_sesudah,
                "expansion_sebelum": sebelum["expansion_conv"],
                "expansion_sesudah": sesudah["expansion_conv"],
                "depthwise_sebelum": sebelum["depthwise_conv"],
                "depthwise_sesudah": sesudah["depthwise_conv"],
                "projection_sebelum": sebelum["projection_conv"],
                "projection_sesudah": sesudah["projection_conv"],
                "total_blok_sebelum": sebelum["total_blok"],
                "total_blok_sesudah": sesudah["total_blok"],
                "hemat_persen": (1 - sesudah["total_blok"] / sebelum["total_blok"]) * 100,
            }
            rows.append(row)
            semua_baris_csv.append(row)

        cetak_markdown(rows, f"Rasio {ratio_key}")

        # ----- Verifikasi wajib -----
        residual_sesudah = hitung_residual(pruned_model)
        total_rekonstruksi = sum(r["total_blok_sesudah"] for r in rows) + residual_sesudah
        total_ground_truth = sum(p.numel() for p in pruned_model.parameters())

        cocok = (total_rekonstruksi == total_ground_truth)
        if not cocok:
            semua_verifikasi_lulus = False
        status = "LULUS" if cocok else f"GAGAL (selisih {total_rekonstruksi - total_ground_truth:+,})"

        print(f"\n  [VERIFIKASI rasio {ratio_key}] "
              f"sum(16 blok sesudah) + residual = {sum(r['total_blok_sesudah'] for r in rows):,} + "
              f"{residual_sesudah:,} = {total_rekonstruksi:,}  vs  "
              f"sum(p.numel() for p in pruned_model.parameters()) = {total_ground_truth:,}  "
              f"-> {status}")

        ringkasan_total.append({
            "rasio": ratio_key,
            "total_sebelum": total_ground_truth_sebelum,
            "total_sesudah": total_ground_truth,
            "hemat_persen": (1 - total_ground_truth / total_ground_truth_sebelum) * 100,
            "verifikasi": status,
        })

    # ----- Tabel ringkasan total model -----
    print(f"\n\n### Ringkasan Total Parameter Model (ground truth: sum(p.numel() for p in model.parameters()))\n")
    print("| Rasio | Total sebelum | Total sesudah | Hemat % | Verifikasi |")
    print("|---|---|---|---|---|")
    print(f"| 0% (baseline) | {total_ground_truth_sebelum:,} | {total_ground_truth_sebelum:,} | 0.00% | -- |")
    for r in ringkasan_total:
        print(f"| {r['rasio']} | {r['total_sebelum']:,} | {r['total_sesudah']:,} | "
              f"{r['hemat_persen']:.2f}% | {r['verifikasi']} |")

    # ----- Simpan CSV (satu file, kolom rasio) -----
    fieldnames = ["rasio", "block_idx", "n_channel_sebelum", "n_channel_sesudah", "n_channel_dipangkas",
                  "expansion_sebelum", "expansion_sesudah", "depthwise_sebelum", "depthwise_sesudah",
                  "projection_sebelum", "projection_sesudah", "total_blok_sebelum", "total_blok_sesudah",
                  "hemat_persen"]
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(semua_baris_csv)
    print(f"\n[INFO] CSV disimpan: {CSV_PATH}")

    print(f"\n{'='*70}")
    print(f"[KESIMPULAN AKHIR] Semua verifikasi (7 rasio) LULUS? {semua_verifikasi_lulus}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
