"""
53_ablation_l1_expansion.py
Ablation Norma L1 dihitung dari bobot EXPANSION CONV 1x1 (BUKAN depthwise
seperti compute_l1_scores() di src/pruning.py, dipakai formula aktif) --
pembanding L1 depthwise yang selama ini dipakai, analog dengan GM
depthwise (scripts/archive/42_ablation_gm.py, arsip) vs GM-Ekspansi
(scripts/43_ablation_gm_expansion.py, dipakai sekarang) yang sudah
dibandingkan sebelumnya.

Fungsi skor baru (hitung_skor_l1_ekspansi_semua_layer) TIDAK menulis
ulang traversal -- _get_prunable_layers() diimpor APA ADANYA dari
src/pruning.py, cari_conv_ekspansi() diimpor dinamis dari
scripts/43_ablation_gm_expansion.py (fungsi itu sendiri TIDAK diubah).
Rumus skor SAMA PERSIS dengan compute_l1_scores() (jumlah nilai mutlak
bobot per filter, dim 1,2,3), hanya sumber tensornya beda (expansion,
bukan depthwise).

VERIFIKASI WAJIB dijalankan SEBELUM training apapun (di kedua mode,
review maupun --jalankan-eksperimen): jumlah channel per layer L1-ekspansi
harus identik dengan compute_l1_scores() (depthwise) di SEMUA 16 layer.
Kalau ada satu saja yang tidak cocok, skrip BERHENTI (return), TIDAK
lanjut ke ablasi apapun.

Prosedur fine-tuning: pola IDENTIK dengan scripts/43_ablation_gm_expansion.py
-- 30 epoch (CFG.BASELINE_EPOCHS), CFG.FINETUNE_LR, Adam+StepLR
(train_model()), seed=CFG.SEED (42) dipanggil SEKALI sebelum loop rasio
(keterbatasan metodologis yang sama berlaku, CLAUDE.md Bagian 9). Evaluasi
VALIDASI dan TEST terpisah. Mode operasi SAMA seperti skrip 43: TANPA
--jalankan-eksperimen hanya menampilkan rencana + estimasi durasi lalu
berhenti (model baseline TIDAK dimuat, tidak ada training).

Akurasi VALIDASI L1 DEPTHWISE (lama) TIDAK dihitung ulang -- dibaca apa
adanya dari outputs/ablation_val_results.json
(ablation_val_results.{r}%.l1.accuracy), SUDAH dikonfirmasi tersedia
untuk ketujuh rasio sebelum skrip ini ditulis (dicek langsung, bukan
diasumsikan).

Checkpoint: ablation_l1exp_{rasio}pct_30ep.pth -- nama baru, TIDAK
menimpa checkpoint L1 depthwise yang ada (ablation_l1_{rasio}pct_30ep.pth).

Jalankan:
    venv/Scripts/python.exe scripts/53_ablation_l1_expansion.py
        (mode tinjau -- verifikasi + rencana + estimasi durasi, tidak melatih apa pun)
    venv/Scripts/python.exe scripts/53_ablation_l1_expansion.py --jalankan-eksperimen
"""

import sys
import os
import json
import argparse
import random
import time
import importlib.util

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.config import CFG
from src.dataset import get_dataloaders
from src.model import train_model, evaluate_model, load_checkpoint
from src.pruning import (
    compute_l1_scores, normalize_min_max, get_pruning_mask, apply_pruning, _get_prunable_layers,
)
from src.visualize import print_results_table

# ----- Impor dinamis cari_conv_ekspansi dari skrip 43 -----
# (nama file diawali angka -> tidak bisa `import` biasa; TIDAK mengubah
# skrip 43, hanya memuat definisinya sebagai modul terpisah -- pola sama
# dipakai scripts/44/45/46/48/49/51/52)
_JALUR_SKRIP_43 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "43_ablation_gm_expansion.py")
_spec_43 = importlib.util.spec_from_file_location("_skrip_43_ablation_gm_expansion", _JALUR_SKRIP_43)
_mod_43 = importlib.util.module_from_spec(_spec_43)
_spec_43.loader.exec_module(_mod_43)
cari_conv_ekspansi = _mod_43.cari_conv_ekspansi

CHECKPOINT_NAME = "baseline_9class.pth"
VAL_RESULTS_PATH = CFG.OUTPUT_DIR / "ablation_val_results.json"
RESULTS_PATH = CFG.OUTPUT_DIR / "ablation_l1_expansion.json"
EPOCHS = CFG.BASELINE_EPOCHS  # 30
CRITERION_NAME = "l1exp"
MENIT_PER_RASIO_HISTORIS = (43.75 + 45.95 + 71.67) / 3  # basis sama dipakai skrip 43


def set_seed(seed=CFG.SEED):
    random.seed(seed)
    __import__("numpy").random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def hitung_skor_l1_ekspansi_semua_layer(model):
    """
    s_L1exp(c) = sum(|W_c|), W_c = baris bobot EXPANSION conv 1x1
    channel c (dim 1,2,3 dijumlahkan) -- rumus SAMA PERSIS dengan
    compute_l1_scores() di src/pruning.py, hanya sumber tensornya beda
    (expansion, bukan depthwise).

    Args:
        model: MobileNetV2

    Returns:
        dict: {layer_idx: tensor skor L1-Ekspansi per channel}
    """
    prunable = _get_prunable_layers(model)
    l1_exp_scores = {}

    for layer_idx, (block_idx, dw_module) in enumerate(prunable):
        block = model.features[block_idx]
        exp_conv = cari_conv_ekspansi(block)
        if exp_conv is None:
            print(f"[PERINGATAN] layer {layer_idx} (block {block_idx}): conv ekspansi "
                  f"tidak ditemukan, dilewati.")
            continue
        weights = exp_conv.weight.data  # (mid_channels, in_channels, 1, 1)
        scores = weights.abs().sum(dim=[1, 2, 3])
        l1_exp_scores[layer_idx] = scores.cpu()

    print(f"[L1-Ekspansi] Dihitung untuk {len(l1_exp_scores)} lapisan (skip t=1)")
    return l1_exp_scores


def verifikasi_kesejajaran_channel(model):
    """TUGAS 2: bandingkan jumlah channel L1-depthwise (lama) vs
    L1-ekspansi (baru) per layer. Return True kalau SEMUA layer cocok,
    False kalau ada satu saja yang beda -- pemanggil WAJIB berhenti
    kalau False."""
    l1_depthwise = compute_l1_scores(model)
    l1_ekspansi = hitung_skor_l1_ekspansi_semua_layer(model)

    print(f"\n{'=' * 70}")
    print("VERIFIKASI WAJIB: kesejajaran channel L1-depthwise vs L1-ekspansi")
    print(f"{'=' * 70}")
    print(f"{'layer_idx':>10} {'len(L1 depthwise)':>18} {'len(L1 ekspansi)':>17}  status")
    print("-" * 55)

    semua_layer = sorted(set(l1_depthwise.keys()) | set(l1_ekspansi.keys()))
    semua_cocok = True
    for idx in semua_layer:
        n_dw = len(l1_depthwise.get(idx, []))
        n_exp = len(l1_ekspansi.get(idx, []))
        cocok = (n_dw == n_exp and n_dw > 0)
        if not cocok:
            semua_cocok = False
        print(f"{idx:>10} {n_dw:>18} {n_exp:>17}  {'OK' if cocok else '!!! TIDAK COCOK !!!'}")

    print("-" * 55)
    print(f"[VERIFIKASI WAJIB] Semua {len(semua_layer)} layer sejajar jumlah channelnya? {semua_cocok}")
    return semua_cocok, l1_ekspansi


def run_ablation_l1exp_single(l1_ekspansi_raw, model_baseline, ratio):
    """Mirror run_ablation_single_criterion() di src/pruning.py UNTUK
    KRITERIA L1-EKSPANSI -- fungsi itu tidak mengenalnya, logikanya
    ditulis ulang di sini (pola sama dipakai run_ablation_gmexp_single()
    di scripts/43). src/pruning.py TIDAK diubah."""
    scores = normalize_min_max(l1_ekspansi_raw)
    masks = get_pruning_mask(scores, ratio)
    pruned_model = apply_pruning(model_baseline, masks)
    return pruned_model, masks


def run_single_ratio(l1_ekspansi_raw, model_baseline, dataloaders, dataset_sizes, ratio, device):
    ratio_persen = int(round(ratio * 100))
    ckpt_name = f"ablation_{CRITERION_NAME}_{ratio_persen}pct_{EPOCHS}ep.pth"
    ckpt_path = CFG.CHECKPOINT_DIR / ckpt_name

    if ckpt_path.exists():
        print(f"\n{'=' * 60}")
        print(f"SKIP: L1-Ekspansi | {ratio_persen}% | {EPOCHS}ep (checkpoint sudah ada: {ckpt_name})")
        print(f"{'=' * 60}")
        pruned_model, _ = load_checkpoint(ckpt_path, device)
    else:
        print(f"\n{'=' * 60}")
        print(f"ABLATION: L1-Ekspansi saja | Rasio {ratio_persen}% | {EPOCHS} epoch")
        print(f"{'=' * 60}")

        pruned_model, masks = run_ablation_l1exp_single(l1_ekspansi_raw, model_baseline, ratio)

        pruned_model, history = train_model(
            model=pruned_model,
            dataloaders=dataloaders,
            dataset_sizes=dataset_sizes,
            num_epochs=EPOCHS,
            lr=CFG.FINETUNE_LR,
            device=device,
            checkpoint_path=ckpt_path,
            phase_name=f"Ablation L1-Ekspansi {ratio_persen}%"
        )

    metrics_val, _, _ = evaluate_model(pruned_model, dataloaders["val"], device)
    print_results_table(metrics_val, f"L1-Ekspansi | {ratio_persen}% -- VALIDASI")

    metrics_test, _, _ = evaluate_model(pruned_model, dataloaders["test"], device)
    print_results_table(metrics_test, f"L1-Ekspansi | {ratio_persen}% -- TEST")

    return metrics_val, metrics_test, ckpt_name


def muat_l1_depthwise_val_existing():
    """Baca akurasi VALIDASI L1 depthwise (lama) yang SUDAH ADA --
    TIDAK dihitung ulang, TIDAK dilatih ulang."""
    hasil = {}
    if VAL_RESULTS_PATH.exists():
        with open(VAL_RESULTS_PATH, encoding="utf-8") as f:
            val_data = json.load(f)
        for ratio_key, entry in val_data.get("ablation_val_results", {}).items():
            if "l1" in entry:
                hasil[ratio_key] = entry["l1"]["accuracy"]
    return hasil


def cetak_tabel_perbandingan(hasil_l1exp, l1_depthwise_val):
    print(f"\n{'=' * 90}")
    print("TABEL PERBANDINGAN: L1 DEPTHWISE (lama) vs L1 EKSPANSI (baru) -- AKURASI VALIDASI")
    print(f"{'=' * 90}")
    print(f"{'Rasio':>6} {'L1 Depthwise (val)':>20} {'L1 Ekspansi (val)':>19} {'Selisih (pp)':>13}")
    print("-" * 90)

    baris = []
    for ratio in CFG.PRUNING_RATIOS:
        ratio_key = f"{int(ratio * 100)}%"
        dw = l1_depthwise_val.get(ratio_key)
        exp = hasil_l1exp.get(ratio_key, {}).get("validasi", {}).get("accuracy")

        def fmt(x):
            return f"{x*100:.2f}%" if x is not None else "N/A"

        selisih = (exp - dw) * 100 if (dw is not None and exp is not None) else None
        selisih_str = f"{selisih:+.2f}" if selisih is not None else "N/A"
        print(f"{ratio_key:>6} {fmt(dw):>20} {fmt(exp):>19} {selisih_str:>13}")
        baris.append({"rasio": ratio_key, "l1_depthwise_val": dw, "l1_ekspansi_val": exp,
                       "selisih_pp": selisih})
    print("=" * 90)
    return baris


def main():
    parser = argparse.ArgumentParser(
        description="Ablation study kriteria tunggal L1-Ekspansi (expansion conv), 7 rasio"
    )
    parser.add_argument(
        "--jalankan-eksperimen", action="store_true",
        help="Jalankan (atau lanjutkan) ketujuh rasio. Tanpa flag ini, skrip HANYA "
             "menjalankan verifikasi + menampilkan rencana dan perkiraan durasi, lalu berhenti."
    )
    args = parser.parse_args()

    ratios = CFG.PRUNING_RATIOS

    print("=" * 70)
    print("ABLATION STUDY: L1-EKSPANSI SAJA (expansion conv, |bobot|) -- 7 RASIO")
    print("=" * 70)
    print(f"  Rasio   : {[f'{r*100:.0f}%' for r in ratios]}")
    print(f"  Epoch   : {EPOCHS}")
    print(f"  LR      : {CFG.FINETUNE_LR}")
    print(f"  Seed    : {CFG.SEED}")
    print(f"  Optimizer: Adam + StepLR(step_size=10, gamma=0.1)  [via train_model(), src/model.py]")

    # ----- Muat baseline & VERIFIKASI WAJIB (Tugas 2) -- SELALU dijalankan, kedua mode -----
    device = torch.device("cpu")
    ckpt_path = CFG.CHECKPOINT_DIR / CHECKPOINT_NAME
    if not ckpt_path.exists():
        print(f"[ERROR] Checkpoint tidak ditemukan: {ckpt_path}")
        return
    model_baseline, _ = load_checkpoint(ckpt_path, device)
    model_baseline.eval()

    semua_cocok, l1_ekspansi_raw = verifikasi_kesejajaran_channel(model_baseline)
    if not semua_cocok:
        print("\n[BERHENTI] Verifikasi kesejajaran channel GAGAL -- ablasi TIDAK dijalankan.")
        return

    # ----- Cek data L1 depthwise (lama) tersedia -----
    l1_depthwise_val = muat_l1_depthwise_val_existing()
    if len(l1_depthwise_val) < len(ratios):
        print(f"\n[PERINGATAN] Akurasi validasi L1 depthwise (lama) hanya ditemukan untuk "
              f"{len(l1_depthwise_val)}/{len(ratios)} rasio di {VAL_RESULTS_PATH}. "
              f"Tabel perbandingan nanti akan punya N/A untuk rasio yang hilang.")
    else:
        print(f"\n[INFO] Akurasi validasi L1 depthwise (lama) tersedia lengkap untuk {len(ratios)} rasio "
              f"di {VAL_RESULTS_PATH} -- TIDAK dihitung ulang.")

    # ----- Rencana + estimasi durasi -----
    sudah_ada, belum_ada = [], []
    for ratio in ratios:
        ckpt_name = f"ablation_{CRITERION_NAME}_{int(ratio*100)}pct_{EPOCHS}ep.pth"
        (sudah_ada if (CFG.CHECKPOINT_DIR / ckpt_name).exists() else belum_ada).append(ratio)

    print(f"\n{'=' * 70}")
    print("RENCANA")
    print("=" * 70)
    print(f"  Checkpoint sudah ada (di-skip)  : {[f'{r*100:.0f}%' for r in sudah_ada] if sudah_ada else '(tidak ada)'}")
    print(f"  Akan dilatih (30 epoch)         : {[f'{r*100:.0f}%' for r in belum_ada] if belum_ada else '(tidak ada)'}")

    estimasi_menit = len(belum_ada) * MENIT_PER_RASIO_HISTORIS
    print(f"\n{'=' * 70}")
    print("PERKIRAAN DURASI")
    print("=" * 70)
    print(f"  Rasio akan dilatih : {len(belum_ada)}")
    print(f"  Basis perkiraan    : rerata {MENIT_PER_RASIO_HISTORIS:.1f} menit/rasio "
          f"(riwayat scripts/43_ablation_gm_expansion.py, arsitektur+epoch identik)")
    print(f"  Estimasi total     : {estimasi_menit:.0f} menit (~{estimasi_menit/60:.1f} jam)")
    print("  (Perkiraan berbasis riwayat, BUKAN kalibrasi langsung -- durasi aktual bisa berbeda.)")
    print("=" * 70)

    if not args.jalankan_eksperimen:
        print("\n[INFO] Mode TINJAU SAJA (tanpa --jalankan-eksperimen). Berhenti di sini.")
        print("[INFO] Tidak ada fine-tuning yang dijalankan.")
        print("[INFO] Jalankan ulang dengan --jalankan-eksperimen setelah konfirmasi diberikan.")
        return

    # ----- Jalankan -----
    set_seed()
    SPLIT_DIR = CFG.DATASET_DIR.parent / "dataset_split"
    dataloaders, dataset_sizes = get_dataloaders(SPLIT_DIR)

    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, encoding="utf-8") as f:
            hasil_l1exp = json.load(f).get("results", {})
        print(f"\n[INFO] Melanjutkan dari hasil sebagian yang sudah ada: {RESULTS_PATH}")
    else:
        hasil_l1exp = {}

    start_time = time.time()
    for i, ratio in enumerate(ratios, 1):
        ratio_key = f"{int(ratio * 100)}%"
        elapsed = time.time() - start_time
        print(f"\n>>> Rasio {i}/{len(ratios)} | {ratio_key} | Waktu berjalan: {elapsed/60:.1f} menit")

        metrics_val, metrics_test, ckpt_name = run_single_ratio(
            l1_ekspansi_raw, model_baseline, dataloaders, dataset_sizes, ratio, device
        )
        hasil_l1exp[ratio_key] = {
            "checkpoint": ckpt_name,
            "validasi": {"accuracy": metrics_val["accuracy"], "f1_score": metrics_val["f1_score"]},
            "test": {"accuracy": metrics_test["accuracy"], "f1_score": metrics_test["f1_score"]},
            "num_params": metrics_test["num_params"],
            "model_size_mb": metrics_test["model_size_mb"],
            "flops": metrics_test["flops"],
            "inference_ms": metrics_test["inference_ms"],
        }

        save_data = {
            "criterion": "l1_expansion",
            "definisi": "s_L1exp(c) = sum(|W_c|), W_c = baris bobot EXPANSION CONV 1x1 channel c "
                        "-- rumus sama dengan compute_l1_scores() (src/pruning.py), sumber tensor beda "
                        "(expansion, bukan depthwise). Pembanding L1 depthwise yang dipakai formula aktif.",
            "epochs_used": EPOCHS,
            "seed": CFG.SEED,
            "results": hasil_l1exp,
            "status": f"{i}/{len(ratios)} rasio diproses",
        }
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Hasil sementara disimpan: {RESULTS_PATH}")

    total_time = time.time() - start_time
    save_data["status"] = "SELESAI"
    save_data["total_time_seconds"] = total_time

    baris_perbandingan = cetak_tabel_perbandingan(hasil_l1exp, l1_depthwise_val)
    save_data["perbandingan_l1_depthwise_vs_ekspansi"] = baris_perbandingan

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print(f"[SELESAI] Ablation L1-Ekspansi saja -- total waktu: {total_time/60:.1f} menit "
          f"({total_time/3600:.2f} jam)")
    print(f"Hasil: {RESULTS_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
