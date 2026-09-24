"""
46_visualisasi_peta_fitur.py
Visualisasi peta fitur channel intermediate berskor I(c) tertinggi vs
terendah, pada satu inverted residual block di tengah jaringan, untuk
model baseline (dilatih pada dataset daun Poaceae, BUKAN pretrained
mentah, BUKAN model yang sudah dipangkas).

Tidak menulis ulang logika skoring -- compute_l1_scores(),
compute_entropy_scores(), normalize_min_max(), compute_importance_scores(),
_get_prunable_layers() diimpor APA ADANYA dari src/pruning.py.
hitung_skor_gm_ekspansi_semua_layer() diimpor dinamis dari
scripts/43_ablation_gm_expansion.py (pola sama dengan skrip 44/45/46
lama/48/49 -- src/pruning.py TIDAK diubah).

Bobot w1/w2/w3 untuk I(c): dipakai bobot rasio 20% dari
outputs/multicriteria_per_rasio.json (bobot_per_rasio) -- 20% dipilih
karena itu titik operasi utama tesis (README.md "Hasil Utama"), BUKAN
dihitung ulang di sini. Skor I(c) itu sendiri TIDAK bergantung rasio
pemangkasan mana yang akhirnya dipilih (skor mentah sama untuk ketujuh
rasio, CLAUDE.md Bagian 2) -- yang bergantung rasio hanya bobotnya, jadi
pilihan rasio referensi ini hanya memengaruhi PERINGKAT channel yang
divisualisasikan, bukan validitas skor itu sendiri.

Titik hook: SETELAH depthwise conv + BN + ReLU6 (container block.conv[1]),
BUKAN sebelum aktivasi -- titik yang SAMA PERSIS dipakai
compute_entropy_scores() di src/pruning.py untuk S_H, supaya peta fitur
yang divisualisasikan konsisten dengan definisi skor entropi yang
menyusun I(c).

Skrip ini TIDAK menarik kesimpulan apa pun -- hanya mencetak angka
(skor, statistik aktivasi, vmin/vmax) dan menyimpan gambar apa adanya.

Jalankan:
    venv/Scripts/python.exe scripts/46_visualisasi_peta_fitur.py
"""

import sys
import os
import json
import importlib.util

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets

from src.config import CFG
from src.model import load_checkpoint
from src.dataset import get_dataloaders, get_transforms
from src.pruning import (
    compute_l1_scores, compute_entropy_scores, normalize_min_max,
    compute_importance_scores, _get_prunable_layers,
)

# ----- Impor dinamis hitung_skor_gm_ekspansi_semua_layer dari skrip 43 -----
# (pola identik dengan 44/45/46-lama/48/49, src/pruning.py TIDAK diubah)
_JALUR_SKRIP_43 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "43_ablation_gm_expansion.py")
_spec_43 = importlib.util.spec_from_file_location("_skrip_43_ablation_gm_expansion", _JALUR_SKRIP_43)
_mod_43 = importlib.util.module_from_spec(_spec_43)
_spec_43.loader.exec_module(_mod_43)
hitung_skor_gm_ekspansi_semua_layer = _mod_43.hitung_skor_gm_ekspansi_semua_layer

PER_RASIO_PATH = CFG.OUTPUT_DIR / "multicriteria_per_rasio.json"
RASIO_ACUAN_BOBOT = "20%"  # titik operasi utama tesis -- lihat docstring
SPLIT_DIR = CFG.DATASET_DIR.parent / "dataset_split"
N_TOP = 5
N_BOTTOM = 5

# Empat kelas dipilih mewakili ketiga tanaman famili Poaceae (padi x2,
# jagung x1, tebu x1), campuran sakit dan sehat. File diambil urutan
# nama file (sorted, deterministik) -- BUKAN acak, supaya reproducible
# tanpa perlu seed.
KELAS_TERPILIH = ["Brown_Spot_Rice", "Common_Rust_Corn", "Rust_Sugarcane", "Healthy_Rice"]


def pilih_gambar_uji():
    """Ambil satu gambar (file pertama secara alfabetis) dari tiap kelas
    di KELAS_TERPILIH, dari subset VALIDASI (dataset_split/val/)."""
    val_dir = SPLIT_DIR / "val"
    hasil = []
    for kelas in KELAS_TERPILIH:
        folder = val_dir / kelas
        file_list = sorted(os.listdir(folder))
        if not file_list:
            raise RuntimeError(f"Folder {folder} kosong.")
        file_terpilih = folder / file_list[0]
        hasil.append((kelas, file_terpilih))
    return hasil


def cari_container_depthwise(block):
    """Cari container Sequential(depthwise_conv, BN, ReLU6) dalam satu
    InvertedResidual block -- titik yang SAMA dipakai
    compute_entropy_scores() untuk hook. Kriteria identifikasi disalin
    dari pola yang sama dipakai _get_prunable_layers()/apply_pruning()
    di src/pruning.py (groups==in_channels untuk depthwise), TIDAK
    menulis ulang logika penentuan block mana yang prunable -- hanya
    mengekstrak container dari block yang SUDAH diidentifikasi
    _get_prunable_layers()."""
    for layer in block.conv:
        if hasattr(layer, "__getitem__") and hasattr(layer, "__len__"):
            for sub in layer:
                if isinstance(sub, torch.nn.Conv2d) and sub.groups == sub.in_channels and sub.groups > 1:
                    return layer
    return None


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

    print("=" * 70)
    print("VISUALISASI PETA FITUR -- CHANNEL SKOR I(c) TERTINGGI vs TERENDAH")
    print("=" * 70)

    model, _ = load_checkpoint(baseline_path, device)
    model.eval()
    print(f"[INFO] Model dimuat: {baseline_path.name} (baseline terlatih, BUKAN pretrained mentah, BUKAN dipangkas)")

    # ----- TUGAS 2: hitung I(c) lewat fungsi pipeline yang sudah ada -----
    with open(PER_RASIO_PATH, encoding="utf-8") as f:
        w = json.load(f)["bobot_per_rasio"][RASIO_ACUAN_BOBOT]
    print(f"\n[INFO] Bobot I(c) dipakai (rasio acuan {RASIO_ACUAN_BOBOT}, titik operasi utama tesis): "
          f"w_L1={w['w_l1']:.4f}  w_GM={w['w_gm']:.4f}  w_Ent={w['w_entropi']:.4f}")

    dataloaders, _ = get_dataloaders(SPLIT_DIR)
    l1_scores = compute_l1_scores(model)
    gm_scores = hitung_skor_gm_ekspansi_semua_layer(model)
    entropy_scores = compute_entropy_scores(
        model, dataloaders["train"], device, samples_per_class=20, seed=CFG.SEED
    )
    importance = compute_importance_scores(
        l1_scores, gm_scores, entropy_scores, w1=w["w_l1"], w2=w["w_gm"], w3=w["w_entropi"]
    )

    # ----- TUGAS 3: pilih block tengah -----
    prunable = _get_prunable_layers(model)
    n_prunable = len(prunable)
    layer_idx_pilihan = n_prunable // 2
    block_idx_pilihan, dw_module_pilihan = prunable[layer_idx_pilihan]
    block_pilihan = model.features[block_idx_pilihan]
    n_channel_blok = dw_module_pilihan.out_channels

    print(f"\n[INFO] Block dipilih: layer_idx={layer_idx_pilihan} (block_idx={block_idx_pilihan} "
          f"di model.features), {n_channel_blok} channel intermediate.")
    print(f"[INFO] Alasan: index tengah dari {n_prunable} layer yang bisa dipangkas "
          f"({layer_idx_pilihan} dari 0-{n_prunable-1}) -- layer awal cenderung menangkap fitur generik "
          f"(tepi, warna), layer akhir terlalu abstrak untuk diinterpretasi mata langsung dari peta fitur.")

    # ----- Peringkat channel pada block terpilih -----
    skor_blok = importance[layer_idx_pilihan]
    urutan = torch.argsort(skor_blok, descending=True)
    channel_top = urutan[:N_TOP].tolist()
    channel_bottom = urutan[-N_BOTTOM:].tolist()
    # Urutkan channel_bottom dari skor terendah ke lebih tinggi (untuk tampilan)
    channel_bottom = sorted(channel_bottom, key=lambda c: skor_blok[c].item())
    channel_top = sorted(channel_top, key=lambda c: -skor_blok[c].item())

    print(f"\n[INFO] {N_TOP} channel skor I(c) TERTINGGI: " +
          ", ".join(f"#{c} ({skor_blok[c].item():.3f})" for c in channel_top))
    print(f"[INFO] {N_BOTTOM} channel skor I(c) TERENDAH: " +
          ", ".join(f"#{c} ({skor_blok[c].item():.3f})" for c in channel_bottom))

    # ----- TUGAS 4: pilih gambar uji -----
    gambar_uji = pilih_gambar_uji()
    print(f"\n[INFO] {len(gambar_uji)} gambar uji dipilih (dari dataset_split/val/):")
    for kelas, path in gambar_uji:
        print(f"    - {path.name}  (kelas: {kelas})")

    # ----- TUGAS 5: hook di conv[1] (depthwise+BN+ReLU6), forward pass per gambar -----
    container_dw = cari_container_depthwise(block_pilihan)
    aktivasi_tertangkap = {}

    def hook_fn(module, input, output):
        aktivasi_tertangkap["output"] = output.detach().cpu()

    handle = container_dw.register_forward_hook(hook_fn)

    transform_eval = get_transforms("test")  # resize + normalisasi saja, TANPA augmentasi
    peta_fitur_per_gambar = []  # list of (kelas, tensor_gambar_tampil, feature_map [C,H,W])

    with torch.no_grad():
        for kelas, path in gambar_uji:
            img_asli = Image.open(path).convert("RGB")
            img_tampil = img_asli.resize((CFG.IMG_SIZE, CFG.IMG_SIZE))  # untuk ditampilkan, TANPA normalisasi
            img_input = transform_eval(img_asli).unsqueeze(0)  # untuk model, DENGAN normalisasi

            _ = model(img_input)
            fmap = aktivasi_tertangkap["output"][0]  # (C, H, W)
            peta_fitur_per_gambar.append((kelas, img_tampil, fmap))

    handle.remove()

    # ----- TUGAS 6 (Perhatian normalisasi tampilan): vmin/vmax GLOBAL,
    # dari SELURUH channel yang ditampilkan (top + bottom, seluruh gambar) -----
    semua_nilai = []
    for _, _, fmap in peta_fitur_per_gambar:
        for c in channel_top + channel_bottom:
            semua_nilai.append(fmap[c].flatten())
    semua_nilai = torch.cat(semua_nilai)
    vmin_global = semua_nilai.min().item()
    vmax_global = semua_nilai.max().item()
    print(f"\n[INFO] vmin/vmax GLOBAL dipakai untuk SEMUA subplot (kedua grid): "
          f"vmin={vmin_global:.4f}  vmax={vmax_global:.4f}")

    # ----- Statistik pendukung per channel (rata-rata & std lintas 4 gambar) -----
    print(f"\n[INFO] Statistik aktivasi per channel (rata-rata dan std, digabung dari {len(gambar_uji)} gambar):")
    for label, daftar_channel in [("TERTINGGI", channel_top), ("TERENDAH", channel_bottom)]:
        print(f"  --- {label} ---")
        for c in daftar_channel:
            nilai_channel = torch.cat([fmap[c].flatten() for _, _, fmap in peta_fitur_per_gambar])
            print(f"    Channel #{c:4d}  I(c)={skor_blok[c].item():.3f}  "
                  f"mean={nilai_channel.mean().item():.4f}  std={nilai_channel.std().item():.4f}")

    # ----- TUGAS 7-9: buat dan simpan dua grid -----
    def buat_grid(daftar_channel, judul_grid, save_path):
        n_kolom = 1 + len(daftar_channel)
        n_baris = len(peta_fitur_per_gambar)
        fig, axes = plt.subplots(n_baris, n_kolom, figsize=(2.2 * n_kolom, 2.4 * n_baris))

        for i, (kelas, img_tampil, fmap) in enumerate(peta_fitur_per_gambar):
            ax_asli = axes[i, 0]
            ax_asli.imshow(img_tampil)
            ax_asli.set_xticks([])
            ax_asli.set_yticks([])
            ax_asli.set_ylabel(kelas, fontsize=9, rotation=90)
            if i == 0:
                ax_asli.set_title("Gambar asli", fontsize=10)

            for j, c in enumerate(daftar_channel):
                ax = axes[i, j + 1]
                im = ax.imshow(fmap[c].numpy(), cmap="viridis", vmin=vmin_global, vmax=vmax_global)
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(f"#{c}\nI(c)={skor_blok[c].item():.3f}", fontsize=9)

        fig.suptitle(judul_grid, fontsize=12, fontweight="bold")
        cbar_ax = fig.add_axes((0.92, 0.15, 0.015, 0.7))
        fig.colorbar(im, cax=cbar_ax)
        fig.tight_layout(rect=(0, 0, 0.90, 0.95))
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Grid disimpan: {save_path}")

    buat_grid(
        channel_top,
        f"Peta Fitur -- 5 Channel Skor I(c) TERTINGGI (block_idx={block_idx_pilihan})",
        CFG.OUTPUT_DIR / "visualisasi_peta_fitur_skor_tinggi.png",
    )
    buat_grid(
        channel_bottom,
        f"Peta Fitur -- 5 Channel Skor I(c) TERENDAH (block_idx={block_idx_pilihan})",
        CFG.OUTPUT_DIR / "visualisasi_peta_fitur_skor_rendah.png",
    )

    print(f"\n{'=' * 70}")
    print("[SELESAI] Data dan gambar disimpan apa adanya, tanpa kesimpulan.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
