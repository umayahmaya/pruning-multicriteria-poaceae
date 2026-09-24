"""
47_kurva_kompresi_per_rasio.py
Kurva kompresi-akurasi untuk FORMULA AKTIF (L1 + GM-Ekspansi + Entropi,
bobot per rasio, CLAUDE.md Bagian 2) -- padanan kurva_kompresi_akurasi.png
lama (yang memakai L1/BN/Entropi/Multi-Kriteria formula AWAL, kini di
outputs/archive/) tapi dengan kriteria dan hasil yang sesuai formula
sekarang.

Empat garis: L1 saja, GM-Ekspansi saja, Entropi saja, Multi-Kriteria
(Aktif). Baseline sebagai garis putus-putus horizontal.

Sumber data (TIDAK dihitung ulang, murni dibaca dan diplot):
    - L1, GM-Ekspansi, Entropi (rata-rata 3 seed 42/123/2024, DENGAN
      error bar std)                  : outputs/multiseed_single_criteria_results.json
    - Multi-Kriteria (rata-rata 3 seed 42/123/2024, DENGAN error bar std)
      dan Baseline (rata-rata 3 seed, DENGAN error bar std)
                                       : outputs/multiseed_per_rasio_results.json

Diperbarui 2026-09-24: versi SEBELUMNYA skrip ini memakai L1/GM-Ekspansi/
Entropi seed 42 saja (dari tabel_hasil_lengkap.json dan
ablation_gm_expansion.json), dengan catatan kaki bahwa ketiganya "belum
multi-seed, CLAUDE.md Bagian 8 butir 10 masih terbuka". Itu SUDAH BASI
sejak butir 10 selesai (48_multiseed_single_criteria.py, 42 run, 31,25
jam komputasi, SEMUA 7 rasio x 3 kriteria x 3 seed) -- keempat garis di
chart ini SEKARANG seluruhnya 3-seed dengan error bar, tidak ada lagi
asimetri data.

Palet warna: 4 slot pertama palet kategorikal skill dataviz (biru, oranye,
aqua, kuning; urutan divalidasi CVD-safe adjacent-pairlist pada
references/palette.md skill dataviz) -- BUKAN direplikasi dari
plot_pruning_comparison() (src/visualize.py, dipakai kurva_kompresi_akurasi.png
LAMA, warnanya tab10 default matplotlib dan LABEL kriterianya untuk formula
awal) supaya tidak mengubah tampilan chart lama itu. Ditambah marker dan
linestyle berbeda per seri (bukan cuma warna) supaya tetap terbedakan kalau
tesis dicetak hitam-putih (CLAUDE.md Bagian 7: dokumen tesis format B5).

Jalankan:
    venv/Scripts/python.exe scripts/47_kurva_kompresi_per_rasio.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt

from src.config import CFG

MULTISEED_SINGLE_PATH = CFG.OUTPUT_DIR / "multiseed_single_criteria_results.json"
MULTISEED_PER_RASIO_PATH = CFG.OUTPUT_DIR / "multiseed_per_rasio_results.json"
SAVE_PATH = CFG.OUTPUT_DIR / "kurva_kompresi_akurasi_per_rasio.png"

# Palet kategorikal tervalidasi (skill dataviz, references/palette.md,
# slot 1-4: biru/oranye/aqua/kuning) -- diverifikasi lewat dokumentasi
# hasil validate_palette.js yang sudah dipublikasikan skill (adjacent
# pairlist, konteks line chart), bukan ditebak.
WARNA = {
    "Multi-Kriteria (Aktif)": "#2a78d6",   # slot 1 biru -- kontribusi utama tesis
    "L1 saja": "#eb6834",                   # slot 2 oranye
    "GM-Ekspansi saja": "#1baf7a",          # slot 3 aqua
    "Entropi saja": "#eda100",              # slot 4 kuning
}
MARKER = {
    "Multi-Kriteria (Aktif)": "o",
    "L1 saja": "s",
    "GM-Ekspansi saja": "^",
    "Entropi saja": "D",
}
LINESTYLE = {
    "Multi-Kriteria (Aktif)": "-",
    "L1 saja": "--",
    "GM-Ekspansi saja": "-.",
    "Entropi saja": ":",
}
# Kunci kriteria di multiseed_single_criteria_results.json -> label chart
KRITERIA_KEY = {
    "L1 saja": "l1",
    "GM-Ekspansi saja": "gm",
    "Entropi saja": "entropy",
}


def main():
    with open(MULTISEED_SINGLE_PATH, encoding="utf-8") as f:
        single = json.load(f)
    with open(MULTISEED_PER_RASIO_PATH, encoding="utf-8") as f:
        multiseed = json.load(f)

    ratios = CFG.PRUNING_RATIOS  # [0.1, ..., 0.7]
    ratio_labels = [f"{int(r*100)}%" for r in ratios]

    single_mean, single_std = {}, {}
    for label, kunci in KRITERIA_KEY.items():
        mean_list, std_list = [], []
        for r in ratios:
            entry = single["kriteria"][kunci][f"{int(r*100)}%"]
            mean_list.append(entry["mean_accuracy"] * 100)
            std_list.append(entry["std_accuracy"] * 100)
        single_mean[label] = mean_list
        single_std[label] = std_list

    multi_mean, multi_std = [], []
    for r in ratios:
        entry = multiseed["configs"][f"multicriteria_per_rasio_{int(r*100)}pct"]
        multi_mean.append(entry["mean_accuracy"] * 100)
        multi_std.append(entry["std_accuracy"] * 100)

    baseline = multiseed["baseline_dipakai_ulang"]
    baseline_mean = baseline["mean_accuracy"] * 100
    baseline_std = baseline["std_accuracy"] * 100

    fig, ax = plt.subplots(figsize=(12, 6.5))

    # Baseline: garis putus-putus + pita simpangan baku (3 seed)
    ax.axhline(y=baseline_mean, color="dimgray", linestyle="--", linewidth=1.5,
               label=f"Baseline ({baseline_mean:.2f}% ± {baseline_std:.2f}pp, 3 seed)",
               alpha=0.8, zorder=1)
    ax.axhspan(baseline_mean - baseline_std, baseline_mean + baseline_std,
               color="dimgray", alpha=0.08, zorder=0)

    # Tiga kriteria tunggal -- rata-rata 3 seed DENGAN error bar std
    for label in ["L1 saja", "GM-Ekspansi saja", "Entropi saja"]:
        ax.errorbar(ratio_labels, single_mean[label], yerr=single_std[label],
                    linestyle=LINESTYLE[label], marker=MARKER[label],
                    color=WARNA[label], label=f"{label} (rata-rata ± std, 3 seed)",
                    markersize=7, linewidth=2, capsize=4, capthick=1.3, zorder=2)

    # Multi-Kriteria (Aktif) -- rata-rata 3 seed DENGAN error bar std
    ax.errorbar(ratio_labels, multi_mean, yerr=multi_std,
                linestyle=LINESTYLE["Multi-Kriteria (Aktif)"],
                marker=MARKER["Multi-Kriteria (Aktif)"], color=WARNA["Multi-Kriteria (Aktif)"],
                label="Multi-Kriteria Aktif (rata-rata ± std, 3 seed)",
                markersize=8, linewidth=2.8, capsize=4, capthick=1.5, zorder=3)

    ax.set_xlabel("Rasio Pemangkasan", fontsize=12)
    ax.set_ylabel("Akurasi Test (%)", fontsize=12)
    ax.set_title("Kurva Kompresi-Akurasi -- Formula Aktif\n"
                  "(L1 + GM-Ekspansi + Entropi, Bobot Per-Rasio)",
                  fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    fig.text(0.5, 0.01,
              "Catatan: seluruh empat garis (L1/GM-Ekspansi/Entropi/Multi-Kriteria) dan "
              "Baseline sudah divalidasi 3 seed (42/123/2024) dengan error bar simpangan baku.",
              ha="center", fontsize=8, style="italic", color="dimgray")

    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(SAVE_PATH, dpi=300, bbox_inches="tight")
    print(f"[INFO] Kurva disimpan: {SAVE_PATH}")


if __name__ == "__main__":
    main()
