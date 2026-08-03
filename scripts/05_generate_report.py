"""
05_generate_report.py
Menghasilkan laporan lengkap dari outputs/tabel_hasil_lengkap.json (hasil
evaluasi ulang 29 checkpoint dengan protokol seragam -- lihat
17_generate_final_table.py): tabel 8 metrik untuk baseline dan keempat
skenario (L1 saja, BN saja, Entropi saja, Multi-Kriteria) x 7 rasio, blok
metadata sumber bobot/kalibrasi/protokol latensi, serta grafik kurva
rasio pemangkasan terhadap akurasi untuk keempat skenario dalam satu
gambar.

Jalankan:
    python scripts/05_generate_report.py
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")  # backend non-interaktif, supaya tidak memblokir di terminal

from src.config import CFG
from src.visualize import plot_pruning_comparison

SCENARIO_LABELS = {
    "l1": "L1 saja",
    "bn": "BN saja",
    "entropy": "Entropi saja",
    "multicriteria": "Multi-Kriteria",
}


def main():
    print("=" * 60)
    print("TAHAP 5: LAPORAN DAN VISUALISASI LENGKAP")
    print("=" * 60)

    table_path = CFG.OUTPUT_DIR / "tabel_hasil_lengkap.json"
    if not table_path.exists():
        print(f"\n[ERROR] {table_path} tidak ditemukan.")
        print("Jalankan 17_generate_final_table.py terlebih dahulu.")
        return

    with open(table_path) as f:
        table_data = json.load(f)
    print(f"[OK] {table_path.name} dimuat (status: {table_data.get('status')})")

    metadata = table_data["metadata"]
    results = table_data["results"]

    # Bobot -- wajib lengkap, berhenti dengan pesan jelas kalau tidak (bukan
    # diam-diam memakai NaN)
    weights_used = metadata.get("weights_used")
    if not weights_used or not all(k in weights_used for k in ("w1_l1", "w2_bn", "w3_entropy")):
        print("\n[ERROR] Metadata 'weights_used' tidak lengkap di tabel_hasil_lengkap.json "
              "(butuh kunci w1_l1, w2_bn, w3_entropy).")
        print("Jalankan 17_generate_final_table.py ulang setelah memastikan "
              "outputs/ablation_val_results.json berisi optimal_weights_val yang valid.")
        return
    w1, w2, w3 = weights_used["w1_l1"], weights_used["w2_bn"], weights_used["w3_entropy"]

    # Blok metadata
    print(f"\n{'=' * 80}")
    print("METADATA SUMBER DATA")
    print(f"{'=' * 80}")
    print(f"  Sumber bobot   : {metadata['weights_source']}")
    print(f"    w1(L1)={w1:.4f}  w2(BN)={w2:.4f}  w3(H)={w3:.4f}")
    ec = metadata["entropy_calibration"]
    print(f"  Kalibrasi entropi : split={ec['source_split']}, "
          f"{ec['samples_per_class']} gambar/kelas x {ec['num_classes']} kelas "
          f"({ec['total_calibration_images']} total), seed={ec['seed']}")
    print(f"    Transform: {ec['transform']}")
    ip = metadata["inference_protocol"]
    print(f"  Protokol latensi  : {ip['runs']} runs, {ip['warmup']} warmup, "
          f"{ip['num_threads']} thread, {ip['device']}, lapor {ip['statistic_reported']}")
    ep = metadata["epochs"]
    print(f"  Epoch             : baseline={ep['baseline']}, fine-tune={ep['finetune_pruned']}")
    print(f"  Subset evaluasi   : {metadata['eval_subset']}")
    print(f"{'=' * 80}")

    # Tabel per skenario, 8 metrik lengkap
    print(f"\n{'=' * 80}")
    print(f"TABEL RINGKASAN HASIL PENELITIAN")
    print(f"Model Klasifikasi Penyakit Daun Tanaman Famili Poaceae")
    print(f"Pruning Terstruktur Berbasis Skoring Multi-Kriteria pada MobileNetV2")
    print(f"{'=' * 80}")

    base = results["baseline"]
    print(f"\n{'Skenario':<16} {'Rasio':<7} {'Akurasi':<9} {'Prec':<8} {'Recall':<8} "
          f"{'F1':<8} {'Param':<12} {'MB':<7} {'FLOPs(M)':<10} {'Inf(ms)'}")
    print(f"{'-' * 100}")
    print(f"{'Baseline':<16} {'-':<7} {base['accuracy']*100:>6.2f}%  "
          f"{base['precision']:.4f}  {base['recall']:.4f}  {base['f1_score']:.4f}  "
          f"{base['num_params']:>10,}  {base['model_size_mb']:>5.2f}  "
          f"{base['flops']/1e6:>8.3f}  {base['inference_ms']:>6.2f}")

    for scenario_key, scenario_label in SCENARIO_LABELS.items():
        for ratio_key, m in results[scenario_key].items():
            print(f"{scenario_label:<16} {ratio_key:<7} {m['accuracy']*100:>6.2f}%  "
                  f"{m['precision']:.4f}  {m['recall']:.4f}  {m['f1_score']:.4f}  "
                  f"{m['num_params']:>10,}  {m['model_size_mb']:>5.2f}  "
                  f"{m['flops']/1e6:>8.3f}  {m['inference_ms']:>6.2f}")
    print(f"{'=' * 80}")

    # Rasio optimal (dari skenario multi-kriteria, sesuai kontribusi utama tesis)
    multi = results["multicriteria"]
    best_ratio = max(multi, key=lambda k: multi[k]["accuracy"])
    best = multi[best_ratio]

    print(f"\n{'=' * 60}")
    print(f"PERBANDINGAN BASELINE vs MULTI-KRITERIA OPTIMAL ({best_ratio})")
    print(f"{'=' * 60}")

    def pct_change(new, old):
        if old == 0:
            return 0
        return ((new - old) / old) * 100

    print(f"  {'Metrik':<20} {'Baseline':<14} {'Pruned':<14} {'Perubahan'}")
    print(f"  {'-' * 56}")
    print(f"  {'Akurasi':<20} {base['accuracy']*100:>8.2f}%    "
          f"{best['accuracy']*100:>8.2f}%    "
          f"{pct_change(best['accuracy'], base['accuracy']):>+.2f}%")
    print(f"  {'F1-Score':<20} {base['f1_score']:>10.4f}  "
          f"{best['f1_score']:>10.4f}  "
          f"{pct_change(best['f1_score'], base['f1_score']):>+.2f}%")
    print(f"  {'Parameter':<20} {base['num_params']:>10,}  "
          f"{best['num_params']:>10,}  "
          f"{pct_change(best['num_params'], base['num_params']):>+.2f}%")
    print(f"  {'Ukuran (MB)':<20} {base['model_size_mb']:>10.2f}  "
          f"{best['model_size_mb']:>10.2f}  "
          f"{pct_change(best['model_size_mb'], base['model_size_mb']):>+.2f}%")
    print(f"  {'FLOPs (M)':<20} {base['flops']/1e6:>10.3f}  "
          f"{best['flops']/1e6:>10.3f}  "
          f"{pct_change(best['flops'], base['flops']):>+.2f}%")
    print(f"  {'Inferensi (ms)':<20} {base['inference_ms']:>10.2f}  "
          f"{best['inference_ms']:>10.2f}  "
          f"{pct_change(best['inference_ms'], base['inference_ms']):>+.2f}%")
    print(f"{'=' * 60}")

    # Grafik: kurva rasio pemangkasan vs akurasi, keempat skenario + baseline
    plot_data = {"Baseline": base}
    for scenario_key, scenario_label in SCENARIO_LABELS.items():
        plot_data[scenario_label] = results[scenario_key]

    chart_path = CFG.OUTPUT_DIR / "kurva_kompresi_akurasi.png"
    plot_pruning_comparison(plot_data, metric="accuracy", save_path=chart_path, show=False)

    print(f"\n[SELESAI] Laporan selesai.")
    print(f"File output tersimpan di: {CFG.OUTPUT_DIR}")
    print(f"Grafik disimpan di: {chart_path}")


if __name__ == "__main__":
    main()
