# Klasifikasi Penyakit Daun Tanaman Famili Poaceae
## Pruning Terstruktur Berbasis Skoring Multi-Kriteria pada MobileNetV2

**Nurul Umayah Hafilda** | Universitas Hasanuddin | 2026

---

## Struktur Proyek

```
Pra Penelitian 2/
├── src/                                            # Modul Python (library)
│   ├── __init__.py
│   ├── config.py                                   # Konfigurasi terpusat
│   ├── dataset.py                                  # Loading, praproses, split, DataLoader
│   ├── model.py                                    # MobileNetV2, training, evaluasi, load/save checkpoint
│   ├── pruning.py                                  # Skoring multi-kriteria I(c) + pruning terstruktur
│   └── visualize.py                                # Confusion matrix, grafik, tabel
├── scripts/                                        # Skrip eksekusi (jalankan dari root proyek)
│   ├── 01_prepare_dataset.py                       # Persiapan dan split dataset 70/15/15 stratified
│   ├── 02_train_baseline.py                        # Training baseline MobileNetV2
│   ├── 03_ablation_study.py                        # Ablation 3 kriteria tunggal x 7 rasio (bobot dari akurasi test -- lihat catatan di "Urutan Menjalankan")
│   ├── 04_pruning_multicriteria.py                 # Pruning multi-kriteria 7 rasio (--load_weights atau --w1/--w2/--w3 manual)
│   ├── 05_generate_report.py                       # Cetak tabel + grafik dari outputs/tabel_hasil_lengkap.json
│   ├── 06_deploy_flask.py                          # Web deployment Flask (antarmuka prediksi)
│   ├── 07_recompute_weights_val.py                 # Hitung ulang w1/w2/w3 dari akurasi VALIDASI (koreksi kebocoran data di 03)
│   ├── 08_compare_channel_selection.py             # Korelasi Spearman I(c) vs kriteria tunggal, bobot lama vs baru, semua rasio
│   ├── 10_smoke_test_entropy.py                    # Uji langsung compute_entropy_scores() (determinisme, jumlah sampel, dsb.)
│   ├── 11_ablation_entropy_traincal.py             # Ablation entropi ulang dengan kalibrasi TRAIN (koreksi kebocoran data)
│   ├── 12_multicriteria_valweights.py              # Pruning multi-kriteria ulang, 7 rasio, memakai bobot val-derived
│   ├── 13_remeasure_inference.py                   # Pengukuran ulang waktu inferensi presisi (1 thread, 300 run, median+std)
│   ├── 14_eval_multicriteria_valweights_on_val.py  # Evaluasi 7 checkpoint valweights pada subset val
│   ├── 15_multiseed_validation.py                  # Validasi multi-seed (42/123/2024): baseline, rasio 20%, rasio 60%
│   ├── 16_multiseed_remaining_ratios.py            # Validasi multi-seed rasio sisanya: 10/30/40/50/70% (seed 123, 2024)
│   ├── 17_generate_final_table.py                  # Evaluasi ulang 29 checkpoint, satu protokol seragam -> tabel_hasil_lengkap.json
│   ├── 18_prepare_demo_images.py                   # Susun citra demonstrasi (top-3 benar/kelas + daftar salah prediksi)
│   └── test_system.py                              # Pengujian otomatis 7 tahap (dependensi s/d endpoint Flask)
├── dataset/                                         # TIDAK disertakan di repositori -- lihat "Sumber Dataset"
├── dataset_split/                                   # TIDAK disertakan di repositori -- lihat "Sumber Dataset"
├── checkpoints/                                     # TIDAK disertakan di repositori -- lihat "Sumber Dataset"
├── logs/                                            # Catatan proses training per epoch
├── outputs/                                         # Tabel, grafik, hasil JSON -- lihat outputs/README_OUTPUTS.md
├── requirements.txt
├── CLAUDE.md
└── README.md
```

## Sumber Dataset

`dataset/`, `dataset_split/`, dan `checkpoints/` sengaja **tidak disertakan** di repositori ini karena ukurannya (lihat CLAUDE.md Bagian 11). Dataset disusun dari tiga sumber publik, masing-masing menyumbang tiga dari sembilan kelas, lalu diseimbangkan ke 129 citra per kelas:

- **Padi** (Brown_Spot_Rice, Healthy_Rice, Leaf_Blast_Rice) -- Hasan (2023), DOI [10.17632/hx6f852hw4.2](https://doi.org/10.17632/hx6f852hw4.2)
- **Jagung** (Common_Rust_Corn, Gray_Leaf_Spot_Corn, Healthy_Corn) -- Ahmad (2025), DOI [10.17632/vy629dngm8.1](https://doi.org/10.17632/vy629dngm8.1)
- **Tebu** (Healthy_Sugarcane, Mosaic_Sugarcane, Rust_Sugarcane) -- Daphal dan Koli (2022), DOI [10.17632/9424skmnrk.1](https://doi.org/10.17632/9424skmnrk.1)

Total 1.161 citra (129 x 9 kelas), dibagi 70/15/15 stratified dengan seed 42 (lihat `scripts/01_prepare_dataset.py`).

## Hasil Utama

| | Baseline | Multi-Kriteria 20% |
|---|---|---|
| Akurasi test | 96,00% | 96,00% |
| Jumlah parameter | 2.235.401 | 1.874.851 (turun 16,13%) |
| FLOPs | 326.218.240 | 270.836.921 (turun 16,98%) |

Rasio 20% adalah titik operasi terpilih. Angka 96,00% pada tabel di atas adalah hasil satu tarikan (seed 42). Validasi multi-seed (seed 42, 123, 2024) pada konfigurasi yang sama menghasilkan rerata akurasi 94,67%, rentang 93,14% sampai 96,00%, dan simpangan baku 1,17 poin persentase (lihat `outputs/multiseed_results.json`) -- selisihnya terhadap baseline TIDAK boleh diklaim sebagai peningkatan tanpa mempertimbangkan sebaran ini. Parameter dan FLOPs tetap lebih kecil dari baseline pada rasio ini, tidak bergantung pada seed. Rincian lengkap kedelapan metrik (akurasi, precision, recall, F1, jumlah parameter, ukuran MB, FLOPs, waktu inferensi) untuk seluruh 4 skenario x 7 rasio ada di `outputs/tabel_hasil_lengkap.json`. Berkas mana yang sah dikutip untuk bab hasil, dan mana yang arsip metodologi lama, dijelaskan di `outputs/README_OUTPUTS.md`.

## Cara Memulai

### 1. Setup Environment
```bash
cd "D:\S2 UMAYAH\TESIS\Pra Penelitian 2"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Edit Config
Buka `src/config.py`, ubah `ROOT_DIR` sesuai path di laptop kamu.

### 3. Siapkan Dataset
Letakkan gambar ke 9 subfolder di `dataset/` (lihat "Sumber Dataset" di atas untuk sumber dan DOI):
- `Leaf_Blast_Rice/`, `Brown_Spot_Rice/`, `Healthy_Rice/`
- `Rust_Sugarcane/`, `Mosaic_Sugarcane/`, `Healthy_Sugarcane/`
- `Common_Rust_Corn/`, `Gray_Leaf_Spot_Corn/`, `Healthy_Corn/`

### 4. Jalankan Skrip
Lihat "Urutan Menjalankan" di bawah.

## Urutan Menjalankan

Alur utama, skrip 01 sampai 06, dijalankan berurutan:
```bash
venv/Scripts/python.exe scripts/01_prepare_dataset.py
venv/Scripts/python.exe scripts/02_train_baseline.py
venv/Scripts/python.exe scripts/03_ablation_study.py
venv/Scripts/python.exe scripts/04_pruning_multicriteria.py --load_weights
venv/Scripts/python.exe scripts/05_generate_report.py
venv/Scripts/python.exe scripts/06_deploy_flask.py
```

Skrip 07 ke atas **bukan** bagian dari alur utama ini. Skrip 07, 11, dan 12 adalah **koreksi metodologis** (memperbaiki sumber bobot ablation dari akurasi test menjadi akurasi validasi, dan kalibrasi entropi dari val menjadi train -- lihat CLAUDE.md Bagian 4 dan 9). Skrip 08, 10, 13 sampai 18 adalah **analisis lanjutan** (korelasi seleksi channel, smoke test, pengukuran ulang waktu inferensi, validasi multi-seed, tabel hasil akhir, dan penyiapan citra demonstrasi). Hasil yang sah dikutip untuk bab hasil berasal dari rantai skrip yang telah dikoreksi (07 → 11/12 → 17), bukan dari keluaran default skrip 03/04 saja -- lihat `outputs/README_OUTPUTS.md`.

## Opsi Command Line

### 01_prepare_dataset.py
```bash
python scripts/01_prepare_dataset.py --max_per_class 400     # Batasi per kelas
python scripts/01_prepare_dataset.py --force_resplit          # Split ulang
```

### 02_train_baseline.py
```bash
python scripts/02_train_baseline.py --epochs 50 --lr 0.0005
python scripts/02_train_baseline.py --resume                  # Lanjutkan training
```

### 03_ablation_study.py
```bash
python scripts/03_ablation_study.py --ratio 0.3               # Satu rasio
python scripts/03_ablation_study.py --all_ratios               # Semua rasio
python scripts/03_ablation_study.py --epochs 5                 # Epoch ablation
```

### 04_pruning_multicriteria.py
```bash
python scripts/04_pruning_multicriteria.py --load_weights      # Bobot dari ablation
python scripts/04_pruning_multicriteria.py --ratios 0.2 0.3 0.4  # Rasio tertentu
python scripts/04_pruning_multicriteria.py --w1 0.35 --w2 0.34 --w3 0.31  # Bobot manual
```

### 06_deploy_flask.py
```bash
python scripts/06_deploy_flask.py --port 8080
python scripts/06_deploy_flask.py --checkpoint checkpoints/baseline_9class.pth
```
