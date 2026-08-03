# Klasifikasi Penyakit Daun Tanaman Famili Poaceae
## Pruning Terstruktur Berbasis Skoring Multi-Kriteria pada MobileNetV2

**Nurul Umayah Hafilda** | Universitas Hasanuddin | 2026

---

## Struktur Proyek

```
poaceae_project/
├── src/                          # Modul Python (library)
│   ├── __init__.py
│   ├── config.py                 # Konfigurasi terpusat
│   ├── dataset.py                # Loading, praproses, split, DataLoader
│   ├── model.py                  # MobileNetV2, training, evaluasi
│   ├── pruning.py                # Sistem penskoran multi-kriteria + pruning
│   └── visualize.py              # Confusion matrix, grafik, tabel
├── scripts/                      # Script Python per tahap (jalankan berurutan)
│   ├── 01_prepare_dataset.py     # Persiapan dan split dataset
│   ├── 02_train_baseline.py      # Training baseline MobileNetV2
│   ├── 03_ablation_study.py      # Ablation study 3 kriteria tunggal
│   ├── 04_pruning_multicriteria.py  # Pruning multi-kriteria 7 rasio
│   ├── 05_generate_report.py     # Laporan dan visualisasi
│   └── 06_deploy_flask.py        # Web deployment Flask
├── dataset/                      # Folder dataset (9 subfolder)
├── checkpoints/                  # Model checkpoint (.pth)
├── outputs/                      # Gambar, grafik, hasil JSON
├── requirements.txt
└── README.md
```

## Cara Memulai

### 1. Setup Environment
```bash
cd D:\S2 UMAYAH\TESIS\Poaceae
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Edit Config
Buka `src/config.py`, ubah `ROOT_DIR` sesuai path di laptop kamu.

### 3. Siapkan Dataset
Letakkan gambar ke 9 subfolder di `dataset/`:
- `Leaf_Blast/`, `Brown_Spot/`, `Healthy_Rice/`
- `Rust_Sugarcane/`, `Mosaic_Sugarcane/`, `Healthy_Sugarcane/`
- `Common_Rust_Corn/`, `Gray_Leaf_Spot_Corn/`, `Healthy_Corn/`

### 4. Jalankan Script Berurutan
```bash
python scripts/01_prepare_dataset.py
python scripts/02_train_baseline.py
python scripts/03_ablation_study.py
python scripts/04_pruning_multicriteria.py --load_weights
python scripts/05_generate_report.py
python scripts/06_deploy_flask.py
```

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
python scripts/06_deploy_flask.py --checkpoint checkpoints/multicriteria_30pct_10ep.pth
```
