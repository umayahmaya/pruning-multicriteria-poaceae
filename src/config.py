"""
config.py - Konfigurasi Terpusat Proyek Tesis
"""

import sys

# Paksa stdout ke UTF-8 agar karakter non-ASCII (mis. box-drawing di
# print statement) tidak gagal encode di konsol Windows berdefault cp1252.
# Semua skrip mengimpor config, jadi perbaikan ini berlaku menyeluruh.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
from pathlib import Path


class CFG:
    # ==================== PATH ====================
    ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DATASET_DIR = ROOT_DIR / "dataset"
    OUTPUT_DIR = ROOT_DIR / "outputs"
    CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
    LOG_DIR = ROOT_DIR / "logs"

    # ==================== DATASET ====================
    CLASS_NAMES = [
        "Brown_Spot_Rice",
        "Common_Rust_Corn",
        "Gray_Leaf_Spot_Corn",
        "Healthy_Corn",
        "Healthy_Rice",
        "Healthy_Sugarcane",
        "Leaf_Blast_Rice",
        "Mosaic_Sugarcane",
        "Rust_Sugarcane",
    ]
    NUM_CLASSES = len(CLASS_NAMES)

    PLANT_MAP = {
        "Brown_Spot_Rice": "Padi",
        "Common_Rust_Corn": "Jagung",
        "Gray_Leaf_Spot_Corn": "Jagung",
        "Healthy_Corn": "Jagung",
        "Healthy_Rice": "Padi",
        "Healthy_Sugarcane": "Tebu",
        "Leaf_Blast_Rice": "Padi",
        "Mosaic_Sugarcane": "Tebu",
        "Rust_Sugarcane": "Tebu",
    }

    TRAIN_RATIO = 0.70
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

    # ==================== MODEL ====================
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_WORKERS = 2

    # ==================== TRAINING BASELINE ====================
    BASELINE_EPOCHS = 30
    BASELINE_LR = 1e-3
    OPTIMIZER = "Adam"
    WEIGHT_DECAY = 1e-4

    # ==================== FINE-TUNING ====================
    # Harus setara dengan BASELINE_EPOCHS agar perbandingan akurasi
    # baseline vs model hasil pruning tetap adil.
    FINETUNE_EPOCHS = 30
    FINETUNE_LR = 1e-4

    # ==================== PRUNING ====================
    PRUNING_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    # ==================== EVALUASI ====================
    INFERENCE_RUNS = 300
    WARMUP_RUNS = 30

    # ==================== SEED ====================
    SEED = 42
    DEVICE = "cuda"


for d in [CFG.OUTPUT_DIR, CFG.CHECKPOINT_DIR, CFG.LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)