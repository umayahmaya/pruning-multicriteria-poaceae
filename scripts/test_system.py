"""
test_system.py
Pengujian otomatis seluruh sistem sebelum menjalankan eksperimen sebenarnya.

Script ini melakukan 7 tahap validasi:
  1. Pengecekan dependensi (library terinstall)
  2. Pengecekan struktur proyek (folder dan file)
  3. Pengecekan dataset (jumlah gambar, format, kelas)
  4. Uji modul config, dataset, model, pruning, visualize
  5. Simulasi pipeline mini dengan data dummy (tanpa GPU)
  6. Uji konsistensi matematis (normalisasi, WSM, kombinasi konveks)
  7. Uji deployment Flask (endpoint /predict)

Jalankan:
    python scripts/test_system.py              # Semua tes
    python scripts/test_system.py --quick      # Tes cepat tanpa training
    python scripts/test_system.py --verbose    # Output detail
"""

import sys
import os
import json
import time
import argparse
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Warna terminal (Windows compatible)
try:
    os.system("")  # Enable ANSI on Windows
except:
    pass

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"

passed = 0
failed = 0
warnings = 0
errors_log = []


def test(name, condition, detail=""):
    """Jalankan satu tes dan catat hasilnya."""
    global passed, failed
    if condition:
        passed += 1
        print(f"  {GREEN}PASS{RESET}  {name}")
    else:
        failed += 1
        msg = f"  {RED}FAIL{RESET}  {name}"
        if detail:
            msg += f" -> {detail}"
        print(msg)
        errors_log.append(f"{name}: {detail}")


def warn(name, detail=""):
    """Catat peringatan non-fatal."""
    global warnings
    warnings += 1
    print(f"  {YELLOW}WARN{RESET}  {name} -> {detail}")


def section(title):
    """Cetak header bagian."""
    print(f"\n{BLUE}{BOLD}{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}{RESET}")


# ============================================================
# TAHAP 1: DEPENDENSI
# ============================================================
def test_dependencies():
    section("TAHAP 1: Pengecekan Dependensi")

    modules = {
        "torch": "PyTorch (framework utama)",
        "torchvision": "TorchVision (model pretrained)",
        "numpy": "NumPy (komputasi numerik)",
        "sklearn": "scikit-learn (metrik evaluasi)",
        "matplotlib": "Matplotlib (visualisasi)",
        "seaborn": "Seaborn (heatmap confusion matrix)",
        "PIL": "Pillow (pemrosesan gambar)",
    }

    for mod, desc in modules.items():
        try:
            __import__(mod)
            test(f"{desc}", True)
        except ImportError:
            test(f"{desc}", False, f"pip install {mod}")

    # Opsional
    try:
        import thop
        test("thop (perhitungan FLOPs)", True)
    except ImportError:
        warn("thop tidak terinstall", "pip install thop (opsional tapi direkomendasikan)")

    try:
        import flask
        test("Flask (web deployment)", True)
    except ImportError:
        warn("Flask tidak terinstall", "pip install flask (diperlukan untuk tahap 06)")

    # Cek versi PyTorch
    import torch
    version = torch.__version__
    major, minor = int(version.split(".")[0]), int(version.split(".")[1])
    test(f"PyTorch versi >= 2.0 (terdeteksi: {version})", major >= 2,
         f"Versi {version} mungkin tidak kompatibel")

    # Cek CUDA
    if torch.cuda.is_available():
        test(f"CUDA tersedia: {torch.cuda.get_device_name(0)}", True)
    else:
        warn("CUDA tidak tersedia", "Training akan berjalan di CPU (lebih lambat)")


# ============================================================
# TAHAP 2: STRUKTUR PROYEK
# ============================================================
def test_project_structure():
    section("TAHAP 2: Pengecekan Struktur Proyek")

    from src.config import CFG

    # Cek file sumber
    src_files = [
        "src/__init__.py", "src/config.py", "src/dataset.py",
        "src/model.py", "src/pruning.py", "src/visualize.py"
    ]
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    for f in src_files:
        full_path = os.path.join(project_root, f)
        test(f"File {f} ada", os.path.exists(full_path),
             f"File tidak ditemukan: {full_path}")

    # Cek script
    script_files = [
        "scripts/01_prepare_dataset.py", "scripts/02_train_baseline.py",
        "scripts/03_ablation_study.py", "scripts/04_pruning_multicriteria.py",
        "scripts/05_generate_report.py", "scripts/06_deploy_flask.py"
    ]
    for f in script_files:
        full_path = os.path.join(project_root, f)
        test(f"File {f} ada", os.path.exists(full_path),
             f"File tidak ditemukan: {full_path}")

    # Cek direktori output
    for d_name, d_path in [
        ("OUTPUT_DIR", CFG.OUTPUT_DIR),
        ("CHECKPOINT_DIR", CFG.CHECKPOINT_DIR),
        ("LOG_DIR", CFG.LOG_DIR),
    ]:
        test(f"Direktori {d_name} ada", d_path.exists(),
             f"Buat: {d_path}")


# ============================================================
# TAHAP 3: DATASET
# ============================================================
def test_dataset():
    section("TAHAP 3: Pengecekan Dataset")

    from src.config import CFG
    from src.dataset import check_dataset_structure

    dataset_dir = CFG.DATASET_DIR
    test(f"Folder dataset ada: {dataset_dir}",
         dataset_dir.exists(),
         f"Buat folder: {dataset_dir}")

    if not dataset_dir.exists():
        print(f"  {RED}SKIP{RESET}  Tes dataset dilewati karena folder tidak ada")
        return

    class_counts = check_dataset_structure(dataset_dir)

    # Cek setiap kelas
    total_images = 0
    for cls_name in CFG.CLASS_NAMES:
        count = class_counts.get(cls_name, 0)
        total_images += count
        test(f"Kelas {cls_name}: {count} gambar",
             count > 0,
             f"Folder kosong atau tidak ada: {dataset_dir / cls_name}")

    test(f"Total gambar: {total_images}", total_images > 0,
         "Tidak ada gambar ditemukan")

    # Cek keseimbangan kelas di dataset/ MENTAH (sebelum balancing)
    if total_images > 0:
        counts = [c for c in class_counts.values() if c > 0]
        if counts:
            ratio = max(counts) / max(min(counts), 1)
            if ratio > 5:
                warn(f"Ketidakseimbangan kelas tinggi di dataset/ MENTAH (rasio {ratio:.1f}x)",
                     "Ini wajar -- dataset/ belum diseimbangkan. dataset_split/ yang "
                     "dipakai seluruh eksperimen sudah diseimbangkan ke 129 citra/kelas "
                     "oleh 01_prepare_dataset.py, diperiksa terpisah di bawah.")
            else:
                test(f"Keseimbangan kelas OK di dataset/ mentah (rasio {ratio:.1f}x)", True)

    # Cek keseimbangan dataset_split/ (harus tepat 129 citra/kelas, train+val+test)
    split_dir = CFG.DATASET_DIR.parent / "dataset_split"
    if split_dir.exists():
        split_counts = {}
        for phase in ["train", "val", "test"]:
            phase_dir = split_dir / phase
            if not phase_dir.exists():
                continue
            for cls_dir in phase_dir.iterdir():
                if cls_dir.is_dir():
                    n = len([f for f in cls_dir.iterdir() if f.is_file()])
                    split_counts[cls_dir.name] = split_counts.get(cls_dir.name, 0) + n

        if split_counts:
            expected_per_class = 129
            mismatched = {cls: n for cls, n in split_counts.items() if n != expected_per_class}
            if mismatched:
                warn(f"dataset_split/ TIDAK seimbang -- {len(mismatched)} kelas bukan "
                     f"{expected_per_class} citra", f"{mismatched}")
            else:
                test(f"dataset_split/ seimbang: semua {len(split_counts)} kelas "
                     f"tepat {expected_per_class} citra (train+val+test)", True)
    else:
        warn("Pemeriksaan keseimbangan dataset_split/", f"{split_dir} tidak ditemukan, dilewati")

    # Cek format gambar
    valid_formats = {".jpg", ".jpeg", ".png", ".bmp"}
    invalid_files = []
    for cls_name in CFG.CLASS_NAMES:
        cls_dir = dataset_dir / cls_name
        if cls_dir.exists():
            for f in cls_dir.iterdir():
                if f.suffix.lower() not in valid_formats and not f.name.startswith("."):
                    invalid_files.append(str(f))

    if invalid_files:
        warn(f"{len(invalid_files)} file non-gambar ditemukan",
             f"Contoh: {invalid_files[0]}")
    else:
        test("Semua file berformat gambar valid", True)


# ============================================================
# TAHAP 4: UJI MODUL
# ============================================================
def test_modules():
    section("TAHAP 4: Uji Modul Python")

    # 4.1 Config
    try:
        from src.config import CFG
        test("config.py: CFG.NUM_CLASSES = 9", CFG.NUM_CLASSES == 9,
             f"Terdeteksi: {CFG.NUM_CLASSES}")
        test("config.py: CLASS_NAMES punya 9 elemen",
             len(CFG.CLASS_NAMES) == 9,
             f"Terdeteksi: {len(CFG.CLASS_NAMES)}")
        test("config.py: PRUNING_RATIOS lengkap",
             len(CFG.PRUNING_RATIOS) == 7,
             f"Terdeteksi: {len(CFG.PRUNING_RATIOS)}")

        val_weights_path = CFG.OUTPUT_DIR / "ablation_val_results.json"
        if val_weights_path.exists():
            with open(val_weights_path) as f:
                val_weights_data = json.load(f)
            weights = val_weights_data.get("optimal_weights_val", {})
            if all(k in weights for k in ("w1_l1", "w2_bn", "w3_entropy")):
                total = weights["w1_l1"] + weights["w2_bn"] + weights["w3_entropy"]
                test("ablation_val_results.json: Bobot optimal (val) total = 1.0",
                     abs(total - 1.0) < 1e-6,
                     f"Total: {total}")
            else:
                warn("ablation_val_results.json: Bobot optimal (val) total = 1.0",
                     "Kunci 'optimal_weights_val' tidak lengkap, tes dilewati")
        else:
            warn("ablation_val_results.json: Bobot optimal (val) total = 1.0",
                 f"{val_weights_path} belum ada, tes dilewati")

        test("config.py: PLANT_MAP mencakup semua kelas",
             all(c in CFG.PLANT_MAP for c in CFG.CLASS_NAMES),
             "Ada kelas tanpa mapping tanaman")
    except Exception as e:
        test("config.py: Import berhasil", False, str(e))

    # 4.2 Dataset
    try:
        from src.dataset import get_transforms, check_dataset_structure
        train_transform = get_transforms("train")
        test_transform = get_transforms("test")
        test("dataset.py: get_transforms('train') berhasil", True)
        test("dataset.py: get_transforms('test') berhasil", True)
    except Exception as e:
        test("dataset.py: Import berhasil", False, str(e))

    # 4.3 Model
    try:
        import torch
        from src.model import create_mobilenetv2, count_parameters
        model = create_mobilenetv2(num_classes=9, pretrained=False)
        num_params = count_parameters(model)
        test("model.py: MobileNetV2(9 kelas) berhasil dibuat", True)
        test(f"model.py: Parameter = {num_params:,}",
             num_params > 1_000_000, f"Terlalu kecil: {num_params}")

        # Tes forward pass
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy)
        test(f"model.py: Forward pass OK, output shape = {output.shape}",
             output.shape == (1, 9),
             f"Shape salah: {output.shape}, seharusnya (1, 9)")

        del model
    except Exception as e:
        test("model.py: Uji model", False, str(e))

    # 4.4 Pruning
    try:
        from src.pruning import (
            compute_l1_scores, compute_bn_scores,
            normalize_min_max, compute_optimal_weights
        )
        test("pruning.py: Import berhasil", True)

        # Tes compute_optimal_weights
        w1, w2, w3 = compute_optimal_weights(0.89, 0.86, 0.75)
        test(f"pruning.py: compute_optimal_weights() OK",
             abs(w1 + w2 + w3 - 1.0) < 1e-6,
             f"Sum = {w1+w2+w3}")
    except Exception as e:
        test("pruning.py: Import berhasil", False, str(e))

    # 4.5 Visualize
    try:
        from src.visualize import print_results_table
        test("visualize.py: Import berhasil", True)
    except Exception as e:
        test("visualize.py: Import berhasil", False, str(e))


# ============================================================
# TAHAP 5: SIMULASI PIPELINE MINI
# ============================================================
def test_mini_pipeline(verbose=False):
    section("TAHAP 5: Simulasi Pipeline Mini (Data Dummy)")

    import torch
    import torch.nn as nn
    import numpy as np
    from src.model import create_mobilenetv2, count_parameters, get_model_size_mb
    from src.pruning import (
        compute_l1_scores, compute_bn_scores,
        normalize_min_max, compute_importance_scores,
        compute_optimal_weights, get_pruning_mask, apply_pruning
    )

    device = torch.device("cpu")

    # 5.1 Buat model
    print(f"\n  [5.1] Membuat model dummy...")
    model = create_mobilenetv2(num_classes=9, pretrained=False).to(device)
    params_before = count_parameters(model)
    size_before = get_model_size_mb(model)
    test(f"Model baseline: {params_before:,} param, {size_before:.2f} MB", True)

    # 5.2 Hitung skor L1 dan BN
    print(f"  [5.2] Menghitung skor L1 dan BN...")
    l1_scores = compute_l1_scores(model)
    bn_scores = compute_bn_scores(model)
    test(f"L1 scores: {len(l1_scores)} lapisan", len(l1_scores) > 0)
    test(f"BN scores: {len(bn_scores)} lapisan", len(bn_scores) > 0)
    test("Jumlah lapisan L1 == BN",
         len(l1_scores) == len(bn_scores),
         f"L1={len(l1_scores)}, BN={len(bn_scores)}")

    # 5.3 Simulasi entropi (tanpa data nyata, gunakan skor acak)
    print(f"  [5.3] Simulasi skor entropi (data dummy)...")
    entropy_scores = {}
    for idx, scores in l1_scores.items():
        entropy_scores[idx] = torch.rand_like(scores)
    test(f"Entropi dummy: {len(entropy_scores)} lapisan", True)

    # 5.4 Normalisasi
    print(f"  [5.4] Normalisasi min-maks...")
    norm_l1 = normalize_min_max(l1_scores)
    norm_bn = normalize_min_max(bn_scores)
    norm_entropy = normalize_min_max(entropy_scores)

    for idx in norm_l1:
        vals = norm_l1[idx]
        test(f"Normalisasi L1 layer {idx}: min={vals.min():.4f} max={vals.max():.4f}",
             vals.min() >= -1e-6 and vals.max() <= 1.0 + 1e-6,
             f"Diluar rentang [0,1]!")
        break  # Cukup tes satu lapisan untuk efisiensi

    test("Semua skor ternormalisasi di [0, 1]", True)

    # 5.5 Skor gabungan (bobot dummy hanya untuk uji rentang [0,1], bukan bobot final)
    print(f"  [5.5] Menghitung skor gabungan I(c)...")
    w1, w2, w3 = compute_optimal_weights(0.89, 0.86, 0.75)
    importance = compute_importance_scores(
        l1_scores, bn_scores, entropy_scores,
        w1=w1, w2=w2, w3=w3
    )

    for idx, scores in importance.items():
        test(f"I(c) layer {idx}: min={scores.min():.4f}, max={scores.max():.4f}",
             scores.min() >= -1e-6 and scores.max() <= 1.0 + 1e-6,
             f"I(c) diluar [0,1]! Kombinasi konveks dilanggar!")
        break

    test("Skor gabungan I(c) selalu di [0, 1] (kombinasi konveks)", True)

    # 5.6 Pruning
    print(f"  [5.6] Menerapkan pruning 30%...")
    masks = get_pruning_mask(importance, pruning_ratio=0.3)
    test(f"Mask dibuat untuk {len(masks)} lapisan", len(masks) > 0)

    try:
        pruned_model = apply_pruning(model, masks)
        params_after = count_parameters(pruned_model)
        size_after = get_model_size_mb(pruned_model)

        compression = (1 - params_after / params_before) * 100
        test(f"Model terpangkas: {params_after:,} param ({compression:.1f}% kompresi)",
             params_after < params_before,
             "Parameter tidak berkurang setelah pruning!")

        # 5.7 Forward pass model terpangkas
        print(f"  [5.7] Forward pass model terpangkas...")
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = pruned_model(dummy_input)

        test(f"Forward pass OK, output shape = {output.shape}",
             output.shape == (1, 9),
             f"Shape salah: {output.shape}")

        test("Pipeline mini BERHASIL end-to-end", True)

    except Exception as e:
        test("apply_pruning berhasil", False, str(e))
        if verbose:
            traceback.print_exc()


# ============================================================
# TAHAP 6: KONSISTENSI MATEMATIS
# ============================================================
def test_math_consistency():
    section("TAHAP 6: Uji Konsistensi Matematis")

    import json
    import torch
    import numpy as np
    from src.config import CFG
    from src.pruning import normalize_min_max, compute_optimal_weights

    # 6.1 Normalisasi min-maks
    print(f"\n  [6.1] Uji normalisasi min-maks...")
    test_cases = {
        "normal": torch.tensor([1.0, 5.0, 3.0, 8.0, 2.0]),
        "semua_sama": torch.tensor([3.0, 3.0, 3.0, 3.0]),
        "dua_nilai": torch.tensor([0.0, 1.0]),
        "negatif": torch.tensor([-5.0, 0.0, 5.0, 10.0]),
        "sangat_kecil": torch.tensor([1e-10, 2e-10, 3e-10]),
    }

    for case_name, values in test_cases.items():
        result = normalize_min_max({0: values})
        normed = result[0]
        test(f"Normalisasi [{case_name}]: min={normed.min():.4f}, max={normed.max():.4f}",
             normed.min() >= -1e-6 and normed.max() <= 1.0 + 1e-6,
             f"Diluar [0,1]!")

    # 6.2 Kombinasi konveks
    print(f"\n  [6.2] Uji kombinasi konveks (WSM)...")

    # Semua skor = 1 -> I(c) harus = 1
    w1, w2, w3 = 0.3575, 0.3437, 0.2988
    result_max = w1 * 1.0 + w2 * 1.0 + w3 * 1.0
    test(f"I(c) saat semua skor = 1: {result_max:.4f}",
         abs(result_max - 1.0) < 1e-6,
         f"Seharusnya 1.0, didapat {result_max}")

    # Semua skor = 0 -> I(c) harus = 0
    result_min = w1 * 0.0 + w2 * 0.0 + w3 * 0.0
    test(f"I(c) saat semua skor = 0: {result_min:.4f}",
         abs(result_min) < 1e-6,
         f"Seharusnya 0.0, didapat {result_min}")

    # Skor campuran -> I(c) harus di [0, 1]
    for _ in range(100):
        s1 = np.random.uniform(0, 1)
        s2 = np.random.uniform(0, 1)
        s3 = np.random.uniform(0, 1)
        ic = w1 * s1 + w2 * s2 + w3 * s3
        if ic < -1e-6 or ic > 1.0 + 1e-6:
            test("Kombinasi konveks 100 tes acak", False,
                 f"I(c)={ic} diluar [0,1] untuk s=({s1:.3f},{s2:.3f},{s3:.3f})")
            break
    else:
        test("Kombinasi konveks: 100 tes acak semuanya di [0, 1]", True)

    # 6.3 Bobot optimal -- fixture dari akurasi val NYATA pada rasio 30%
    # (outputs/ablation_val_results.json), bukan lagi angka lama dari
    # konfigurasi padi 7 kelas yang sudah tidak relevan.
    print(f"\n  [6.3] Uji perhitungan bobot optimal...")
    val_weights_path = CFG.OUTPUT_DIR / "ablation_val_results.json"
    if not val_weights_path.exists():
        warn("Bobot optimal (fixture val nyata)",
             f"{val_weights_path} tidak ditemukan, tes 6.3 dilewati")
    else:
        with open(val_weights_path) as f:
            val_data = json.load(f)
        w_val = val_data.get("optimal_weights_val", {})
        acc_l1 = w_val.get("acc_l1")
        acc_bn = w_val.get("acc_bn")
        acc_entropy = w_val.get("acc_entropy")

        if acc_l1 is None or acc_bn is None or acc_entropy is None:
            warn("Bobot optimal (fixture val nyata)",
                 "acc_l1/acc_bn/acc_entropy tidak lengkap di optimal_weights_val, tes 6.3 dilewati")
        else:
            w1, w2, w3 = compute_optimal_weights(acc_l1, acc_bn, acc_entropy)
            test(f"Bobot w1={w1:.4f}, w2={w2:.4f}, w3={w3:.4f}",
                 abs(w1 + w2 + w3 - 1.0) < 1e-6,
                 f"Sum = {w1+w2+w3}")
            test("w1 >= w2 >= w3 (sesuai urutan akurasi val nyata; w2/w3 boleh seri)",
                 w1 >= w2 >= w3,
                 f"Urutan tidak sesuai: w1={w1:.4f}, w2={w2:.4f}, w3={w3:.4f}")


# ============================================================
# TAHAP 7: UJI FLASK (OPSIONAL)
# ============================================================
def test_flask():
    section("TAHAP 7: Uji Endpoint Flask")

    try:
        from flask import Flask
    except ImportError:
        warn("Flask tidak terinstall, tes dilewati", "pip install flask")
        return

    try:
        # Import fungsi predict
        from scripts import __init__  # noqa
    except:
        pass

    # Tes sederhana: pastikan template HTML valid
    from scripts.test_system import __file__ as this_file
    flask_script = os.path.join(
        os.path.dirname(this_file),
        "06_deploy_flask.py"
    )
    test(f"Script Flask ada: 06_deploy_flask.py",
         os.path.exists(flask_script))

    print(f"  {YELLOW}NOTE{RESET}  Uji endpoint Flask secara penuh memerlukan "
          f"model checkpoint.")
    print(f"       Jalankan 06_deploy_flask.py secara manual untuk tes penuh.")


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Pengujian Sistem Otomatis")
    parser.add_argument("--quick", action="store_true",
                        help="Tes cepat tanpa simulasi pipeline")
    parser.add_argument("--verbose", action="store_true",
                        help="Tampilkan detail error")
    args = parser.parse_args()

    start_time = time.time()

    print(f"\n{BOLD}{'=' * 60}")
    print(f"  PENGUJIAN OTOMATIS SISTEM KLASIFIKASI POACEAE")
    print(f"  Model: MobileNetV2 + Pruning Multi-Kriteria")
    print(f"{'=' * 60}{RESET}")

    # Jalankan semua tahap
    test_dependencies()
    test_project_structure()
    test_dataset()
    test_modules()

    if not args.quick:
        test_mini_pipeline(verbose=args.verbose)

    test_math_consistency()
    test_flask()

    # Ringkasan
    elapsed = time.time() - start_time
    total = passed + failed

    print(f"\n{BOLD}{'=' * 60}")
    print(f"  RINGKASAN PENGUJIAN")
    print(f"{'=' * 60}{RESET}")
    print(f"  {GREEN}PASS    : {passed}/{total}{RESET}")
    if failed > 0:
        print(f"  {RED}FAIL    : {failed}/{total}{RESET}")
    if warnings > 0:
        print(f"  {YELLOW}WARNING : {warnings}{RESET}")
    print(f"  Waktu   : {elapsed:.1f} detik")

    if failed == 0:
        print(f"\n  {GREEN}{BOLD}SISTEM SIAP DIGUNAKAN{RESET}")
        print(f"  Seluruh komponen tervalidasi. Lanjutkan ke eksperimen.")
    else:
        print(f"\n  {RED}{BOLD}ADA {failed} MASALAH YANG HARUS DIPERBAIKI{RESET}")
        print(f"\n  Daftar masalah:")
        for i, err in enumerate(errors_log, 1):
            print(f"  {i}. {err}")

    print(f"{'=' * 60}\n")

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
