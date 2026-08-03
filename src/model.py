"""
model.py - Model MobileNetV2 dan Fungsi Training/Evaluasi

Modul ini menangani:
1. Inisialisasi MobileNetV2 pretrained ImageNet
2. Modifikasi classifier head untuk 9 kelas
3. Training loop dengan validasi
4. Evaluasi dan pengukuran metrik
5. Pengukuran waktu inferensi
"""

import time
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

from src.config import CFG
from src.pruning import apply_pruning, _get_prunable_layers


def create_mobilenetv2(num_classes=None, pretrained=True):
    """
    Membuat model MobileNetV2 dengan classifier head untuk num_classes.

    Args:
        num_classes: Jumlah kelas output (default: CFG.NUM_CLASSES)
        pretrained: Gunakan bobot pretrained ImageNet

    Returns:
        model: MobileNetV2 yang siap dilatih
    """
    if num_classes is None:
        num_classes = CFG.NUM_CLASSES

    # Muat MobileNetV2 pretrained
    if pretrained:
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        model = models.mobilenet_v2(weights=weights)
        print("[INFO] MobileNetV2 pretrained ImageNet dimuat.")
    else:
        model = models.mobilenet_v2(weights=None)
        print("[INFO] MobileNetV2 tanpa pretrained dimuat.")

    # Ganti classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes),
    )
    print(f"[INFO] Classifier head diganti: {in_features} -> {num_classes} kelas")

    return model


def count_parameters(model):
    """Menghitung jumlah parameter yang dapat dilatih."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model):
    """Menghitung ukuran model dalam MB."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def measure_flops(model, device=None):
    """
    Mengukur FLOPs model menggunakan thop.

    Catatan: thop.profile() memasang buffer total_ops/total_params secara
    permanen (in-place) pada setiap submodule yang diprofil dan tidak
    membersihkannya sendiri. Kalau model asli yang diprofil, buffer itu ikut
    tersalin lewat copy.deepcopy() (mis. saat pruning) dan akhirnya tersimpan
    ke checkpoint lewat state_dict(). Untuk mencegahnya, yang diprofil adalah
    SALINAN model, bukan model aslinya -- pola yang sama dengan
    measure_inference_time().

    Returns:
        flops: Jumlah FLOPs (float)
        params: Jumlah parameter (int)
    """
    if device is None:
        device = next(model.parameters()).device

    try:
        from thop import profile
        profile_model = copy.deepcopy(model)
        dummy = torch.randn(1, 3, CFG.IMG_SIZE, CFG.IMG_SIZE).to(device)
        flops, params = profile(profile_model, inputs=(dummy,), verbose=False)
        del profile_model
        return flops, params
    except ImportError:
        print("[PERINGATAN] thop tidak terinstall. Jalankan: pip install thop")
        return 0, count_parameters(model)


def measure_inference_time(model, device=None, runs=None, warmup=None):
    """
    Mengukur median waktu inferensi pada CPU, dengan 1 thread supaya hasil
    konsisten dan tidak dipengaruhi penjadwalan OS/multi-thread. Jumlah
    thread PyTorch dikembalikan ke nilai semula di blok finally, supaya
    tidak memperlambat training/evaluasi lain yang berjalan setelah fungsi
    ini dipanggil di tengah pipeline -- termasuk kalau terjadi exception.

    Args:
        model: Model PyTorch
        device: Device untuk inferensi (default: CPU)
        runs: Jumlah ulangan (default: CFG.INFERENCE_RUNS)
        warmup: Jumlah pemanasan (default: CFG.WARMUP_RUNS)

    Returns:
        tuple: (median_time_ms, std_time_ms, runs)
    """
    if runs is None:
        runs = CFG.INFERENCE_RUNS
    if warmup is None:
        warmup = CFG.WARMUP_RUNS

    original_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        # Selalu ukur di CPU untuk konsistensi
        cpu_model = copy.deepcopy(model).cpu().eval()
        dummy = torch.randn(1, 3, CFG.IMG_SIZE, CFG.IMG_SIZE)

        # Pemanasan
        with torch.no_grad():
            for _ in range(warmup):
                _ = cpu_model(dummy)

        # Pengukuran
        times = []
        with torch.no_grad():
            for _ in range(runs):
                start = time.perf_counter()
                _ = cpu_model(dummy)
                end = time.perf_counter()
                times.append((end - start) * 1000)  # konversi ke ms
    finally:
        torch.set_num_threads(original_threads)

    median_time = float(np.median(times))
    avg_time = float(np.mean(times))
    std_time = float(np.std(times))
    print(f"[INFO] Inferensi: median={median_time:.2f} ms "
          f"(rata-rata={avg_time:.2f} +/- {std_time:.2f} ms, "
          f"{runs} runs, {warmup} warmup, 1 thread)")
    return median_time, std_time, runs


def train_model(model, dataloaders, dataset_sizes, num_epochs, lr,
                device=None, checkpoint_path=None, phase_name="baseline"):
    """
    Training loop dengan validasi per epoch.

    Args:
        model: Model PyTorch
        dataloaders: dict dengan "train" dan "val"
        dataset_sizes: dict dengan ukuran dataset per fase
        num_epochs: Jumlah epoch
        lr: Learning rate
        device: Device (default: auto-detect)
        checkpoint_path: Path untuk menyimpan model terbaik
        phase_name: Nama fase untuk logging

    Returns:
        model: Model dengan bobot terbaik
        history: dict dengan riwayat training
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training {phase_name} pada {device}, {num_epochs} epoch, lr={lr}")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 40)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == "train":
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc.item())

            print(f"  {phase:5s} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Simpan model terbaik berdasarkan validasi
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print(f"  >> Model terbaik diperbarui! Val Acc: {best_acc:.4f}")

    print(f"\n[SELESAI] {phase_name} - Best Val Acc: {best_acc:.4f}")

    # Muat bobot terbaik
    model.load_state_dict(best_model_wts)

    # Simpan checkpoint
    if checkpoint_path:
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": model.state_dict(),
            "best_acc": best_acc,
            "history": history,
            "num_classes": CFG.NUM_CLASSES,
            "class_names": CFG.CLASS_NAMES,
        }, checkpoint_path)
        print(f"[INFO] Checkpoint disimpan: {checkpoint_path}")

    return model, history


def evaluate_model(model, dataloader, device=None):
    """
    Evaluasi lengkap model pada data test.

    Args:
        model: Model PyTorch
        dataloader: DataLoader untuk test set
        device: Device

    Returns:
        metrics: dict berisi semua metrik evaluasi
        all_preds: numpy array prediksi
        all_labels: numpy array label asli
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Hitung metrik
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
    rec = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    metrics = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
    }

    # Hitung metrik efisiensi
    metrics["num_params"] = count_parameters(model)
    metrics["model_size_mb"] = get_model_size_mb(model)

    flops, _ = measure_flops(model, device)
    metrics["flops"] = flops

    inference_ms, inference_std_ms, inference_runs = measure_inference_time(model)
    metrics["inference_ms"] = inference_ms
    metrics["inference_std_ms"] = inference_std_ms
    metrics["inference_runs"] = inference_runs

    # Cetak ringkasan
    print("\n" + "=" * 60)
    print("HASIL EVALUASI")
    print("=" * 60)
    print(f"  Akurasi      : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision    : {prec:.4f}")
    print(f"  Recall       : {rec:.4f}")
    print(f"  F1-Score     : {f1:.4f}")
    print(f"  Parameter    : {metrics['num_params']:,}")
    print(f"  Ukuran (MB)  : {metrics['model_size_mb']:.2f}")
    print(f"  FLOPs        : {flops/1e6:.3f} M")
    print(f"  Inferensi    : {inference_ms:.2f} ms (median, std={inference_std_ms:.2f}, n={inference_runs})")
    print("=" * 60)

    # Classification report per kelas
    print("\nClassification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=CFG.CLASS_NAMES,
        digits=4
    ))

    return metrics, all_preds, all_labels


def _infer_masks_from_state_dict(model, state_dict):
    """
    Simpulkan mask pemangkasan per block dari bentuk tensor conv depthwise
    di checkpoint["model_state_dict"], dibandingkan dengan lebar penuh pada
    `model` (MobileNetV2 belum dipangkas). Dipakai load_checkpoint() untuk
    merekonstruksi arsitektur pasca-pruning sebelum memuat bobot -- isi mask
    itu sendiri arbitrer (hanya menentukan JUMLAH channel yang dipertahankan),
    karena bobot aslinya akan langsung ditimpa oleh state_dict checkpoint.

    Returns:
        dict: {layer_idx: tensor boolean mask} -- hanya untuk block yang
        jumlah channel-nya di checkpoint berbeda dari model penuh.
    """
    prunable = _get_prunable_layers(model)
    module_to_name = {m: n for n, m in model.named_modules()}

    masks = {}
    for layer_idx, (block_idx, dw_module) in enumerate(prunable):
        full_width = dw_module.out_channels
        dw_name = module_to_name.get(dw_module)
        dw_key = f"{dw_name}.weight" if dw_name else None

        if dw_key is None or dw_key not in state_dict:
            continue  # tidak bisa disimpulkan, biarkan lebar penuh

        n_keep = state_dict[dw_key].shape[0]
        if n_keep == full_width:
            continue  # block ini tidak dipangkas

        mask = torch.zeros(full_width, dtype=torch.bool)
        mask[:n_keep] = True
        masks[layer_idx] = mask

    return masks


def load_checkpoint(checkpoint_path, device=None):
    """
    Memuat model dari checkpoint. Bentuk tensor di
    checkpoint["model_state_dict"] dipakai untuk menyimpulkan arsitektur --
    termasuk model hasil pruning yang channel intermediate-nya lebih kecil
    dari model penuh -- sebelum bobot dimuat. Model yang tidak dipangkas
    (mis. baseline) tetap dimuat seperti biasa karena tidak ada block yang
    perlu direkonstruksi.

    Args:
        checkpoint_path: Path ke file checkpoint (.pth)
        device: Device

    Returns:
        model: Model dengan arsitektur dan bobot sesuai checkpoint
        checkpoint: dict berisi metadata
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes = checkpoint.get("num_classes", CFG.NUM_CLASSES)
    state_dict = checkpoint["model_state_dict"]

    model = create_mobilenetv2(num_classes=num_classes, pretrained=False)

    masks = _infer_masks_from_state_dict(model, state_dict)
    if masks:
        model = apply_pruning(model, masks)

    model.load_state_dict(state_dict)
    model = model.to(device)

    # Verifikasi arsitektur: jumlah parameter model yang dimuat harus cocok
    # dengan checkpoint["num_params"] jika kunci itu tersedia. Mencegah model
    # termuat dengan arsitektur yang salah tanpa disadari.
    expected_num_params = checkpoint.get("num_params")
    if expected_num_params is not None:
        actual_num_params = count_parameters(model)
        if actual_num_params != expected_num_params:
            raise RuntimeError(
                f"Verifikasi arsitektur gagal saat memuat {checkpoint_path}: "
                f"jumlah parameter model ({actual_num_params:,}) tidak cocok "
                f"dengan checkpoint['num_params'] ({expected_num_params:,})."
            )

    print(f"[INFO] Checkpoint dimuat: {checkpoint_path}")
    print(f"  Best Acc: {checkpoint.get('best_acc', 'N/A')}")

    return model, checkpoint
