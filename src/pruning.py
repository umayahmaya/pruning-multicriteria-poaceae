"""
pruning.py - Sistem Penskoran Multi-Kriteria dan Structured Pruning

Modul ini mengimplementasikan kontribusi metodologis utama tesis:
1. Perhitungan tiga skor kepentingan channel:
   - S_L1: Norma L1 bobot filter (sisi hulu / penyebab)
   - S_BN: Faktor skala BN gamma (sisi tengah / gerbang)
   - S_H:  Entropi Shannon peta fitur (sisi hilir / akibat)
2. Normalisasi min-maks per lapisan
3. Fungsi penskoran gabungan I(c) = w1*S_L1 + w2*S_BN + w3*S_H
   (adaptasi Weighted Sum Model, Fishburn 1967)
4. Structured pruning pada channel intermediate inverted residual block
5. Rekonstruksi arsitektur MobileNetV2 pasca-pruning

Referensi utama:
    - Fishburn, P.C. (1967). Additive Utilities. Operations Research, 15(3).
    - He & Xiao (2024). Structured Pruning Survey. IEEE TPAMI.
    - Sandler et al. (2018). MobileNetV2. IEEE CVPR.
"""

import copy
import random
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from torchvision import datasets
from torch.utils.data import DataLoader, Subset

from src.config import CFG
from src.dataset import get_transforms


# ============================================================
# BAGIAN 1: PERHITUNGAN TIGA SKOR KEPENTINGAN CHANNEL
# ============================================================

def _get_prunable_layers(model):
    """
    Mengidentifikasi lapisan depthwise yang berada di blok dengan expansion layer.
    Block tanpa expansion (t=1, seperti features[1]) di-skip karena
    DW channel = input channel dan pruning akan merusak dependency.

    Returns:
        list of (block_idx, depthwise_module) tuples
    """
    prunable = []
    for block_idx, block in enumerate(model.features):
        if not hasattr(block, "conv"):
            continue

        dw_module = None
        has_expansion = False

        for layer in block.conv:
            if hasattr(layer, "__getitem__"):
                for sub in layer:
                    if isinstance(sub, nn.Conv2d) and sub.groups == sub.in_channels and sub.groups > 1:
                        dw_module = sub
                    if isinstance(sub, nn.Conv2d) and sub.kernel_size == (1, 1) and sub.groups == 1:
                        if sub.out_channels > sub.in_channels:
                            has_expansion = True

        if dw_module is not None and has_expansion:
            prunable.append((block_idx, dw_module))

    return prunable


def compute_l1_scores(model):
    """
    Menghitung skor Norma L1 untuk channel intermediate yang bisa dipangkas.
    Hanya blok dengan expansion layer yang dihitung (skip blok t=1).

    Posisi dalam rantai komputasi: SISI HULU (penyebab)

    Args:
        model: MobileNetV2

    Returns:
        dict: {layer_idx: tensor skor L1 per channel}
    """
    prunable = _get_prunable_layers(model)
    l1_scores = {}

    for layer_idx, (block_idx, dw_module) in enumerate(prunable):
        weights = dw_module.weight.data
        scores = weights.abs().sum(dim=[1, 2, 3])
        l1_scores[layer_idx] = scores.cpu()

    print(f"[L1-Norm] Dihitung untuk {len(l1_scores)} lapisan depthwise (skip t=1)")
    return l1_scores


def compute_bn_scores(model):
    """
    Menghitung skor faktor skala BN Gamma untuk channel intermediate yang bisa dipangkas.
    Hanya blok dengan expansion layer yang dihitung (skip blok t=1).

    Posisi dalam rantai komputasi: SISI TENGAH (gerbang)

    Args:
        model: MobileNetV2

    Returns:
        dict: {layer_idx: tensor skor BN gamma per channel}
    """
    prunable = _get_prunable_layers(model)
    bn_scores = {}

    for layer_idx, (block_idx, dw_module) in enumerate(prunable):
        block = model.features[block_idx]
        # Cari BN setelah depthwise di Conv2dNormActivation yang berisi DW
        for layer in block.conv:
            if hasattr(layer, "__getitem__"):
                found_dw = False
                for sub in layer:
                    if sub is dw_module:
                        found_dw = True
                    elif found_dw and isinstance(sub, nn.BatchNorm2d):
                        gamma = sub.weight.data.abs()
                        bn_scores[layer_idx] = gamma.cpu()
                        break

    print(f"[BN Gamma] Dihitung untuk {len(bn_scores)} lapisan BN (skip t=1)")
    return bn_scores


def compute_entropy_scores(model, dataloader, device=None,
                            samples_per_class=20, seed=CFG.SEED):
    """
    Menghitung skor Entropi Shannon dari peta fitur keluaran channel.
    Entropi dihitung pada histogram peta fitur menggunakan data kalibrasi
    yang diambil seimbang per kelas secara deterministik (seed tetap),
    memakai transform evaluasi (resize + normalisasi saja, tanpa augmentasi)
    agar nilai piksel kalibrasi reproducible antar run.

    Posisi dalam rantai komputasi: SISI HILIR (akibat)
    Mengukur: kekayaan informasi yang diproduksi channel saat memproses data nyata

    Args:
        model: MobileNetV2
        dataloader: DataLoader untuk data kalibrasi (harus dari data latih;
            transform augmentasinya diabaikan, diganti transform evaluasi)
        device: Device
        samples_per_class: Jumlah gambar kalibrasi per kelas (default: 20)
        seed: Seed pengambilan sampel agar deterministik (default: CFG.SEED)

    Returns:
        dict: {layer_idx: tensor skor entropi per channel}
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Ambil sampel kalibrasi seimbang per kelas secara deterministik
    dataset = dataloader.dataset
    indices_by_class = {}
    for idx, label in enumerate(dataset.targets):
        indices_by_class.setdefault(label, []).append(idx)

    rng = random.Random(seed)
    calib_indices = []
    for label in sorted(indices_by_class):
        class_indices = sorted(indices_by_class[label])
        rng.shuffle(class_indices)
        calib_indices.extend(class_indices[:samples_per_class])

    # Muat ulang dataset yang sama dengan transform evaluasi (bukan transform
    # augmentasi milik dataloader train) agar piksel kalibrasi reproducible
    eval_dataset = datasets.ImageFolder(dataset.root, transform=get_transforms("test"))
    if len(eval_dataset) != len(dataset) or eval_dataset.targets != dataset.targets:
        raise RuntimeError(
            "Urutan dataset berubah antara pemindaian train dan eval, "
            "indeks kalibrasi tidak bisa dipetakan dengan aman."
        )
    calib_loader = DataLoader(
        Subset(eval_dataset, calib_indices),
        batch_size=dataloader.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Pasang hook untuk menangkap feature map setelah depthwise+BN+ReLU6
    feature_maps = {}
    hooks = []

    def get_hook(idx):
        def hook_fn(module, input, output):
            if idx not in feature_maps:
                feature_maps[idx] = []
            feature_maps[idx].append(output.detach().cpu())
        return hook_fn

    # Pasang hook hanya pada blok yang punya expansion (skip t=1)
    prunable = _get_prunable_layers(model)
    for layer_idx, (block_idx, dw_module) in enumerate(prunable):
        block = model.features[block_idx]
        # Cari Conv2dNormActivation yang berisi depthwise
        for layer in block.conv:
            if hasattr(layer, "__getitem__"):
                for sub in layer:
                    if sub is dw_module:
                        hook = layer.register_forward_hook(get_hook(layer_idx))
                        hooks.append(hook)
                        break

    # Forward pass pada seluruh sampel kalibrasi
    with torch.no_grad():
        for inputs, _ in calib_loader:
            inputs = inputs.to(device)
            _ = model(inputs)

    # Hapus hooks
    for hook in hooks:
        hook.remove()

    # Hitung entropi Shannon per channel
    entropy_scores = {}
    for idx, fmaps_list in feature_maps.items():
        # Gabungkan semua batch: [total_samples, channels, H, W]
        all_fmaps = torch.cat(fmaps_list, dim=0)
        num_channels = all_fmaps.shape[1]
        channel_entropies = torch.zeros(num_channels)

        for c in range(num_channels):
            # Ambil feature map channel c dari semua sampel
            fmap = all_fmaps[:, c, :, :].flatten().numpy()

            # Buat histogram (256 bin)
            hist, _ = np.histogram(fmap, bins=256, density=True)

            # Hitung entropi Shannon: H = -sum(p * log2(p + epsilon))
            hist = hist + 1e-10  # epsilon untuk menghindari log(0)
            hist = hist / hist.sum()  # normalisasi ke probabilitas
            entropy = -np.sum(hist * np.log2(hist))
            channel_entropies[c] = entropy

        entropy_scores[idx] = channel_entropies

    print(f"[Entropi] Dihitung untuk {len(entropy_scores)} lapisan "
          f"menggunakan {len(calib_indices)} gambar kalibrasi "
          f"({samples_per_class} per kelas x {len(indices_by_class)} kelas, seed={seed})")

    return entropy_scores


# ============================================================
# BAGIAN 2: NORMALISASI DAN FUNGSI PENSKORAN GABUNGAN
# ============================================================

def normalize_min_max(scores_dict):
    """
    Normalisasi min-maks per lapisan.

    Rumus: S_norm = (S - S_min) / (S_max - S_min + epsilon)

    Jaminan matematis: output selalu berada di selang [0, 1]
    karena pembilang >= 0 dan <= penyebut.

    Args:
        scores_dict: {layer_idx: tensor skor mentah}

    Returns:
        dict: {layer_idx: tensor skor ternormalisasi [0, 1]}
    """
    normalized = {}
    for idx, scores in scores_dict.items():
        s_min = scores.min()
        s_max = scores.max()
        denominator = s_max - s_min

        if denominator < 1e-8:
            # Semua channel punya skor yang sama
            normalized[idx] = torch.full_like(scores, 0.5)
        else:
            normalized[idx] = (scores - s_min) / denominator

    return normalized


def compute_importance_scores(l1_scores, bn_scores, entropy_scores,
                               w1, w2, w3):
    """
    Menghitung skor kepentingan gabungan I(c) menggunakan WSM (Fishburn, 1967).

    I(c) = w1 * S_norm_L1(c) + w2 * S_norm_BN(c) + w3 * S_norm_H(c)

    Syarat: w1 + w2 + w3 = 1, w1 >= 0, w2 >= 0, w3 >= 0
    Properti: I(c) selalu berada di [0, 1] (kombinasi konveks)

    Args:
        l1_scores: dict skor L1 mentah
        bn_scores: dict skor BN mentah
        entropy_scores: dict skor entropi mentah
        w1, w2, w3: bobot (wajib diisi eksplisit, tidak ada default)

    Returns:
        dict: {layer_idx: tensor skor gabungan I(c) per channel}
    """
    # Verifikasi syarat WSM
    total_weight = w1 + w2 + w3
    assert abs(total_weight - 1.0) < 1e-6, \
        f"Jumlah bobot harus = 1.0, didapat {total_weight}"
    assert w1 >= 0 and w2 >= 0 and w3 >= 0, \
        "Semua bobot harus >= 0"

    print(f"[WSM] Bobot: w1(L1)={w1:.4f}, w2(BN)={w2:.4f}, w3(H)={w3:.4f}")
    print(f"[WSM] Verifikasi: w1+w2+w3 = {total_weight:.4f}")

    # Normalisasi min-maks per lapisan
    norm_l1 = normalize_min_max(l1_scores)
    norm_bn = normalize_min_max(bn_scores)
    norm_entropy = normalize_min_max(entropy_scores)

    # Hitung skor gabungan
    importance_scores = {}
    common_layers = set(norm_l1.keys()) & set(norm_bn.keys()) & set(norm_entropy.keys())

    for idx in sorted(common_layers):
        importance = (
            w1 * norm_l1[idx] +
            w2 * norm_bn[idx] +
            w3 * norm_entropy[idx]
        )
        importance_scores[idx] = importance

        # Verifikasi properti kombinasi konveks
        assert importance.min() >= -1e-6, f"I(c) < 0 pada layer {idx}!"
        assert importance.max() <= 1.0 + 1e-6, f"I(c) > 1 pada layer {idx}!"

    print(f"[WSM] Skor gabungan dihitung untuk {len(importance_scores)} lapisan")
    return importance_scores


# ============================================================
# BAGIAN 3: STRUCTURED PRUNING
# ============================================================

def get_pruning_mask(importance_scores, pruning_ratio):
    """
    Membuat mask pemangkasan berdasarkan skor kepentingan.
    Channel dengan skor terendah (paling tidak penting) akan dipangkas.

    Args:
        importance_scores: dict {layer_idx: tensor skor I(c)}
        pruning_ratio: float, persentase channel yang dipangkas (0.0 - 1.0)

    Returns:
        dict: {layer_idx: tensor boolean mask (True = pertahankan)}
    """
    masks = {}
    total_pruned = 0
    total_channels = 0

    for idx, scores in importance_scores.items():
        num_channels = len(scores)
        num_prune = int(num_channels * pruning_ratio)

        # Pastikan minimal 1 channel tersisa
        num_prune = min(num_prune, num_channels - 1)

        # Urutkan berdasarkan skor (ascending)
        _, sorted_indices = torch.sort(scores)

        # Channel yang dipangkas = yang skornya terendah
        prune_indices = sorted_indices[:num_prune]

        # Buat mask: True = pertahankan, False = pangkas
        mask = torch.ones(num_channels, dtype=torch.bool)
        mask[prune_indices] = False

        masks[idx] = mask
        total_pruned += num_prune
        total_channels += num_channels

    print(f"[Pruning] Rasio: {pruning_ratio*100:.0f}%")
    print(f"  Channel total  : {total_channels}")
    print(f"  Channel dipangkas: {total_pruned}")
    print(f"  Channel tersisa  : {total_channels - total_pruned}")

    return masks


def apply_pruning(model, masks):
    """
    Menerapkan pruning pada model MobileNetV2 dengan merekonstruksi
    arsitektur berdasarkan mask.

    Struktur MobileNetV2 InvertedResidual:
    - Block t=1 (features[1]): conv[0]=DW_BN_ReLU, conv[1]=Proj1x1, conv[2]=BN
    - Block t>1 (features[2-17]): conv[0]=Exp1x1_BN_ReLU, conv[1]=DW_BN_ReLU,
                                   conv[2]=Proj1x1, conv[3]=BN

    Args:
        model: MobileNetV2 asli
        masks: dict {layer_idx: boolean mask}

    Returns:
        pruned_model: MobileNetV2 baru dengan channel yang sudah dipangkas
    """
    pruned_model = copy.deepcopy(model)
    layer_idx = 0

    for block_idx, block in enumerate(pruned_model.features):
        if not hasattr(block, "conv"):
            continue

        # Cek apakah blok ini punya depthwise conv
        has_depthwise = False
        has_expansion = False
        for layer in block.conv:
            if isinstance(layer, nn.Sequential) or hasattr(layer, "__getitem__"):
                for sub in layer:
                    if isinstance(sub, nn.Conv2d) and sub.groups == sub.in_channels:
                        has_depthwise = True
                    if isinstance(sub, nn.Conv2d) and sub.kernel_size == (1, 1) and sub.groups == 1:
                        if sub.out_channels > sub.in_channels:
                            has_expansion = True

        if not has_depthwise:
            continue

        # SKIP blok tanpa expansion layer (t=1, seperti features[1])
        # karena DW channel = input channel dan pruning akan merusak
        # dependency ke lapisan sebelumnya
        if not has_expansion:
            continue

        if layer_idx not in masks:
            layer_idx += 1
            continue

        mask = masks[layer_idx]
        keep_idx = torch.where(mask)[0]
        n_keep = len(keep_idx)

        if n_keep == len(mask):
            layer_idx += 1
            continue

        device = next(block.parameters()).device

        # Proses setiap sublayer di block.conv
        for i, layer in enumerate(block.conv):
            # === Conv2dNormActivation (container dengan Conv2d + BN + ReLU6) ===
            if hasattr(layer, "__getitem__") and hasattr(layer, "__len__"):
                for j, sub in enumerate(layer):
                    if isinstance(sub, nn.Conv2d):
                        if sub.groups == sub.in_channels and sub.groups > 1:
                            # DEPTHWISE CONV
                            new_dw = nn.Conv2d(
                                n_keep, n_keep, sub.kernel_size,
                                stride=sub.stride, padding=sub.padding,
                                groups=n_keep, bias=(sub.bias is not None)
                            ).to(device)
                            new_dw.weight.data = sub.weight.data[keep_idx]
                            if sub.bias is not None:
                                new_dw.bias.data = sub.bias.data[keep_idx]
                            layer[j] = new_dw

                        elif sub.kernel_size == (1, 1) and sub.groups == 1:
                            # EXPANSION 1x1 (output > input)
                            new_exp = nn.Conv2d(
                                sub.in_channels, n_keep,
                                kernel_size=1, bias=(sub.bias is not None)
                            ).to(device)
                            new_exp.weight.data = sub.weight.data[keep_idx]
                            if sub.bias is not None:
                                new_exp.bias.data = sub.bias.data[keep_idx]
                            layer[j] = new_exp

                    elif isinstance(sub, nn.BatchNorm2d):
                        if sub.num_features != n_keep:
                            new_bn = nn.BatchNorm2d(n_keep).to(device)
                            new_bn.weight.data = sub.weight.data[keep_idx]
                            new_bn.bias.data = sub.bias.data[keep_idx]
                            new_bn.running_mean = sub.running_mean[keep_idx]
                            new_bn.running_var = sub.running_var[keep_idx]
                            layer[j] = new_bn

            # === Standalone Conv2d (projection 1x1) ===
            elif isinstance(layer, nn.Conv2d):
                if layer.kernel_size == (1, 1) and layer.groups == 1:
                    if layer.in_channels != n_keep:
                        # PROJECTION 1x1 (input dipangkas)
                        new_proj = nn.Conv2d(
                            n_keep, layer.out_channels,
                            kernel_size=1, bias=(layer.bias is not None)
                        ).to(device)
                        new_proj.weight.data = layer.weight.data[:, keep_idx]
                        if layer.bias is not None:
                            new_proj.bias.data = layer.bias.data
                        block.conv[i] = new_proj

            # === Standalone BatchNorm2d (setelah projection) ===
            elif isinstance(layer, nn.BatchNorm2d):
                # BN setelah projection tidak dipangkas karena
                # dimensinya mengikuti output projection (bottleneck)
                pass

        layer_idx += 1

    return pruned_model


# ============================================================
# BAGIAN 4: ABLATION STUDY
# ============================================================

def run_ablation_single_criterion(model, dataloader, pruning_ratio,
                                   criterion_name, device=None):
    """
    Menjalankan pruning dengan satu kriteria tunggal untuk ablation study.
    Ini adalah kasus batas dari I(c) di mana satu bobot = 1 dan dua lainnya = 0.

    Args:
        model: MobileNetV2 baseline
        dataloader: DataLoader kalibrasi
        pruning_ratio: float
        criterion_name: "l1", "bn", atau "entropy"
        device: Device

    Returns:
        pruned_model: Model terpangkas
        masks: Mask pemangkasan
    """
    if criterion_name == "l1":
        scores = compute_l1_scores(model)
        scores = normalize_min_max(scores)
    elif criterion_name == "bn":
        scores = compute_bn_scores(model)
        scores = normalize_min_max(scores)
    elif criterion_name == "entropy":
        scores = compute_entropy_scores(model, dataloader, device)
        scores = normalize_min_max(scores)
    else:
        raise ValueError(f"Kriteria tidak dikenal: {criterion_name}")

    masks = get_pruning_mask(scores, pruning_ratio)
    pruned_model = apply_pruning(model, masks)

    return pruned_model, masks


def compute_optimal_weights(acc_l1, acc_bn, acc_entropy):
    """
    Menghitung bobot optimal berdasarkan rasio akurasi kriteria tunggal.

    Rumus: w_k = Akurasi_k / (Akurasi_L1 + Akurasi_BN + Akurasi_Entropy)

    Metode: Objective weighting (Multi-Criteria Decision Analysis)
    Sifat: Objektif, reproducible, tanpa intervensi subjektif

    Args:
        acc_l1: Akurasi skenario L1 saja
        acc_bn: Akurasi skenario BN gamma saja
        acc_entropy: Akurasi skenario entropi saja

    Returns:
        tuple: (w1, w2, w3)
    """
    total = acc_l1 + acc_bn + acc_entropy
    w1 = acc_l1 / total
    w2 = acc_bn / total
    w3 = acc_entropy / total

    print(f"\n{'='*60}")
    print(f"PERHITUNGAN BOBOT OPTIMAL")
    print(f"{'='*60}")
    print(f"  Akurasi L1      : {acc_l1:.4f}")
    print(f"  Akurasi BN      : {acc_bn:.4f}")
    print(f"  Akurasi Entropi : {acc_entropy:.4f}")
    print(f"  Total           : {total:.4f}")
    print(f"  ─────────────────────────")
    print(f"  w1 (L1)         : {w1:.4f}")
    print(f"  w2 (BN)         : {w2:.4f}")
    print(f"  w3 (Entropi)    : {w3:.4f}")
    print(f"  w1 + w2 + w3    : {w1+w2+w3:.4f}")
    print(f"{'='*60}")

    return w1, w2, w3
