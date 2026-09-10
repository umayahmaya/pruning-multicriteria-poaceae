"""
src/gm_scores.py
Skor GM (jarak antar-channel) untuk lapisan DEPTHWISE CONVOLUTION yang
bisa dipangkas -- MODUL BARU, TERPISAH dari skor "GM" yang dipakai di
36_skor_redundansi.py/37/38/40/41 (yang dihitung dari bobot conv 1x1
EKSPANSI, bukan dari depthwise). Definisi di modul ini SENGAJA berbeda,
sesuai permintaan eksplisit:

    s_GM(c) = sum_{j != c} ||W_c - W_j||_2

W_c = bobot filter DEPTHWISE channel c (bentuk asli [1, kH, kW], mis.
[1, 3, 3] untuk MobileNetV2), DIRATAKAN jadi vektor. Skor adalah JUMLAH
(bukan rata-rata) jarak Euclidean channel c terhadap SELURUH channel
lain j di lapisan yang sama -- dihitung lewat torch.norm (Euclidean,
p=2 bawaan) atas selisih tiap pasangan.

Interpretasi: skor besar = channel jauh dari semua channel lain = unik/
informatif. Skor kecil = channel dekat dengan banyak channel lain =
redundan. Arah skor ini SUDAH selaras dengan konvensi get_pruning_mask()
di src/pruning.py (memangkas skor TERENDAH pada tiap lapisan): skor
kecil = redundan = HARUS dipangkas = HARUS berskor rendah -- TIDAK perlu
dibalik tandanya, dipakai apa adanya.

Lapisan yang diproses SAMA seperti kriteria L1/BN/entropi di
src/pruning.py -- hanya blok inverted residual yang punya expansion
layer (t>1), diidentifikasi lewat _get_prunable_layers() (fungsi yang
SUDAH ADA di src/pruning.py, diimpor TIDAK DIUBAH, tidak ditulis ulang).

File ini BARU, TIDAK mengubah src/pruning.py atau modul lain manapun.
"""

import torch

from src.pruning import _get_prunable_layers


def compute_gm_scores(model):
    """
    Menghitung skor GM (jumlah jarak Euclidean ke seluruh channel lain di
    lapisan yang sama) untuk tiap channel intermediate yang bisa
    dipangkas, dari bobot filter DEPTHWISE CONV itu sendiri.

    s_GM(c) = sum_{j != c} ||W_c - W_j||_2

    Posisi dalam rantai komputasi: SISI HULU (penyebab), sama seperti
    L1 -- data-free, tidak butuh data kalibrasi.

    Args:
        model: MobileNetV2

    Returns:
        dict: {layer_idx: tensor skor GM per channel}
    """
    prunable = _get_prunable_layers(model)
    gm_scores = {}

    for layer_idx, (block_idx, dw_module) in enumerate(prunable):
        W = dw_module.weight.data  # (channels, 1, kH, kW)
        X = W.reshape(W.shape[0], -1).float()  # (channels, kH*kW) -- W_c diratakan

        # Jarak berpasangan lewat broadcasting + torch.norm (Euclidean).
        # diff[c, j] = W_c - W_j; diff[c, c] = 0 sehingga otomatis tidak
        # menyumbang apa pun saat dijumlahkan (memenuhi "j != c" tanpa
        # perlu masking eksplisit).
        diff = X.unsqueeze(1) - X.unsqueeze(0)          # (channels, channels, kH*kW)
        jarak = torch.norm(diff, p=2, dim=2)             # (channels, channels)
        skor = jarak.sum(dim=1)                          # sum_{j != c} ||W_c - W_j||

        gm_scores[layer_idx] = skor.cpu()

    print(f"[GM] Dihitung untuk {len(gm_scores)} lapisan depthwise (skip t=1)")
    return gm_scores
