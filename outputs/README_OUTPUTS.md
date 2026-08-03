# Panduan outputs/

Dokumen ini mencatat berkas mana yang menjadi SUMBER SAH untuk pelaporan
tesis, dan berkas mana yang merupakan ARSIP metodologi lama yang TIDAK BOLEH
dikutip di bab hasil.

## Sumber sah (kutip ini)

- **tabel_hasil_lengkap.json** -- tabel utama 8 metrik lengkap (akurasi,
  precision, recall, F1, jumlah parameter, ukuran MB, FLOPs, waktu
  inferensi) untuk baseline dan keempat skenario (L1 saja, BN saja, Entropi
  saja, Multi-Kriteria) x 7 rasio, dievaluasi ulang dalam satu protokol
  seragam. Sumber utama bab hasil.
- **ablation_val_results.json** -- bobot w1/w2/w3 final (`optimal_weights_val`),
  diturunkan dari akurasi VALIDASI (bukan uji). Sumber bobot yang benar.
- **multicriteria_valweights_results.json** -- hasil pruning multi-kriteria
  dengan bobot val-derived (seed 42), evaluasi test.
- **multicriteria_valweights_val_results.json** -- evaluasi checkpoint
  multi-kriteria (bobot val) pada subset val.
- **multiseed_results.json**, **multiseed_remaining_results.json**,
  **multiseed_seed{42,123,2024}.json** -- validasi multi-seed (mean, std)
  untuk baseline dan ketujuh rasio multi-kriteria.
- **channel_selection_comparison.json** -- korelasi Spearman dan persentase
  channel berbeda, multi-kriteria vs kriteria tunggal, per rasio 10-70%.
- **inference_remeasured.json** -- pengukuran latensi presisi (1 thread,
  300 runs, median + std + jumlah runs).
- **ablation_entropy_traincal_results.json** -- hasil ablation entropi
  dengan kalibrasi train yang sudah diperbaiki.
- **kurva_kompresi_akurasi.png** -- grafik rasio pemangkasan vs akurasi,
  keempat skenario dalam satu gambar.
- **penanganan_penyakit.json** -- informasi penanganan untuk kesembilan
  kelas (nama penyakit, patogen, tipe patogen, ringkasan, penanganan,
  pencegahan, sumber), dipakai Panel Informasi Penanganan di
  `06_deploy_flask.py`. Disusun dari sumber resmi lembaga penelitian dan
  penyuluhan pertanian (IRRI, universitas, lembaga extension). TIDAK
  memuat rekomendasi merek pestisida maupun dosis -- lihat kunci
  `_catatan` di dalam berkas untuk keterangan lengkapnya.

## Arsip metodologi lama (JANGAN dikutip untuk bab hasil)

- **ablation_results.json** -- bobot w1/w2/w3 diturunkan dari akurasi UJI
  (kebocoran data, lihat CLAUDE.md butir 4.1). Kolom kriteria entropi
  memakai kalibrasi VAL lama (num_batches=10), bukan kalibrasi train yang
  sudah diperbaiki. Masih dirujuk `07_recompute_weights_val.py` dan
  `08_compare_channel_selection.py` khusus untuk tabel perbandingan
  "bobot lama vs bobot baru" -- JANGAN dihapus, tapi jangan dikutip
  sebagai bobot final.
- **multicriteria_results.json** -- hasil pruning dengan bobot test-derived
  lama di atas. Sudah sepenuhnya digantikan
  `multicriteria_valweights_results.json` / `tabel_hasil_lengkap.json`.

## Checkpoint terkait (di checkpoints/, bukan outputs/)

- `ablation_entropy_{10-70}pct_30ep.pth` (tanpa akhiran `_traincal`) --
  kalibrasi entropi VAL lama. Dasar kolom "entropy" di `ablation_results.json`.
- `multicriteria_{10-70}pct_30ep.pth` (tanpa akhiran `_valweights`) --
  bobot test-derived lama. Dasar `multicriteria_results.json`.
- Checkpoint default deployment: `multicriteria_20pct_30ep_valweights.pth`
  (lihat `DEFAULT_CHECKPOINT` di `scripts/06_deploy_flask.py`).
