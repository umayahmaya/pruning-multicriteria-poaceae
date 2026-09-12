# Panduan outputs/

Direorganisasi 2026-09-12: file arsip dipindah ke `outputs/archive/<kategori>/`
supaya tidak tertukar dengan hasil aktif saat penulisan Bab IV. Tidak ada file
yang dihapus dan tidak ada angka yang dihitung ulang -- murni pemindahan.

## Aktif (dipakai untuk hasil akhir tesis)

Tetap di `outputs/` langsung (bukan di `archive/`).

- **`multicriteria_per_rasio.json`** -- hasil I(c) FORMULA AKTIF (L1 +
  GM-Ekspansi + Entropi, bobot `w_k(r)` per rasio, CLAUDE.md Bagian 2) untuk
  ketujuh rasio: bobot yang dipakai, akurasi val/test, F1, params, ukuran,
  FLOPs, waktu inferensi. **Sumber utama bab hasil.**
  Sumber: `scripts/44_multicriteria_per_rasio.py`.
- **`multiseed_per_rasio_results.json`** -- validasi multi-seed (42/123/2024)
  formula aktif untuk seluruh 7 rasio + baseline (mean, std, min, maks).
  Sumber: `scripts/45_multiseed_per_rasio.py` (rasio 20%, 60%) dan
  `scripts/46_multiseed_remaining5.py` (rasio 10/30/40/50/70%).
- **`tabel_hasil_lengkap.json`** -- tabel 8 metrik (akurasi, precision,
  recall, F1, params, MB, FLOPs, inferensi) untuk baseline + L1/BN/Entropi
  tunggal + Multi-Kriteria FORMULA AWAL, x 7 rasio, dievaluasi ulang dalam
  satu protokol seragam. Kolom L1/BN/Entropi tunggal masih valid dikutip
  sebagai pembanding kriteria tunggal; kolom "multicriteria" adalah formula
  AWAL/ARSIP (S_BN, bobot konstan) -- untuk multi-kriteria FINAL kutip
  `multicriteria_per_rasio.json`, bukan kolom ini.
  Sumber: `scripts/17_generate_final_table.py`.
- **`ablation_val_results.json`** -- akurasi VALIDASI ablation L1/BN/Entropi
  tunggal per rasio. Dipakai formula AKTIF sebagai sumber akurasi L1 dan
  Entropi untuk menghitung bobot per-rasio (lihat `multicriteria_per_rasio.json`
  bagian `bobot_per_rasio`). Catatan: field `optimal_weights_val` di dalam
  file ini adalah skema bobot LAMA (rasio acuan tunggal, termasuk BN) dan
  TIDAK dipakai formula aktif -- hanya sub-field akurasi L1/Entropi per
  rasio yang masih terpakai.
  Sumber: `scripts/07_recompute_weights_val.py`.
- **`ablation_gm_expansion.json`** -- ablation kriteria GM-Ekspansi tunggal
  (val & test, 7 rasio), sumber akurasi GM untuk bobot per-rasio di atas.
  Sumber: `scripts/43_ablation_gm_expansion.py`.
- **`multiseed_results.json`** -- validasi multi-seed formula AWAL untuk
  baseline + rasio 20%/60%. Field `configs.baseline` dipakai ulang oleh
  `multiseed_per_rasio_results.json` (baseline tidak bergantung formula
  I(c), jadi tetap valid). Field `configs.multicriteria_20pct` dan
  `multicriteria_60pct` di file ini TIDAK dipakai lagi (formula awal,
  digantikan `multiseed_per_rasio_results.json`), dibiarkan di file ini
  apa adanya karena field `baseline` masih jadi dependensi aktif.
  Sumber: `scripts/15_multiseed_validation.py` (diarsipkan, lihat di bawah).
- **`citra_demo.json`** -- daftar citra demonstrasi (top-3 benar/kelas +
  salah prediksi), tidak terkait pilihan kriteria/bobot, masih dipakai
  untuk demo sidang. Sumber: `scripts/18_prepare_demo_images.py`.
- **`penanganan_penyakit.json`** -- info penanganan 9 kelas penyakit,
  tidak terkait kriteria/bobot, dipakai Panel Informasi Penanganan di
  `scripts/06_deploy_flask.py`.

### Gambar dan checkpoint ekspor aktif

- **`cm_baseline_9class.png`**, **`training_baseline.png`** -- confusion
  matrix dan kurva training baseline. Sumber: `scripts/02_train_baseline.py`.
- **`cm_multicriteria_per_rasio_{10-70}pct.png`** (7 file) -- confusion
  matrix formula AKTIF, satu per rasio. Sumber: `scripts/44_multicriteria_per_rasio.py`.
- **`kurva_kompresi_akurasi.png`** -- grafik rasio vs akurasi, 4 kurva
  (L1/BN/Entropi/Multi-Kriteria) dari `tabel_hasil_lengkap.json`. Kurva
  L1/BN/Entropi masih valid sebagai pembanding kriteria tunggal; kurva
  "Multi-Kriteria" di gambar ini formula AWAL/ARSIP (sama seperti kolom
  "multicriteria" di `tabel_hasil_lengkap.json`) -- belum ada versi kurva
  untuk formula aktif per-rasio. Sumber: `scripts/05_generate_report.py`.
- **`baseline_9class.pte`** -- ekspor ExecuTorch model baseline (tidak
  bergantung formula I(c), tetap berlaku). Sumber: `scripts/19_test_executorch_export.py`.
- **`multicriteria_20pct_30ep_valweights.pte`**, **`multicriteria_60pct_30ep_valweights.pte`**
  -- ekspor ExecuTorch checkpoint FORMULA AWAL. **AKTIF secara teknis**
  (masih disalin `scripts/21_export_android_assets.py` ke aset Android),
  BUKAN karena formulanya masih dianggap resmi -- ini instans KEDUA dari
  masalah "belum diselaraskan" yang sama seperti checkpoint default Flask
  di bawah: aplikasi Android saat ini JUGA mengirim model formula AWAL,
  bukan `multicriteria_per_rasio_20pct_30ep.pte` (belum pernah diekspor).

**Perhatian -- belum diselaraskan (dua tempat):**
1. Checkpoint default deployment Flask masih `multicriteria_20pct_30ep_valweights.pth`
   (formula AWAL), bukan `multicriteria_per_rasio_20pct_30ep.pth` (formula
   aktif) -- lihat `DEFAULT_CHECKPOINT` di `scripts/06_deploy_flask.py`.
2. Aset Android (`scripts/21_export_android_assets.py`) menyalin
   `multicriteria_{20,60}pct_30ep_valweights.pte` (formula AWAL) -- checkpoint
   formula aktif belum pernah diekspor ke `.pte` sama sekali.

Keduanya belum diganti karena memengaruhi demo/deployment yang berjalan,
perlu konfirmasi eksplisit sebelum diubah.

## Arsip

Tidak dihapus, hanya dipindah ke `outputs/archive/<kategori>/`. Nama file
tidak diubah. JANGAN dikutip sebagai hasil final di bab hasil.

### `archive/bn_gamma/` -- kriteria kedua masih S_BN (gamma batch norm)

Kriteria kedua I(c) diganti S_BN -> S_GM-Ekspansi pada 26 Agustus 2026
(CLAUDE.md Bagian 2, catatan versi). File berikut menghitung/membandingkan
skor dengan BN masih jadi salah satu dari tiga kriteria:

- `ablation_results.json` -- bobot w1/w2/w3 (L1/BN/Entropi) diturunkan dari
  akurasi UJI (kebocoran data, Aturan 1) DAN skema rasio-acuan-tunggal.
  Sumber: `scripts/03_ablation_study.py` (skrip TETAP AKTIF -- checkpoint
  L1/BN yang dihasilkannya masih dipakai `tabel_hasil_lengkap.json`, hanya
  JSON bobot-nya sendiri yang usang).
- `korelasi_per_lapisan.json` -- korelasi Spearman L1 vs BN vs Entropi per
  lapisan. Sumber: `scripts/archive/29_korelasi_per_lapisan.py`.
- `bobot_per_rasio_results.json` -- eksperimen bobot per-rasio PALING AWAL,
  masih 3 kriteria L1/BN/Entropi (rumus `Acc_k/sum(Acc)`, beda dari rumus
  final `d_k=Acc_k-min+0.01`). Sumber: `scripts/archive/32_bobot_per_rasio.py`.
- `bandingkan_mask_beta_results.json`, `multicriteria_beta20_results.json`
  -- varian bobot power-law dari file di atas, masih BN.
  Sumber: `scripts/archive/33_bandingkan_mask_beta.py`,
  `scripts/archive/34_multicriteria_beta20.py`.
- `channel_selection_comparison.json` -- korelasi Spearman I(c) vs kriteria
  tunggal formula AWAL (L1/BN/Entropi, bobot konstan), belum dihitung ulang
  untuk formula aktif. Sumber: `scripts/archive/08_compare_channel_selection.py`.
- `cm_multicriteria_beta20_50pct.png`, `cm_multicriteria_beta20_60pct.png`
  -- confusion matrix varian bobot power-law beta=20, masih BN. Sumber:
  `scripts/archive/34_multicriteria_beta20.py`.

### `archive/bobot_tunggal/` -- bobot konstan/rasio-acuan-tunggal (bukan per rasio)

Skema bobot per-rasio (`w_k(r)`, diturunkan dari akurasi validasi tiap
rasio) menggantikan bobot konstan pada 26 Agustus 2026. File berikut
memakai bobot TUNGGAL yang sama untuk ketujuh rasio:

- `multicriteria_results.json` -- pruning dengan bobot test-derived lama
  (`ablation_results.json`). Sumber: `scripts/04_pruning_multicriteria.py`
  (skrip TETAP AKTIF via mode `--load_weights`, hanya mode bobot-manual
  lamanya usang).
- `multicriteria_valweights_results.json`, `multicriteria_valweights_val_results.json`
  -- pruning dengan bobot val-derived (benar secara Aturan 1, tapi tetap
  konstan/rasio-acuan-tunggal). Sumber: `scripts/archive/14_eval_multicriteria_valweights_on_val.py`
  dan skrip `12_multicriteria_valweights.py` (skrip 12 TETAP AKTIF, dirujuk
  README.md sebagai bagian rantai reproduksi `tabel_hasil_lengkap.json`).
- `multiseed_remaining_results.json`, `multiseed_seed42.json`,
  `multiseed_seed123.json`, `multiseed_seed2024.json` -- validasi multi-seed
  formula AWAL untuk rasio selain 20%/60% dan rincian per-seed. Digantikan
  `multiseed_per_rasio_results.json`. Sumber: `scripts/archive/15_multiseed_validation.py`,
  `scripts/archive/16_multiseed_remaining_ratios.py`.
- `cm_multicriteria_{10-70}pct.png` (7 file, bobot test-derived, sumber
  `scripts/04_pruning_multicriteria.py`) dan `cm_multicriteria_{10-70}pct_valweights.png`
  (7 file, bobot val-derived konstan, sumber `scripts/12_multicriteria_valweights.py`)
  -- confusion matrix formula AWAL, digantikan `cm_multicriteria_per_rasio_{10-70}pct.png`.

### `archive/lain/` -- superseded karena alasan lain (bukan BN maupun bobot tunggal semata)

- `ablation_gm_saja.json` -- kriteria GM dari bobot DEPTHWISE conv
  (`src/archive/gm_scores.py`), varian GM yang TIDAK diadopsi -- formula
  final memakai GM-Ekspansi dari conv EKSPANSI (`scripts/43_ablation_gm_expansion.py`).
  Sumber: `scripts/archive/42_ablation_gm.py`.
- `skor_redundansi_results.json`, `kriteria_redundansi_tunggal_results.json`,
  `multikriteria_gm_results.json`, `bobot_berbasis_korelasi_results.json`
  -- keluarga eksperimen kriteria GM/COS berbasis jarak ke MEDIAN GEOMETRIK
  (algoritma Weiszfeld) dan skema bobot berbasis korelasi/grid-search --
  TIDAK diadopsi. Formula final memakai GM-Ekspansi jarak berpasangan
  (bukan median geometrik) dan bobot per-rasio dari selisih akurasi validasi
  (bukan korelasi/grid). Termasuk `cm_redundansi_cos_{40,50}pct.png`,
  `cm_redundansi_gm_{40,50}pct.png` (skrip 37) dan
  `cm_multikriteria_gm_config{A,B,C,D}_40pct.png` (skrip 38). Sumber:
  `scripts/archive/36_skor_redundansi.py`,
  `37_kriteria_redundansi_tunggal.py`, `38_multikriteria_gm.py`,
  `40_bobot_berbasis_korelasi.py`, `41_pencarian_kisi_bobot.py`.
- `alokasi_anggaran_entropi_preview_40pct.json`, `alokasi_anggaran_entropi_results.json`,
  `cm_alokasi_acak_40pct.png`, `cm_alokasi_entropi_40pct.png`
  -- skema alokasi rasio pemangkasan PER LAPISAN berbasis entropi (L1 murni
  memilih channel di dalam lapisan) -- alternatif dari WSM per-channel,
  tidak diadopsi. Sumber: `scripts/archive/30_alokasi_anggaran_entropi.py`.
  `Figure_1.png` -- duplikat tak bernama dari `cm_alokasi_entropi_40pct.png`
  (dikonfirmasi dari isi gambar: judul "CM Alokasi-Entropi 40%"), kemungkinan
  tersimpan tanpa sengaja lewat nama default matplotlib.
- `ablation_entropy_traincal_results.json` -- hasil ablation entropi
  (kalibrasi train, metodologi TIDAK berubah sampai sekarang) versi mentah
  sebelum standardisasi protokol evaluasi skrip 17 -- redundan dengan kolom
  "entropy" di `tabel_hasil_lengkap.json` yang mengevaluasi ulang checkpoint
  yang sama dengan protokol seragam. Sumber: `scripts/11_ablation_entropy_traincal.py`
  (skrip TETAP AKTIF -- checkpoint `ablation_entropy_*_traincal.pth` yang
  dihasilkannya masih dipakai `tabel_hasil_lengkap.json`, hanya JSON
  keluaran mentahnya sendiri yang redundan).
- `inference_remeasured.json` -- pengukuran latensi presisi terpisah,
  mendahului `scripts/17_generate_final_table.py` yang kini mengukur ulang
  dengan protokol identik (1 thread, 300 run) secara terintegrasi. Sumber:
  `scripts/archive/13_remeasure_inference.py`.
- `executorch_export_test.json`, `executorch_runtime_test.json` -- verifikasi
  ekspor ExecuTorch (4 Agustus 2026) dilakukan pada checkpoint FORMULA AWAL
  (`multicriteria_20pct_30ep_valweights.pth`, dkk.), sebelum formula aktif
  ada. Mekanisme ekspor ExecuTorch itu sendiri masih valid -- hanya hasil
  verifikasi kesetaraan prediksi ini yang terikat checkpoint lama dan perlu
  diulang pada checkpoint `multicriteria_per_rasio_*` kalau mau dikutip
  untuk deployment saat ini. Sumber: `scripts/19_test_executorch_export.py`,
  `scripts/20_test_executorch_runtime.py` (KEDUA skrip TETAP AKTIF sebagai
  alat/mekanisme, hanya hasil ujinya yang terikat checkpoint lama).

### Modul kode superseded (bukan file `outputs/`, dicatat di sini untuk keterkaitan)

- `src/archive/gm_scores.py` -- `compute_gm_scores()`, skor GM dari bobot
  DEPTHWISE conv. Hanya dipakai `scripts/archive/42_ablation_gm.py`. Formula
  final memakai `hitung_skor_gm_ekspansi_semua_layer()` di
  `scripts/43_ablation_gm_expansion.py` (dari conv EKSPANSI, TIDAK diarsipkan).
- `scripts/archive/*.py` (17 skrip: 08, 13, 14, 15, 16, 29-34, 36-38, 40-42)
  -- lihat rujukan sumber di setiap entri di atas. Skrip yang MASIH AKTIF
  meski sebagian keluarannya diarsipkan (03, 04, 06, 07, 11, 12,
  17, 19, 20) TIDAK dipindah -- checkpoint atau data yang dihasilkannya
  masih dipakai berkas aktif di atas.

## Checkpoint terkait (di `checkpoints/`, bukan `outputs/`, TIDAK dipindah)

- `ablation_entropy_{10-70}pct_30ep.pth` (tanpa akhiran `_traincal`) --
  kalibrasi entropi VAL lama, dasar `archive/bn_gamma/ablation_results.json`.
- `multicriteria_{10-70}pct_30ep.pth` (tanpa akhiran `_valweights`) --
  bobot test-derived lama, dasar `archive/bobot_tunggal/multicriteria_results.json`.
- `multicriteria_per_rasio_{10-70}pct_30ep.pth` -- checkpoint formula AKTIF.
  Dasar `multicriteria_per_rasio.json`.
- `multicriteria_per_rasio_{10-70}pct_30ep_seed{123,2024}.pth` (dan
  `_seed42.pth` untuk rasio 10/30/40/50/70%) -- checkpoint validasi
  multi-seed formula aktif. Dasar `multiseed_per_rasio_results.json`.
