# Panduan outputs/

Dokumen ini mencatat berkas mana yang menjadi SUMBER SAH untuk pelaporan
tesis, dan berkas mana yang merupakan ARSIP metodologi lama yang TIDAK BOLEH
dikutip di bab hasil.

**Status (10 September 2026):** formula I(c) berganti 26 Agustus 2026 --
S_BN diganti S_GM-Ekspansi, bobot konstan diganti bobot per-rasio (lihat
CLAUDE.md Bagian 2). Bagian "Sumber sah" di bawah sudah mencerminkan ini;
berkas hasil multi-kriteria dari formula SEBELUM 26 Agustus dipindah ke
tingkat "Arsip formula I(c) awal", bukan dihapus.

## Sumber sah (kutip ini)

- **multicriteria_per_rasio.json** -- hasil I(c) FORMULA AKTIF (L1 +
  GM-Ekspansi + Entropi, bobot `w_k(r)` per rasio, lihat CLAUDE.md Bagian 2)
  untuk ketujuh rasio: bobot yang dipakai, akurasi val dan test, F1,
  jumlah parameter, ukuran MB, FLOPs, waktu inferensi. **Sumber utama bab
  hasil untuk kontribusi metodologis multi-kriteria.** BELUM divalidasi
  multi-seed -- lihat CLAUDE.md Bagian 8 butir 9.
- **ablation_gm_expansion.json** -- ablation kriteria GM-Ekspansi tunggal
  (val dan test), sumber akurasi untuk menghitung bobot per-rasio di atas.
- **ablation_val_results.json** -- bobot w1/w2/w3 (skema lama, masih
  dipakai sebagai sumber akurasi L1 dan Entropi tunggal untuk bobot
  per-rasio formula baru), diturunkan dari akurasi VALIDASI (bukan uji).
- **tabel_hasil_lengkap.json** -- tabel 8 metrik lengkap untuk baseline
  dan keempat skenario (L1 saja, BN saja, Entropi saja, Multi-Kriteria
  FORMULA AWAL/ARSIP) x 7 rasio. Kolom L1/BN/Entropi tunggal masih valid
  dikutip sebagai pembanding kriteria tunggal; kolom "multicriteria" di
  file ini adalah formula LAMA (S_BN, bobot konstan) -- untuk hasil
  multi-kriteria FINAL kutip `multicriteria_per_rasio.json`, bukan kolom
  ini.
- **multiseed_results.json**, **multiseed_remaining_results.json**,
  **multiseed_seed{42,123,2024}.json** -- validasi multi-seed (mean, std)
  untuk baseline dan ketujuh rasio multi-kriteria FORMULA AWAL/ARSIP.
  Belum ada padanannya untuk formula per-rasio yang baru.
- **channel_selection_comparison.json** -- korelasi Spearman dan persentase
  channel berbeda, multi-kriteria vs kriteria tunggal, per rasio 10-70%
  (dihitung untuk formula awal/arsip, belum dihitung ulang untuk formula
  per-rasio).
- **inference_remeasured.json** -- pengukuran latensi presisi (1 thread,
  300 runs, median + std + jumlah runs).
- **ablation_entropy_traincal_results.json** -- hasil ablation entropi
  dengan kalibrasi train yang sudah diperbaiki.
- **kurva_kompresi_akurasi.png** -- grafik rasio pemangkasan vs akurasi,
  keempat skenario dalam satu gambar (formula awal/arsip).
- **penanganan_penyakit.json** -- informasi penanganan untuk kesembilan
  kelas (nama penyakit, patogen, tipe patogen, ringkasan, penanganan,
  pencegahan, sumber), dipakai Panel Informasi Penanganan di
  `06_deploy_flask.py`. Disusun dari sumber resmi lembaga penelitian dan
  penyuluhan pertanian (IRRI, universitas, lembaga extension). TIDAK
  memuat rekomendasi merek pestisida maupun dosis -- lihat kunci
  `_catatan` di dalam berkas untuk keterangan lengkapnya.

## Arsip formula I(c) awal (S_BN, bobot konstan -- dipakai sampai awal Agustus 2026)

Masih berguna sebagai bagian dari riwayat metodologi dan sebagai hasil
kriteria tunggal (L1/BN/Entropi) di `tabel_hasil_lengkap.json`, TAPI kolom
"multicriteria" di dalamnya BUKAN LAGI hasil final -- lihat CLAUDE.md
Bagian 2a.

- **multicriteria_valweights_results.json** -- hasil pruning multi-kriteria
  formula awal (bobot konstan val-derived, seed 42), evaluasi test.
- **multicriteria_valweights_val_results.json** -- evaluasi checkpoint di
  atas pada subset val.

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
  `multicriteria_valweights_results.json`, yang sendiri kini juga arsip
  (lihat di atas).

## Checkpoint terkait (di checkpoints/, bukan outputs/)

- `ablation_entropy_{10-70}pct_30ep.pth` (tanpa akhiran `_traincal`) --
  kalibrasi entropi VAL lama. Dasar kolom "entropy" di `ablation_results.json`.
- `multicriteria_{10-70}pct_30ep.pth` (tanpa akhiran `_valweights`) --
  bobot test-derived lama. Dasar `multicriteria_results.json`.
- `multicriteria_per_rasio_{10-70}pct_30ep.pth` -- checkpoint formula AKTIF
  (bobot per-rasio, S_GM-Ekspansi). Dasar `multicriteria_per_rasio.json`.
- **Perhatian -- belum diselaraskan:** checkpoint default deployment masih
  `multicriteria_20pct_30ep_valweights.pth` (formula AWAL/ARSIP, lihat
  `DEFAULT_CHECKPOINT` di `scripts/06_deploy_flask.py`), BUKAN
  `multicriteria_per_rasio_20pct_30ep.pth` (formula aktif). Belum diganti
  di sini karena mengubah checkpoint default memengaruhi demo/deployment
  yang sedang berjalan -- perlu konfirmasi eksplisit sebelum diubah.
