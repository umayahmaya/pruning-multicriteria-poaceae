# Konteks Proyek Penelitian Tesis

Dokumen ini dibaca otomatis oleh Claude Code pada setiap sesi baru. Isinya adalah konteks penelitian, aturan metodologis yang tidak boleh dilanggar, dan konvensi kerja pada repositori ini.

---

## 1. Identitas Penelitian

**Peneliti:** Nurul Umayah Hafilda (Maya)
**Program:** S2 Teknik Informatika, Universitas Hasanuddin
**Pembimbing:** Prof. Zahir

**Judul tesis:**
Model Klasifikasi Penyakit Daun Tanaman Famili Poaceae Menggunakan Pruning Terstruktur Berbasis Skoring Multi-Kriteria pada MobileNetV2

**Kontribusi metodologis utama:**
Fungsi skoring kepentingan channel berbasis multi-kriteria (L1-norm, jarak antar-channel conv ekspansi/GM, entropi Shannon) untuk pruning terstruktur pada MobileNetV2, bobot per-kriteria diadaptasi dari Weighted Sum Model (Fishburn, 1967) dan sejak 26 Agustus 2026 diturunkan secara adaptif per rasio pemangkasan dari akurasi validasi (lihat Bagian 2).

---

## 2. Metodologi Inti

**Status (10 September 2026):** bagian ini mendeskripsikan formula yang AKTIF dipakai sebagai kontribusi metodologis tesis sejak 26 Agustus 2026. Formula sebelumnya (dipakai sampai awal Agustus 2026, menghasilkan `outputs/tabel_hasil_lengkap.json`) diarsipkan di Bagian 2a untuk ketertelusuran, bukan dihapus.

### Fungsi skoring

```
I(c) = w_L1(r) * S_L1(c) + w_GM(r) * S_GM(c) + w_Ent(r) * S_H(c)
```

Ketiga komponen dinormalisasi min-max per layer sebelum digabungkan, sehingga `I(c)` berada pada rentang [0, 1] sebagai convex combination. Skor mentah S_L1/S_GM/S_H sendiri TIDAK bergantung rasio pemangkasan `r` -- dihitung sekali dari `checkpoints/baseline_9class.pth` dan dipakai ulang untuk ketujuh rasio. Yang bergantung `r` hanyalah bobot `w_k(r)` (lihat subbagian berikut) dan rasio pemangkasan itu sendiri.

| Simbol | Kriteria | Sumber nilai | Sifat |
|---|---|---|---|
| S_L1 | L1-norm bobot filter | Bobot depthwise conv 3x3 | Data-free |
| S_GM | Skor GM-Ekspansi: `sum_{j != c} \|\|W_c - W_j\|\|_2` (jumlah jarak Euclidean berpasangan ke seluruh channel lain di layer yang sama; kecil = redundan = dipangkas) | Bobot conv ekspansi 1x1 (sebelum depthwise) | Data-free |
| S_H | Entropi Shannon feature map | Output feature map, 256 bin | Butuh data kalibrasi |

**Catatan versi:** kriteria kedua semula S_BN (skala gamma batch norm setelah depthwise), diganti S_GM-Ekspansi pada 26 Agustus 2026 (`scripts/43_ablation_gm_expansion.py`), setelah eksplorasi lanjutan pertengahan Agustus 2026 (skrip `22` sampai `43`, belum di-commit ke git) membandingkan beberapa varian kriteria berbasis redundansi channel -- termasuk GM dari bobot depthwise (`src/archive/gm_scores.py`, `scripts/archive/42_ablation_gm.py`), GM/kosinus dari conv ekspansi terhadap median geometrik (`scripts/archive/36_skor_redundansi.py`), korelasi antar-lapisan, dan skema alokasi anggaran kalibrasi entropi. S_GM-Ekspansi (jarak berpasangan dari bobot conv ekspansi) yang akhirnya diadopsi.

### Bobot: per-rasio, diturunkan dari akurasi validasi ablation kriteria tunggal

Sejak 26 Agustus 2026 (`scripts/44_multicriteria_per_rasio.py`), bobot TIDAK LAGI konstan di seluruh rasio (beda dari versi awal, Bagian 2a). Untuk tiap rasio pemangkasan `r`, bobot diturunkan dari selisih akurasi VALIDASI ketiga ablation kriteria tunggal (L1, GM-Ekspansi, Entropi) pada rasio `r` yang sama:

```
d_k(r) = Acc_k(r) - min(Acc_L1(r), Acc_GM(r), Acc_Ent(r)) + 0.01
w_k(r) = d_k(r) / (d_L1(r) + d_GM(r) + d_Ent(r))
```

(selisih akurasi terhadap kriteria terlemah pada rasio itu, plus 0,01 supaya kriteria terlemah tetap mendapat bobot kecil positif -- bukan nol). Konsisten dengan Aturan 1 (Bagian 4): akurasi validasi dipakai untuk keputusan bobot, akurasi test HANYA untuk pelaporan akhir dan TIDAK PERNAH dipakai memilih apa pun (dicatat eksplisit di `outputs/multicriteria_per_rasio.json`).

Sumber akurasi ablation kriteria tunggal: L1 dan Entropi dari `outputs/ablation_val_results.json` (tidak berubah dari versi awal), GM-Ekspansi dari `outputs/ablation_gm_expansion.json`.

| Rasio | w_L1 | w_GM | w_Ent |
|---|---|---|---|
| 10% | 0,333 | 0,333 | 0,333 |
| 20% | 0,280 | 0,280 | 0,441 |
| 30% | 0,455 | 0,333 | 0,212 |
| 40% | 0,544 | 0,333 | 0,122 |
| 50% | 0,466 | 0,466 | 0,069 |
| 60% | 0,170 | 0,562 | 0,268 |
| 70% | 0,086 | 0,383 | 0,531 |

Hasil lengkap per rasio (bobot, akurasi val/test, params, ukuran model, FLOPs, waktu inferensi): `outputs/multicriteria_per_rasio.json`.

### 2a. Formula versi awal (diarsipkan, bukan lagi metodologi resmi)

Dipakai sampai awal Agustus 2026, menghasilkan `outputs/tabel_hasil_lengkap.json` (checkpoint `multicriteria_*_valweights.pth`). Skrip yang mengimplementasikannya (`03`, `04`, `07`, `12`) TIDAK dihapus dan tetap reproducible, tapi jangan dianggap sebagai hasil akhir tesis tanpa instruksi eksplisit.

```
I(c) = w1 * S_L1(c) + w2 * S_BN(c) + w3 * S_H(c)   (bobot KONSTAN di semua rasio)
```

S_BN = skala gamma batch norm pada layer BN setelah depthwise conv, data-free.

```
w1 = 0.3361   (L1-norm)      -- setelah koreksi val, lihat Bagian 8 butir 1
w2 = 0.3320   (BN gamma)
w3 = 0.3320   (Entropi)
```

Bobot sebelum koreksi (keliru, diturunkan dari akurasi TEST): w1=0,3401 / w2=0,3259 / w3=0,3340.

### Lokasi pemangkasan

Pruning hanya diterapkan pada **channel intermediate di dalam inverted residual block**, yaitu hasil expansion conv 1x1 yang diteruskan ke depthwise conv 3x3 lalu ke projection conv 1x1.

Jumlah channel input dan output block **tidak pernah diubah**. Konsekuensinya, koneksi residual tetap valid dan tidak ada dependency yang merambat antar block.

Ketika satu channel intermediate dipangkas, tiga hal harus dihapus bersamaan:
1. Baris bobot pada expansion conv
2. Parameter BN terkait (gamma, beta, running_mean, running_var)
3. Grup depthwise dan kolom input pada projection conv

---

## 3. Dataset

Sembilan kelas dari tiga tanaman famili Poaceae, digabung dalam satu model.

**Urutan alfabetis kelas:**
1. Brown_Spot_Rice
2. Common_Rust_Corn
3. Gray_Leaf_Spot_Corn
4. Healthy_Corn
5. Healthy_Rice
6. Healthy_Sugarcane
7. Leaf_Blast_Rice
8. Mosaic_Sugarcane
9. Rust_Sugarcane

**Komposisi:** 129 gambar per kelas, total 1.161 gambar
**Split:** 70 / 15 / 15 stratified, seed = 42
**Jumlah gambar uji:** sekitar 174, sehingga satu gambar setara 0,57 persen akurasi

**Sumber dataset:**
- Padi: Hasan (2023), DOI 10.17632/hx6f852hw4.2
- Tebu: Daphal dan Koli (2022), DOI 10.17632/9424skmnrk.1
- Jagung: Ahmad (2025), DOI 10.17632/vy629dngm8.1

---

## 4. Aturan Metodologis yang Tidak Boleh Dilanggar

Aturan berikut bersifat mengikat. Jika menemukan kode yang melanggar salah satunya, laporkan sebelum mengubah apa pun.

1. **Bobot ablation harus diturunkan dari akurasi data validasi, bukan data uji.** Menurunkan bobot dari data uji lalu mengevaluasi pada data uji yang sama adalah kebocoran data dan penalaran melingkar.

2. **Gambar kalibrasi entropi harus berasal dari data latih.** Tidak boleh dari data validasi maupun data uji.

3. **Jumlah gambar kalibrasi harus proporsional per kelas.** Untuk konfigurasi sembilan kelas, gunakan sekitar 25 gambar per kelas.

4. **Total epoch baseline dan model hasil pruning harus setara** agar perbandingan adil.

5. **Klaim akurasi harus mempertimbangkan resolusi 0,57 persen per gambar.** Selisih di bawah 2 persen tidak boleh diklaim sebagai peningkatan tanpa pengujian multi-seed.

6. **Pruning harus terstruktur, bukan unstructured.** Dimensi tensor harus benar-benar mengecil karena klaim penelitian mencakup FLOPs dan waktu inferensi, bukan hanya ukuran file.

7. **Setiap sitasi wajib disertai DOI yang sudah diverifikasi.** Dilarang menyebut paper, jurnal, atau penulis tanpa DOI yang bisa ditelusuri.

---

## 5. Metode yang Sudah Dibuang dari Metodologi Final

Jangan menyarankan atau menambahkan kembali metode berikut tanpa instruksi eksplisit.

- **Knowledge distillation.** Sudah diuji pada rasio 30, 40, dan 50 persen. Hasilnya konsisten menurunkan akurasi. Hasil negatif ini didokumentasikan sebagai temuan, bukan kegagalan.
- **Quantization INT8.** Dihapus dari metodologi final atas instruksi pembimbing. Data eksperimen lama disimpan sebagai arsip saja. Bisa kembali hanya jika target deployment berubah ke ESP32.

---

## 6. Struktur Repositori

Lokasi proyek: `D:\S2 UMAYAH\TESIS\Pra Penelitian 2`

```
Pra Penelitian 2/
  src/                    modul inti (config, dataset, model, pruning, visualize)
  scripts/                skrip eksekusi bernomor 01 sampai 06 dan test_system
  dataset/                data mentah sembilan kelas sebelum dipecah
  dataset_split/          hasil split 70/15/15 stratified, seed 42
  checkpoints/            file .pth hasil training
  outputs/                tabel, grafik, confusion matrix, laporan
  logs/                   catatan proses training per epoch
  venv/                   virtual environment
  CLAUDE.md               dokumen ini
  README.md
  requirements.txt
  .gitignore
```

Catatan penting soal lokasi file:
- Skrip bernomor berada di dalam `scripts/`, bukan di root. Jalankan dari root proyek agar path relatif tetap benar.
- `dataset/` berisi data mentah, sedangkan `dataset_split/` berisi hasil pembagian. Jangan tertukar. Gambar kalibrasi entropi harus diambil dari bagian train pada `dataset_split/`.
- Hasil eksperimen disimpan di `outputs/`, bukan `results/`.

**Lingkungan:** Python dengan virtual environment di `venv/`, training pada CPU (laptop Windows), eksperimen berat dipindah ke Google Colab.
**Deployment:** Flask untuk antarmuka web.
**Repositori:** lihat Bagian 11 untuk konvensi remote Git yang berlaku saat ini.

---

## 7. Konvensi Kerja

- Semua perubahan konfigurasi masuk lewat `src/config.py`, jangan menulis nilai konstanta langsung di skrip.
- **Jangan mengubah file di `src/` atau `checkpoints/` selagi ada proses training berjalan di terminal.** Training pada CPU memakan waktu sangat lama dan tidak boleh terganggu.
- Jangan menghapus atau menimpa file di `checkpoints/` tanpa konfirmasi.
- Setiap eksperimen harus reproducible, jadi seed tetap 42 kecuali sedang menjalankan uji multi-seed.
- Jangan menghapus isi `dataset_split/` tanpa konfirmasi, karena pembagian data harus konsisten di seluruh eksperimen.
- Dokumen akhir memakai python-docx, bukan pandoc, karena pandoc merusak format tabel.
- Format dokumen tesis: B5 ukuran 17,60 x 25,00 cm, Arial 10pt, spasi 1,15, margin 2,25 cm.

---

## 8. Pekerjaan yang Masih Terbuka

1. ~~Verifikasi apakah bobot ablation diturunkan dari akurasi validasi atau akurasi uji, perbaiki jika perlu.~~ **SELESAI.** Bobot lama (`outputs/archive/bn_gamma/ablation_results.json`) ternyata diturunkan dari akurasi test. Diperbaiki lewat `07_recompute_weights_val.py`, yang menghasilkan `optimal_weights_val` dari akurasi val di `outputs/ablation_val_results.json`. Dipakai oleh `12_multicriteria_valweights.py` dan kini juga oleh `04_pruning_multicriteria.py --load_weights`.
2. ~~Sesuaikan jumlah gambar kalibrasi entropi untuk konfigurasi sembilan kelas.~~ **SELESAI.** `compute_entropy_scores()` di `src/pruning.py` diperbaiki: 20 citra/kelas (180 total untuk 9 kelas), diambil dari train split (bukan val), deterministik, transform evaluasi tanpa augmentasi. Diverifikasi lewat `10_smoke_test_entropy.py`, dipakai ulang untuk ablation entropi di `11_ablation_entropy_traincal.py`.
3. ~~Hitung korelasi peringkat (Spearman atau Kendall tau) antara I(c) dan tiap kriteria tunggal, serta persentase channel yang berbeda pilihannya pada rasio 30 persen.~~ **SELESAI.** Dihitung di `scripts/archive/08_compare_channel_selection.py`, diperluas ke seluruh 7 rasio (bukan hanya 30%). Hasil di `outputs/archive/bn_gamma/channel_selection_comparison.json`.
4. **TIDAK DIKERJAKAN.** Eksperimen kontrol dengan bobot rata sepertiga sebagai pembanding tidak dijalankan terpisah. Bobot hasil ablation (w1=0,3361, w2=0,3320, w3=0,3320) sudah sangat dekat dengan sepertiga. Uji sensitivitas bobot di `scripts/archive/08_compare_channel_selection.py` (perbandingan bobot lama vs baru, selisih hingga ~0,006 per bobot) menunjukkan perubahan bobot sebesar itu hanya mengubah 0,14 sampai 0,39 persen pilihan channel dengan Spearman 0,9999 di seluruh 7 rasio (lihat `outputs/archive/bn_gamma/channel_selection_comparison.json`), sehingga eksperimen kontrol bobot rata sepertiga terpisah diperkirakan tidak akan memberi informasi baru.
5. ~~Verifikasi kesetaraan total epoch antara baseline dan model hasil pruning.~~ **SELESAI.** `BASELINE_EPOCHS` dan `FINETUNE_EPOCHS` di `src/config.py` sama-sama 30.
6. ~~Pertimbangkan pengujian tiga seed pada konfigurasi optimal untuk melaporkan rata-rata dan simpangan baku.~~ **SELESAI.** Dijalankan di `scripts/archive/15_multiseed_validation.py` (baseline, rasio 20%, rasio 60%; seed 42/123/2024) dan `scripts/archive/16_multiseed_remaining_ratios.py` (rasio 10/30/40/50/70%; seed 123/2024, seed 42 dipakai ulang dari `12_multicriteria_valweights.py`). Hasil di `outputs/multiseed_results.json`, `outputs/archive/bobot_tunggal/multiseed_remaining_results.json`, dan `outputs/archive/bobot_tunggal/multiseed_seed{42,123,2024}.json`.
7. ~~(Prioritas sedang, ditemukan 2026-08-03) Alur utama 01→06 di README.md tidak mandiri tanpa menjalankan skrip 07 dulu, karena 04_pruning_multicriteria.py --load_weights mensyaratkan outputs/ablation_val_results.json yang hanya dihasilkan skrip 07.~~ **SELESAI (2026-08-03).** README.md Bagian "Urutan Menjalankan" diperbarui: skrip 07 dimasukkan ke alur utama di antara 03 dan 04 (urutan menjadi 01, 02, 03, 07, 04, 05, 06), dengan penjelasan bahwa 07 wajib dijalankan karena bobot harus diturunkan dari akurasi validasi sebelum pruning. Skrip 08, 10 sampai 18 tetap didokumentasikan sebagai koreksi metodologis lanjutan (11, 12) dan analisis lanjutan (sisanya), di luar alur utama.
8. ~~(ditemukan 2026-08-04) executorch.runtime.Runtime gagal dimuat di venv/ untuk uji ekspor ExecuTorch, sehingga kesamaan prediksi dan selisih logit belum terverifikasi.~~ **SELESAI (2026-08-04).** `19_test_executorch_export.py` (di `venv/`) memverifikasi poin 1 (ekspor berhasil) dan poin 2 (ukuran berkas) untuk baseline/rasio 20%/rasio 60% -- 8,734 MB, 7,338 MB, 4,537 MB. `20_test_executorch_runtime.py` (di `venv_mobile/`) melengkapi poin 3 dan 4: untuk ketiga model, 20/20 prediksi PyTorch vs ExecuTorch identik, selisih logit maksimum di orde 1e-05 (noise floating-point). Hasil di `outputs/archive/lain/executorch_runtime_test.json`.

   Akar masalah: executorch 1.3.1 (versi terbaru di PyPI) memiliki modul native `_portable_lib` yang gagal dimuat di Windows untuk SEMUA kombinasi torch yang dicoba (torch 2.13.0+cpu yang terpasang di `venv/`, torch 2.13.0 default PyPI, torch nightly 2.14.0.dev) -- kemungkinan cacat pada wheel Windows rilis itu sendiri, bukan soal versi torch. executorch 1.0.1 dengan torch 2.9.1+cpu (versi yang secara eksplisit ia syaratkan) terbukti berfungsi.

   `venv_mobile/` (gitignored, tidak di-commit) adalah lingkungan Python 3.11 terpisah khusus untuk ini. `venv/` (lingkungan penelitian utama) TIDAK diubah lagi untuk mengejar kecocokan versi torch dengan executorch (tidak dipasangi torch nightly dsb.) -- satu-satunya perubahan di `venv/` adalah instalasi `executorch==1.3.1` itu sendiri (untuk `19_test_executorch_export.py`), yang sebagai efek samping menurunkan `scikit-learn` dari 1.9.0 ke 1.7.1 (dependensi transitif `torchao`). Hanya `train_test_split` dan `sklearn.metrics` yang dipakai di proyek ini, API yang stabil di kedua versi. `test_system.py` dijalankan ulang setelah instalasi dan tetap lulus 75 dari 75. `requirements.txt` diperbarui lewat `pip freeze` untuk mencerminkan kondisi `venv/` yang sekarang.

   Cara reproduksi `venv_mobile/`:
   ```
   py -3.11 -m venv venv_mobile
   venv_mobile/Scripts/python.exe -m pip install "executorch==1.0.1"
   venv_mobile/Scripts/python.exe -m pip install "torchvision==0.24.1"
   ```
   (torch 2.9.1+cpu terpasang otomatis sebagai dependensi executorch==1.0.1). `flatc` (kompilator FlatBuffers, dibutuhkan saat serialisasi .pte) di-resolve otomatis lewat env var `FLATC_EXECUTABLE` di dalam `20_test_executorch_runtime.py` sendiri karena `venv_mobile/Scripts` tidak selalu ada di PATH -- tidak perlu setup manual tambahan.

9. **SELESAI (2026-08-26), didokumentasikan di sini 2026-09-10, diperbarui 2026-09-11 setelah validasi multi-seed 20%/60%, DIKOREKSI 2026-09-24 setelah ditemukan cakupan sebenarnya sudah lengkap 7 rasio.** Kriteria kedua I(c) diganti dari S_BN ke S_GM-Ekspansi dan skema bobot diganti dari konstan ke per-rasio -- lihat Bagian 2 (formula aktif) dan Bagian 2a (formula lama, diarsipkan). Hasil seed 42 (tarikan tunggal, RNG sekuensial -- lihat Bagian 9 keterbatasan metodologis): `outputs/multicriteria_per_rasio.json`. Hasil validasi 3 seed (42/123/2024): `outputs/multiseed_per_rasio_results.json`, dihasilkan `scripts/45_multiseed_per_rasio.py` (rasio 20%/60%, 3,15 jam) + `scripts/46_multiseed_remaining5.py` (rasio 10/30/40/50/70%, 11,58 jam) -- **total 14,73 jam, KEDUANYA SUDAH SELESAI, mencakup ketujuh rasio + baseline**, bukan cuma 3 titik seperti sempat tertulis di versi entri ini sebelum 2026-09-24 (kesalahan dokumentasi murni -- file hasilnya sendiri sudah berisi ketujuh rasio sejak sebelum tanggal itu, ke-10 checkpoint seed123/seed2024 untuk rasio 10/30/40/50/70% terverifikasi ada di `checkpoints/`, CLAUDE.md-nya saja yang tidak diperbarui).

   **Tabel lengkap 3-seed vs baseline (95,43%, std 0,81 pp, dipakai ulang dari `scripts/archive/15_multiseed_validation.py` -- baseline tidak bergantung formula I(c)), Aturan 5 ambang 2 pp:**

   | Rasio | Mean 3-seed | Std | Selisih vs baseline | Ambang 2 pp |
   |---|---|---|---|---|
   | 10% | 95,43% | 0,47 pp | -0,00 pp | dalam ambang (setara) |
   | 20% | 96,19% | 0,97 pp | +0,76 pp | dalam ambang (setara) |
   | 30% | 96,38% | 0,27 pp | +0,95 pp | dalam ambang (setara) |
   | 40% | 96,19% | 0,27 pp | +0,76 pp | dalam ambang (setara) |
   | 50% | 92,76% | 1,43 pp | -2,67 pp | **MELEBIHI (turun)** |
   | 60% | 89,33% | 0,54 pp | -6,10 pp | **MELEBIHI (turun)** |
   | 70% | 88,76% | 0,27 pp | -6,67 pp | **MELEBIHI (turun)** |

   **Klaim "rasio 20% melampaui baseline" DICABUT** (per-seed 20%: 97,14% / 94,86% / 96,57% -- angka 97,14% seed 42 saja TIDAK representatif sendirian, sudah dicabut sejak versi entri sebelumnya, tetap berlaku).

   **Rasio 60% menunjukkan penurunan ~6,1 pp yang konsisten di ketiga seed** (per-seed: 89,71% / 88,57% / 89,71%), bukan artefak seed tunggal -- sudah didokumentasikan sejak versi entri sebelumnya, tetap berlaku.

   **Temuan BARU (baru tersurfaces 2026-09-24 karena koreksi cakupan di atas): rasio 50% JUGA melebihi ambang 2 pp (-2,67 pp), bukan cuma 60% dan 70%.** Per-seed 50%: 90,86% / 93,14% / 94,29% (std tertinggi dari ketujuh rasio, 1,43 pp -- lihat Bagian 8 butir 10 soal kenapa std ini tidak diwariskan dari kriteria tunggal manapun). Data mentah disajikan apa adanya -- TIDAK ditarik kesimpulan di sini apakah ini mengubah letak "jurang performa" (performance cliff) yang mendasari H2 dari "antara 50% dan 60%" menjadi "mulai dari 50%"; itu keputusan interpretasi untuk pembimbing/penulis tesis.

   **Catatan metodologis penting: nilai seed-42 "resmi" di `multicriteria_per_rasio.json` (dipakai di README.md, tabel-tabel lain) BERBEDA dari nilai seed-42 di validasi multi-seed ini untuk rasio 10/30/40/50/70%**, karena kelima rasio itu di-retrain FRESH dengan seed di-reset eksplisit untuk validasi multi-seed (`catatan` per-seed di JSON: "DILATIH ULANG FRESH, BUKAN dipakai ulang"), sedangkan nilai "resmi" berasal dari tarikan sekuensial `44_multicriteria_per_rasio.py` (RNG bergantung rasio sebelumnya -- keterbatasan metodologis Bagian 9). Selisihnya bervariasi per rasio: 10% identik (94,86% vs 94,86%), 70% hampir identik (88,00% vs 88,57%), tapi 30% berbeda 0,57 pp, 40% berbeda 1,71 pp, dan **50% berbeda 3,43 pp** (94,29% resmi vs 90,86% retrain fresh) -- ilustrasi konkret seberapa besar dampak keterbatasan RNG-sekuensial yang sudah lama didokumentasikan di Bagian 9, baru sekarang terkuantifikasi angkanya.

   **Item lama di bagian "masih terbuka", status diperbarui:**
   - ~~Seluruh pekerjaan eksplorasi 10-26 Agustus 2026 ... belum pernah di-commit ke git~~ **SELESAI (2026-09-10).** Di-commit ke `main` (`6a6aad8`, `14b5d01`, `0b0a1a1`) dan dipublikasikan ke `publish` (`ddd9bd2`).
   - ~~README.md ... belum menyebut skrip 43/44 sama sekali~~ **SELESAI (2026-09-10).** Bagian "Hasil Utama" dan "Urutan Menjalankan" README.md diperbarui untuk mencakup formula aktif dan rantai skrip 42-44.
   - ~~`outputs/archive/bn_gamma/channel_selection_comparison.json` (Bagian 8 butir 3) dan uji sensitivitas bobot terkait masih hanya membandingkan bobot konstan lama vs lebih lama~~ **SELESAI (2026-09-24).** Dihitung ulang untuk formula aktif lewat `scripts/54_perbandingan_seleksi_channel.py`. Hasil: `outputs/channel_selection_comparison_per_rasio.json`. Data mentah saja, tanpa interpretasi -- keputusan penafsiran diserahkan ke pembimbing/penulis tesis:

     **A. Sensitivitas bobot** (bobot ablation per-rasio vs bobot rata 1/3-1/3-1/3, korelasi Spearman I(c) dan persentase channel yang beda status pangkas/pertahankan):
     | Rasio | w_L1/w_GM/w_Ent | Channel beda | Persentase | Spearman rho |
     |---|---|---|---|---|
     | 10% | 0,333/0,333/0,333 | 0 | 0,00% | 1,0000 |
     | 20% | 0,280/0,280/0,441 | 240 | 3,38% | 0,9792 |
     | 30% | 0,455/0,333/0,212 | 536 | 7,55% | 0,9495 |
     | 40% | 0,544/0,333/0,122 | 1506 | 21,20% | 0,8424 |
     | 50% | 0,466/0,466/0,069 | 2106 | 29,65% | 0,7849 |
     | 60% | 0,170/0,562/0,268 | 848 | 11,94% | 0,9018 |
     | 70% | 0,086/0,383/0,531 | 724 | 10,19% | 0,9144 |

     **B. Multi-kriteria (bobot per-rasio) vs kriteria tunggal** (persentase channel berbeda dan Spearman rho, per rasio -- berbeda dari skrip 08 lama karena bobot bukan konstan lagi sehingga korelasi ikut berubah per rasio, bukan satu nilai untuk semua rasio):
     | Rasio | vs L1 | vs GM-Ekspansi | vs Entropi |
     |---|---|---|---|
     | 10% | 12,02% (854), rho=0,528 | 15,15% (1076), rho=0,403 | 7,57% (538), rho=0,747 |
     | 20% | 24,32% (1728), rho=0,437 | 29,62% (2104), rho=0,297 | 9,23% (656), rho=0,861 |
     | 30% | 26,94% (1914), rho=0,715 | 35,05% (2490), rho=0,422 | 21,42% (1522), rho=0,546 |
     | 40% | 22,52% (1600), rho=0,825 | 39,56% (2810), rho=0,414 | 34,52% (2452), rho=0,351 |
     | 50% | 26,80% (1904), rho=0,691 | 29,42% (2090), rho=0,632 | 44,96% (3194), rho=0,223 |
     | 60% | 43,86% (3116), rho=0,288 | 33,36% (2370), rho=0,724 | 23,09% (1640), rho=0,576 |
     | 70% | 38,85% (2760), rho=0,194 | 41,69% (2962), rho=0,372 | 7,49% (532), rho=0,893 |

10. **SELESAI (2026-09-14, dicatat 2026-09-12).** Validasi multi-seed untuk KETIGA kriteria tunggal (L1, GM-Ekspansi, Entropi) di SELURUH 7 rasio, seed 123 dan 2024 (42 run: 21 pasang kriteria x rasio x 2 seed baru; seed 42 dipakai ulang dari checkpoint yang sudah ada). Cakupan diperluas dari rencana awal ("kriteria terbaik per rasio" saja) karena ditemukan seri antar kriteria pada seed 42 di 3 rasio (20%: GM=Ent; 40%: L1=GM; 70%: GM=Ent), sehingga "terbaik per rasio" tidak selalu tunggal. Skrip: `48_multiseed_single_criteria.py` (31,25 jam komputasi). Hasil: `outputs/multiseed_single_criteria_results.json`.

    **H1 diuji ulang dengan std GABUNGAN kedua sisi** (`sqrt(std_multi^2 + std_terbaik^2)`, kriteria terbaik per rasio ditentukan dari rata-rata 3 seed, bukan seed 42 tunggal seperti pengujian awal) -- ini **mengoreksi** pengujian H1 sepihak sebelumnya:
    - **Di dalam rentang wajar:** 10%, 20%, 50% (revisi dari verdict awal yang sempat menyatakan 50% "di luar rentang, multi lebih rendah" -- itu memakai std kriteria tunggal yang belum ada saat itu).
    - **Di luar rentang, multi LEBIH TINGGI:** 40% (rasio 2,47x), 70% (2,86x), dan 30% (batas tipis -- L1 dan GM seri pada rata-rata 3 seed di rasio ini, verdict berbeda tergantung kriteria seri mana yang dipakai sebagai pembanding).
    - **Di luar rentang, multi LEBIH RENDAH:** 60% (rasio 3,01x) -- **satu-satunya titik lemah H1 yang solid**, tidak berubah oleh koreksi pengujian.

    **Anomali non-monoton per kriteria tunggal** (dicurigai dari `kurva_kompresi_akurasi_per_rasio.png`, sudah diaudit tidak ada bug pemangkasan di sesi sebelumnya -- checkpoint hash unik, jumlah channel per block turun monoton, kelima tensor konsisten di setiap block untuk ketiga rasio yang diuji):
    - L1 bentuk-V di 10-20-30% (95,43 -> 93,90 -> 95,81): bertahan setelah rata-rata 3 seed, amplitudo mengecil dari seed 42 tunggal, dip di 20% ~1,4-1,6x gabungan std -- nyata meski tidak ekstrem.
    - L1 "flat" di 30-40-50% pada seed 42 (semuanya 95,43%): terbukti kebetulan seed tunggal -- rata-rata 3 seed menurun mulus (95,81 -> 94,86 -> 94,29), sesuai ekspektasi monoton wajar.
    - Entropi lonjakan di 60% (89,71 -> 92,57): **terkonfirmasi kuat**, rata-rata 3 seed identik dengan seed 42, ~2,75x gabungan std -- reproducible, bukan keberuntungan seed tunggal.
    - Simpangan baku Multi-Kriteria tertinggi di rasio 50% (1,43 pp, Bagian 8 butir 9) **tidak diwariskan** dari L1 atau GM manapun -- std tertinggi masing-masing kriteria tunggal ada di rasio BERBEDA (L1 tertinggi di 30%=0,97pp; GM tertinggi di 20%=1,40pp; Entropi tertinggi di 70%=1,23pp) -- kemungkinan muncul dari kombinasi bobot w_L1≈w_GM≈0,47 (hampir seimbang) khusus di rasio 50% itu, bukan warisan ketidakstabilan satu kriteria.

11. **[BUG - PERBAIKI SEBELUM SIDANG HASIL]. SELESAI (2026-09-24).** Aplikasi Android dan server Flask demo sempat masih mengirim/memuat model dari formula LAMA (valweights / checkpoint default lama), bukan formula aktif (GM-Ekspansi + bobot per rasio). Ditemukan saat reorganisasi `outputs/` pada 2026-09-12, diperbaiki 2026-09-24 setelah audit kebersihan skrip (Bagian 8 butir 9 sub-item dan butir 12 diselesaikan lebih dulu, lalu `test_system.py` 75/75 PASS dan `py_compile` bersih di seluruh `src/`+`scripts/` sebagai gerbang sebelum bug ini disentuh, sesuai urutan kerja yang diminta).

    **Perbaikan:**
    - `scripts/06_deploy_flask.py`: `MODEL_CHOICES` kini memuat `multicriteria_per_rasio_{10-70}pct_30ep.pth` (8 model: baseline + ketujuh rasio -- sebelumnya cuma 6 disebut di docstring padahal 8 di-load, sekalian diperbaiki). Panel Efisiensi Model membaca `outputs/multicriteria_per_rasio.json` untuk rasio 10-70% (fungsi baru `load_per_rasio_table()`), baseline tetap dari `tabel_hasil_lengkap.json` (formula-independent, tidak perlu diganti). Label "FLOPs" di panel diganti "MACs (thop)" mengikuti temuan butir 12. Diverifikasi lewat permintaan HTTP nyata ke `/predict` (baseline, rasio 20%, rasio 70%) -- bukan cuma baca kode.
    - `scripts/55_export_verify_pte_per_rasio.py` (skrip BARU, pola identik `scripts/20_test_executorch_runtime.py` yang TIDAK diubah): ekspor + verifikasi `.pte` untuk `multicriteria_per_rasio_{20,60}pct_30ep.pth`. Hasil: 20/20 prediksi PyTorch vs ExecuTorch identik, selisih logit maksimum ~1e-05 (kedua rasio) -- `outputs/executorch_runtime_test_per_rasio.json`.
    - `scripts/21_export_android_assets.py`: `PTE_FILENAMES`/`REFERENCE_CHECKPOINT` diarahkan ke checkpoint formula aktif dan `.pte` hasil skrip 55. `android_assets/` diregenerasi; dua `.pte` formula lama yang jadi basi dihapus (gitignored, tidak pernah ter-track git).
    - `scripts/24_uji_banding_android.py`, `25_siapkan_demo_proporsional.py`, `26_banding_demo_sidang.py`, `27_diagnosa_resize.py` (rantai persiapan demo sidang Android, ditemukan ikut memakai checkpoint formula lama saat audit ini, di luar dua file yang awalnya dilaporkan): `CHECKPOINT_NAME` diarahkan ke `multicriteria_per_rasio_20pct_30ep.pth`. `demo_sidang/` diregenerasi ulang (akurasi acuan berubah dari 96,00% jadi 97,14% karena checkpoint beda, komposisi 8 citra ikut berubah -- satu berkas basi dari seleksi lama dihapus); integritas salinan diverifikasi ulang lewat skrip 26 (8/8 status prediksi cocok).
    - `scripts/18_prepare_demo_images.py` (`DEFAULT_CHECKPOINT`, menghasilkan `outputs/citra_demo.json`): **ditemukan terlewat** dari sapuan pertama (2026-09-24, saat audit ulang lebih teliti) karena outputnya murni referensi manual (tidak dibaca skrip lain, jadi tidak muncul di jejak pemanggilan seperti Flask/Android). Diperbaiki dan diregenerasi -- akurasi acuan berubah dari 96,00% jadi 97,14% (checkpoint formula aktif).
    - **Project Android Studio (`D:\Tesis_Android\DeteksiDaunPoaceae`, di luar repositori Python ini) SUDAH diperbaiki (2026-09-24)**, bukan lagi di luar jangkauan: `Klasifikasi.kt` (alur produksi) dan `PreprocessTest.kt` (harness verifikasi) diarahkan ke `multicriteria_per_rasio_20pct_30ep.pte`; `app/src/main/assets/` diisi ulang dengan aset formula aktif (kedua `.pte`, `reference_*`, `labels.txt`, dst.), `.pte` lama dihapus. `gradlew compileDebugKotlin` -- BUILD SUCCESSFUL. `HASIL_VERIFIKASI.md` di project itu TIDAK diubah (tetap sah sebagai catatan historis run 2026-08-04 dengan checkpoint lama), tapi PERLU dijalankan ulang manual (lewat emulator/device) untuk angka verifikasi yang mencerminkan checkpoint aktif -- prediksi end-to-end di device sungguhan TIDAK bisa diverifikasi dari sesi ini.

    Rincian lengkap: `outputs/README_OUTPUTS.md` bagian "Perhatian -- SUDAH DISELARASKAN".

12. **[TEMUAN -- WAJIB DIPERHATIKAN SAAT MENULIS PAPER/TESIS, bukan bug kode].** Seluruh angka "flops" di proyek ini, sejak awal, sebenarnya adalah MACs (multiply-accumulate), bukan FLOPs. Ditemukan saat menyusun `scripts/51_hitung_flops.py` (2026-09-24 -- lihat juga item ini di riwayat kerja sebelumnya).

    **Akar masalah:** `measure_flops()` di `src/model.py` memanggil `thop.profile()` dan menyimpan nilai kembaliannya langsung sebagai `flops` (`src/model.py:99-101,315-316`). Nilai kembalian `thop.profile()` adalah MACs, BUKAN FLOPs -- diverifikasi lewat pembacaan source code thop (`calculate_conv2d_flops()` tidak mengalikan 2) dan lewat perhitungan MAC manual via forward hook (`Cout*Hout*Wout*(Cin/groups)*Kh*Kw` per Conv2d) yang cocok dengan angka thop hingga selisih 0,26% (angka thop sendiri memasukkan BatchNorm2d ke total, manual hook murni conv+linear). FLOPs sebenarnya = 2 x MACs (satu perkalian + satu penjumlahan per MAC), konvensi umum di literatur (termasuk Sandler et al. 2018 yang dirujuk di Bagian 10).

    **`measure_flops()` di `src/model.py` SENGAJA TIDAK diubah** -- konsisten dengan konvensi proyek ini untuk tidak mengubah kode inti yang sudah dipakai banyak skrip lama (`03,04,05,06,07,11,12,17` dan lain-lain, semuanya memanggil `evaluate_model()` yang memanggil `measure_flops()`). Akibatnya, **setiap angka "flops" pada SETIAP file berikut adalah MACs, bukan FLOPs**: `outputs/tabel_hasil_lengkap.json`, `outputs/ablation_val_results.json`, `outputs/multicriteria_per_rasio.json`, `outputs/ablation_gm_expansion.json`, `outputs/ablation_l1_expansion.json` -- berlaku surut untuk seluruh riwayat eksperimen proyek ini, bukan cuma file yang disebut di sini.

    **DIBERSIHKAN (2026-09-24)** lewat `scripts/56_tambah_flops_sebenarnya.py`: kelima file JSON di atas kini masing-masing punya field baru `flops_true_2x_macs` (= 2 x field `flops` asli) di setiap entri yang sebelumnya punya `flops`, ditambahkan secara ADITIF -- field `flops` asli TIDAK diubah/dihapus (supaya kode yang sudah membaca field itu, mis. `load_efficiency_metrics()` di `06_deploy_flask.py`, tidak terdampak), plus catatan `_catatan_macs_vs_flops` di level atas tiap file. Jadi sekarang kalau mau kutip FLOPs sebenarnya, tinggal ambil `flops_true_2x_macs` dari file manapun di atas -- tidak perlu hitung manual x2 lagi atau cuma mengandalkan `tabel_flops_per_rasio.csv`.

    **Angka FLOPs sebenarnya (2 x MACs) sudah dihitung untuk baseline dan ketujuh rasio formula aktif**, lewat `scripts/51_hitung_flops.py`, hasil di `outputs/tabel_flops_per_rasio.csv`:

    | Rasio | MACs (M) | FLOPs sebenarnya (M) | Hemat MACs/FLOPs (%, sama) | Hemat params (%) |
    |---|---|---|---|---|
    | 0% (baseline) | 326,22 | 652,44 | 0,00 | 0,00 |
    | 10% | 298,77 | 597,53 | 8,42 | 8,05 |
    | 20% | 270,84 | 541,67 | 16,98 | 16,13 |
    | 30% | 242,94 | 485,88 | 25,53 | 24,21 |
    | 40% | 215,01 | 430,02 | 34,09 | 32,29 |
    | 50% | 186,46 | 372,92 | 42,84 | 40,42 |
    | 60% | 159,01 | 318,02 | 51,26 | 48,46 |
    | 70% | 131,08 | 262,16 | 59,82 | 56,54 |

    Karena FLOPs = 2 x MACs adalah faktor pengali KONSTAN, seluruh persentase penghematan (kolom "hemat") TIDAK berubah oleh koreksi ini -- hanya angka ABSOLUT (MFLOPs/MMACs) yang berbeda 2x lipat. **Penting untuk penulisan paper/tesis:** kutip angka absolut beban komputasi dari `outputs/tabel_flops_per_rasio.csv` (kolom `flops_M` untuk FLOPs sebenarnya, atau `macs_M` jika memilih melaporkan sebagai MACs -- keduanya sekarang berlabel benar), JANGAN dari kolom `flops` pada file JSON lama di atas tanpa mengoreksinya lebih dulu (kalikan 2 jika ingin FLOPs, atau ganti label jadi MACs jika ingin memakai angka aslinya apa adanya). Data disajikan apa adanya di sini tanpa rekomendasi mana yang dipilih -- itu keputusan penulisan, bukan keputusan teknis.

---

## 9. Keterbatasan Metodologis yang Diketahui

Skrip 03, 12, dan skrip ablation lain memanggil `set_seed()` sekali di awal lalu menjalankan loop rasio secara berurutan, sehingga keadaan RNG saat fine-tuning setiap rasio bergantung pada rasio-rasio sebelumnya. Akibatnya angka pada tabel hasil utama merupakan tarikan tunggal yang tidak dapat direplikasi silang-skrip. Skrip 15 mereset seed di setiap run sehingga reproducible. Ketidakpastian akibat hal ini dikuantifikasi melalui validasi multi-seed dengan simpangan baku 0,81 sampai 1,40 poin persentase.

---

## 10. Referensi Terverifikasi

Hanya gunakan referensi berikut. Untuk referensi baru, DOI wajib diverifikasi terlebih dahulu.

**Fondasi arsitektur dan pruning:**
- Sandler et al. (2018), MobileNetV2, DOI 10.1109/CVPR.2018.00474
- Liu et al. (2017), Network Slimming, DOI 10.1109/ICCV.2017.298
- Fang et al. (2023), DepGraph, DOI 10.1109/CVPR52729.2023.01544

**Literatur pendukung (2022 sampai 2026):**
- Chen et al. (2022), DOI 10.3389/fpls.2022.1023515
- Hu et al. (2022), DOI 10.1080/09540091.2022.2111405
- Qi, Wang, dan Tang (2022), DOI 10.1007/s11063-022-10863-0
- Cheng et al. (2023), DOI 10.1007/s40747-023-01022-6
- Liu et al. (2023), DOI 10.1016/j.neucom.2023.126297
- Lu et al. (2024), DOI 10.1609/aaai.v38i4.28184
- He dan Xiao (2024), DOI 10.1109/TPAMI.2023.3334614
- Cheng, Zhang, dan Shi (2024), DOI 10.1109/TPAMI.2024.3447085
- Mukherjee et al. (2025), DOI 10.1016/j.engappai.2024.109639

**Metodologis:**
- Peffers et al. (2007), Design Science Research, DOI 10.2753/MIS0742-1222240302
- OECD Frascati Manual (2015), DOI 10.1787/9789264239012-en

**Catatan:** Jurnal terbitan MDPI tidak digunakan dalam penelitian ini sesuai ketentuan kampus.

---

## 11. Konvensi Repositori Git

Repositori ini memakai dua remote Git dengan peran berbeda.

- **`publish`** (`https://github.com/umayahmaya/pruning-multicriteria-poaceae.git`) adalah tujuan push untuk seterusnya. Ini remote yang aktif dipakai.
- **`origin`** (`https://github.com/umayahmaya/pruning-multicriteria.git`) tidak dipakai lagi. Riwayat commit lokalnya memuat sekitar 2,9 GB dataset mentah dan checkpoint model (`.pth`) yang sempat ter-commit lalu dihapus di commit berikutnya, sehingga tetap terbawa di riwayat dan melampaui batas 2 GB per push GitHub. Riwayatnya tidak ditulis ulang, remote ini hanya ditinggalkan.

Riwayat commit lengkap (termasuk seluruh proses perbaikan metodologi) tetap tersimpan apa adanya di branch `main` lokal sebagai catatan pribadi, dan tidak direplikasi ke remote mana pun. Branch `publish` hanya berisi satu commit snapshot kode terbaru, tanpa riwayat, tanpa dataset, dan tanpa checkpoint.

Alur kerja untuk memublikasikan perubahan:

```bash
# 1. Kerja seperti biasa di branch main, commit seperti biasa
git checkout main
# ... edit, commit ...

# 2. Saat siap memublikasikan, salin isi pohon kerja terbaru ke branch publish
git checkout publish
git checkout main -- .
git commit -m "snapshot: ..."

# 3. Push snapshot terbaru sebagai main di remote publish
git push publish publish:main

# 4. Kembali ke main untuk melanjutkan kerja
git checkout main
```
