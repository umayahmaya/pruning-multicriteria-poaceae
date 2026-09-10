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

**Catatan versi:** kriteria kedua semula S_BN (skala gamma batch norm setelah depthwise), diganti S_GM-Ekspansi pada 26 Agustus 2026 (`scripts/43_ablation_gm_expansion.py`), setelah eksplorasi lanjutan pertengahan Agustus 2026 (skrip `22` sampai `43`, belum di-commit ke git) membandingkan beberapa varian kriteria berbasis redundansi channel -- termasuk GM dari bobot depthwise (`src/gm_scores.py`, `42_ablation_gm.py`), GM/kosinus dari conv ekspansi terhadap median geometrik (`36_skor_redundansi.py`), korelasi antar-lapisan, dan skema alokasi anggaran kalibrasi entropi. S_GM-Ekspansi (jarak berpasangan dari bobot conv ekspansi) yang akhirnya diadopsi.

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

1. ~~Verifikasi apakah bobot ablation diturunkan dari akurasi validasi atau akurasi uji, perbaiki jika perlu.~~ **SELESAI.** Bobot lama (`outputs/ablation_results.json`) ternyata diturunkan dari akurasi test. Diperbaiki lewat `07_recompute_weights_val.py`, yang menghasilkan `optimal_weights_val` dari akurasi val di `outputs/ablation_val_results.json`. Dipakai oleh `12_multicriteria_valweights.py` dan kini juga oleh `04_pruning_multicriteria.py --load_weights`.
2. ~~Sesuaikan jumlah gambar kalibrasi entropi untuk konfigurasi sembilan kelas.~~ **SELESAI.** `compute_entropy_scores()` di `src/pruning.py` diperbaiki: 20 citra/kelas (180 total untuk 9 kelas), diambil dari train split (bukan val), deterministik, transform evaluasi tanpa augmentasi. Diverifikasi lewat `10_smoke_test_entropy.py`, dipakai ulang untuk ablation entropi di `11_ablation_entropy_traincal.py`.
3. ~~Hitung korelasi peringkat (Spearman atau Kendall tau) antara I(c) dan tiap kriteria tunggal, serta persentase channel yang berbeda pilihannya pada rasio 30 persen.~~ **SELESAI.** Dihitung di `08_compare_channel_selection.py`, diperluas ke seluruh 7 rasio (bukan hanya 30%). Hasil di `outputs/channel_selection_comparison.json`.
4. **TIDAK DIKERJAKAN.** Eksperimen kontrol dengan bobot rata sepertiga sebagai pembanding tidak dijalankan terpisah. Bobot hasil ablation (w1=0,3361, w2=0,3320, w3=0,3320) sudah sangat dekat dengan sepertiga. Uji sensitivitas bobot di `08_compare_channel_selection.py` (perbandingan bobot lama vs baru, selisih hingga ~0,006 per bobot) menunjukkan perubahan bobot sebesar itu hanya mengubah 0,14 sampai 0,39 persen pilihan channel dengan Spearman 0,9999 di seluruh 7 rasio (lihat `outputs/channel_selection_comparison.json`), sehingga eksperimen kontrol bobot rata sepertiga terpisah diperkirakan tidak akan memberi informasi baru.
5. ~~Verifikasi kesetaraan total epoch antara baseline dan model hasil pruning.~~ **SELESAI.** `BASELINE_EPOCHS` dan `FINETUNE_EPOCHS` di `src/config.py` sama-sama 30.
6. ~~Pertimbangkan pengujian tiga seed pada konfigurasi optimal untuk melaporkan rata-rata dan simpangan baku.~~ **SELESAI.** Dijalankan di `15_multiseed_validation.py` (baseline, rasio 20%, rasio 60%; seed 42/123/2024) dan `16_multiseed_remaining_ratios.py` (rasio 10/30/40/50/70%; seed 123/2024, seed 42 dipakai ulang dari `12_multicriteria_valweights.py`). Hasil di `outputs/multiseed_results.json`, `outputs/multiseed_remaining_results.json`, dan `outputs/multiseed_seed{42,123,2024}.json`.
7. ~~(Prioritas sedang, ditemukan 2026-08-03) Alur utama 01→06 di README.md tidak mandiri tanpa menjalankan skrip 07 dulu, karena 04_pruning_multicriteria.py --load_weights mensyaratkan outputs/ablation_val_results.json yang hanya dihasilkan skrip 07.~~ **SELESAI (2026-08-03).** README.md Bagian "Urutan Menjalankan" diperbarui: skrip 07 dimasukkan ke alur utama di antara 03 dan 04 (urutan menjadi 01, 02, 03, 07, 04, 05, 06), dengan penjelasan bahwa 07 wajib dijalankan karena bobot harus diturunkan dari akurasi validasi sebelum pruning. Skrip 08, 10 sampai 18 tetap didokumentasikan sebagai koreksi metodologis lanjutan (11, 12) dan analisis lanjutan (sisanya), di luar alur utama.
8. ~~(ditemukan 2026-08-04) executorch.runtime.Runtime gagal dimuat di venv/ untuk uji ekspor ExecuTorch, sehingga kesamaan prediksi dan selisih logit belum terverifikasi.~~ **SELESAI (2026-08-04).** `19_test_executorch_export.py` (di `venv/`) memverifikasi poin 1 (ekspor berhasil) dan poin 2 (ukuran berkas) untuk baseline/rasio 20%/rasio 60% -- 8,734 MB, 7,338 MB, 4,537 MB. `20_test_executorch_runtime.py` (di `venv_mobile/`) melengkapi poin 3 dan 4: untuk ketiga model, 20/20 prediksi PyTorch vs ExecuTorch identik, selisih logit maksimum di orde 1e-05 (noise floating-point). Hasil di `outputs/executorch_runtime_test.json`.

   Akar masalah: executorch 1.3.1 (versi terbaru di PyPI) memiliki modul native `_portable_lib` yang gagal dimuat di Windows untuk SEMUA kombinasi torch yang dicoba (torch 2.13.0+cpu yang terpasang di `venv/`, torch 2.13.0 default PyPI, torch nightly 2.14.0.dev) -- kemungkinan cacat pada wheel Windows rilis itu sendiri, bukan soal versi torch. executorch 1.0.1 dengan torch 2.9.1+cpu (versi yang secara eksplisit ia syaratkan) terbukti berfungsi.

   `venv_mobile/` (gitignored, tidak di-commit) adalah lingkungan Python 3.11 terpisah khusus untuk ini. `venv/` (lingkungan penelitian utama) TIDAK diubah lagi untuk mengejar kecocokan versi torch dengan executorch (tidak dipasangi torch nightly dsb.) -- satu-satunya perubahan di `venv/` adalah instalasi `executorch==1.3.1` itu sendiri (untuk `19_test_executorch_export.py`), yang sebagai efek samping menurunkan `scikit-learn` dari 1.9.0 ke 1.7.1 (dependensi transitif `torchao`). Hanya `train_test_split` dan `sklearn.metrics` yang dipakai di proyek ini, API yang stabil di kedua versi. `test_system.py` dijalankan ulang setelah instalasi dan tetap lulus 75 dari 75. `requirements.txt` diperbarui lewat `pip freeze` untuk mencerminkan kondisi `venv/` yang sekarang.

   Cara reproduksi `venv_mobile/`:
   ```
   py -3.11 -m venv venv_mobile
   venv_mobile/Scripts/python.exe -m pip install "executorch==1.0.1"
   venv_mobile/Scripts/python.exe -m pip install "torchvision==0.24.1"
   ```
   (torch 2.9.1+cpu terpasang otomatis sebagai dependensi executorch==1.0.1). `flatc` (kompilator FlatBuffers, dibutuhkan saat serialisasi .pte) di-resolve otomatis lewat env var `FLATC_EXECUTABLE` di dalam `20_test_executorch_runtime.py` sendiri karena `venv_mobile/Scripts` tidak selalu ada di PATH -- tidak perlu setup manual tambahan.

9. **SELESAI (2026-08-26), didokumentasikan di sini 2026-09-10.** Kriteria kedua I(c) diganti dari S_BN ke S_GM-Ekspansi dan skema bobot diganti dari konstan ke per-rasio -- lihat Bagian 2 (formula aktif) dan Bagian 2a (formula lama, diarsipkan). Hasil: `outputs/multicriteria_per_rasio.json`. Rasio 20% mencapai 97,14% akurasi test (vs baseline 96,00%, selisih 1,14 pp) -- BELUM boleh diklaim sebagai peningkatan definitif karena di bawah ambang 2 pp pada Aturan 5 (Bagian 4) dan belum diuji multi-seed.

   **Masih terbuka akibat perubahan ini:**
   - Validasi multi-seed (`15`/`16_multiseed_*.py`) hanya menguji formula LAMA (Bagian 2a). Formula per-rasio yang baru (Bagian 2) belum divalidasi multi-seed sama sekali -- simpangan baku pada Bagian 9 TIDAK berlaku untuk formula baru ini.
   - Seluruh pekerjaan eksplorasi 10-26 Agustus 2026 (skrip `22` sampai `44`, seluruh file baru di `outputs/` pada rentang tanggal itu, `src/gm_scores.py`) belum pernah di-commit ke git -- masih berstatus untracked per `git status` per 10 September 2026. Commit ini perlu dibuat (di branch `main`, lalu disalin ke `publish` sesuai Bagian 11) sebelum riwayat kerja sebulan ini berisiko hilang.
   - `channel_selection_comparison.json` (Bagian 8 butir 3) dan uji sensitivitas bobot terkait hanya membandingkan bobot konstan lama vs lebih lama -- belum dihitung ulang untuk bobot per-rasio yang baru.
   - README.md (alur "01 sampai 06" plus skrip 07, Bagian 8 butir 7) belum menyebut skrip 43/44 sama sekali.

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
