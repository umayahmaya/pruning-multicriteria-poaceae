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
Fungsi skoring kepentingan channel berbasis multi-kriteria untuk pruning terstruktur pada MobileNetV2, diadaptasi dari Weighted Sum Model (Fishburn, 1967).

---

## 2. Metodologi Inti

### Fungsi skoring

```
I(c) = w1 * S_L1(c) + w2 * S_BN(c) + w3 * S_H(c)
```

Ketiga komponen dinormalisasi min-max per layer sebelum digabungkan, sehingga `I(c)` berada pada rentang [0, 1] sebagai convex combination.

| Simbol | Kriteria | Sumber nilai | Sifat |
|---|---|---|---|
| S_L1 | L1-norm bobot filter | Bobot depthwise conv 3x3 | Data-free |
| S_BN | Skala gamma batch norm | Layer BN setelah depthwise | Data-free |
| S_H | Entropi Shannon feature map | Output feature map, 256 bin | Butuh data kalibrasi |

### Bobot hasil ablation study (rasio 30 persen)

```
w1 = 0.3401   (L1-norm)
w2 = 0.3259   (BN gamma)
w3 = 0.3340   (Entropi)
```

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

1. Verifikasi apakah bobot ablation diturunkan dari akurasi validasi atau akurasi uji, perbaiki jika perlu.
2. Sesuaikan jumlah gambar kalibrasi entropi untuk konfigurasi sembilan kelas.
3. Hitung korelasi peringkat (Spearman atau Kendall tau) antara I(c) dan tiap kriteria tunggal, serta persentase channel yang berbeda pilihannya pada rasio 30 persen.
4. Jalankan eksperimen kontrol dengan bobot rata sepertiga sebagai pembanding.
5. Verifikasi kesetaraan total epoch antara baseline dan model hasil pruning.
6. Pertimbangkan pengujian tiga seed pada konfigurasi optimal untuk melaporkan rata-rata dan simpangan baku.
7. ~~(Prioritas sedang, ditemukan 2026-08-03) Alur utama 01→06 di README.md tidak mandiri tanpa menjalankan skrip 07 dulu, karena 04_pruning_multicriteria.py --load_weights mensyaratkan outputs/ablation_val_results.json yang hanya dihasilkan skrip 07.~~ **SELESAI (2026-08-03).** README.md Bagian "Urutan Menjalankan" diperbarui: skrip 07 dimasukkan ke alur utama di antara 03 dan 04 (urutan menjadi 01, 02, 03, 07, 04, 05, 06), dengan penjelasan bahwa 07 wajib dijalankan karena bobot harus diturunkan dari akurasi validasi sebelum pruning. Skrip 08, 10 sampai 18 tetap didokumentasikan sebagai koreksi metodologis lanjutan (11, 12) dan analisis lanjutan (sisanya), di luar alur utama.

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
