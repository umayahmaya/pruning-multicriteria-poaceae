# Catatan Citra Demonstrasi Sidang

Delapan citra ini dipilih dengan komposisi **7 benar, 1 salah** (87,50%), mendekati akurasi uji model yang sebenarnya **168/175 = 96.00%** pada seluruh `dataset_split/test` -- rasio bilangan bulat terdekat yang masih bisa direpresentasikan pada sampel sekecil 8 citra sambil tetap menyertakan minimal satu contoh kesalahan (jujur terhadap performa model, bukan hanya menampilkan kasus yang mudah).

Checkpoint: `multicriteria_20pct_30ep_valweights.pth`

Pemilihan deterministik (seed 42): tiap citra benar diambil dari probabilitas prediksi BENAR tertinggi di kelasnya; citra salah diambil dari probabilitas prediksi SALAH tertinggi pada pasangan kelas yang paling sering tertukar.

---

## Tabel Citra Demo

| No | Berkas | Kelas Asli | Prediksi Model | Tiga Kelas Teratas | Status |
|---|---|---|---|---|---|
| 1 | demo_01_benar_Brown_Spot_Rice.jpg | Brown_Spot_Rice | Brown_Spot_Rice | Brown_Spot_Rice (100.00%), Leaf_Blast_Rice (0.00%), Healthy_Rice (0.00%) | BENAR |
| 2 | demo_02_benar_Healthy_Rice.jpg | Healthy_Rice | Healthy_Rice | Healthy_Rice (100.00%), Brown_Spot_Rice (0.00%), Leaf_Blast_Rice (0.00%) | BENAR |
| 3 | demo_03_benar_Leaf_Blast_Rice.jpg | Leaf_Blast_Rice | Leaf_Blast_Rice | Leaf_Blast_Rice (99.98%), Brown_Spot_Rice (0.02%), Rust_Sugarcane (0.00%) | BENAR |
| 4 | demo_04_benar_Common_Rust_Corn.png | Common_Rust_Corn | Common_Rust_Corn | Common_Rust_Corn (100.00%), Leaf_Blast_Rice (0.00%), Brown_Spot_Rice (0.00%) | BENAR |
| 5 | demo_05_benar_Gray_Leaf_Spot_Corn.JPG | Gray_Leaf_Spot_Corn | Gray_Leaf_Spot_Corn | Gray_Leaf_Spot_Corn (100.00%), Brown_Spot_Rice (0.00%), Common_Rust_Corn (0.00%) | BENAR |
| 6 | demo_06_benar_Healthy_Corn.jpg | Healthy_Corn | Healthy_Corn | Healthy_Corn (100.00%), Gray_Leaf_Spot_Corn (0.00%), Healthy_Sugarcane (0.00%) | BENAR |
| 7 | demo_07_benar_Healthy_Sugarcane.jpeg | Healthy_Sugarcane | Healthy_Sugarcane | Healthy_Sugarcane (100.00%), Mosaic_Sugarcane (0.00%), Rust_Sugarcane (0.00%) | BENAR |
| 8 | demo_08_salah_Mosaic_Sugarcane.jpeg | Mosaic_Sugarcane | Healthy_Sugarcane | Healthy_Sugarcane (90.30%), Mosaic_Sugarcane (7.60%), Rust_Sugarcane (2.04%) | SALAH |

---

## Catatan untuk Citra Salah

**demo_08_salah_Mosaic_Sugarcane.jpeg** (kelas asli **Mosaic_Sugarcane**, diprediksi sebagai **Healthy_Sugarcane**) termasuk dalam sekitar **4.00% kesalahan model** pada seluruh citra uji (7 dari 175 citra salah).

**Catatan kejujuran soal "paling sering tertukar":** pada evaluasi ini seluruh kesalahan kelas tebu terjadi tepat 1 kali masing-masing -- **tidak ada pasangan yang benar-benar dominan**. Ketiga pasangan yang seri: Mosaic_Sugarcane -> Healthy_Sugarcane; Rust_Sugarcane -> Common_Rust_Corn; Rust_Sugarcane -> Healthy_Sugarcane. Pasangan Mosaic_Sugarcane -> Healthy_Sugarcane dipilih lewat aturan pemecah seri deterministik (urutan abjad nama kelas), bukan karena frekuensinya lebih tinggi dari yang lain. Yang tetap valid dan konsisten dengan temuan sesi-sesi sebelumnya (mis. `HASIL_VERIFIKASI.md`): seluruh kesalahan kelas tebu pada evaluasi ini bermuara ke kelas tebu lain (bukan ke padi/jagung, kecuali satu kasus Rust_Sugarcane -> Common_Rust_Corn), sehingga tebu tetap area yang paling rawan tertukar dibanding padi atau jagung.
