# Laporan Proyek Machine Learning - Firda Humaira
## Project Overview
Meningkatnya jumlah penonton bioskop sejalan dengan bertambahnya jumlah film yang diproduksi setiap tahunnya. Beragam film dengan berbagai alur cerita, genre, dan tema, baik dari industri perfilman luar negeri maupun dalam negeri, memenuhi pasar. Kondisi ini membuat penonton sering kali kebingungan dalam memilih film yang ingin ditonton, sehingga menghabiskan waktu cukup lama untuk mencari film yang sesuai. Beberapa orang mengandalkan fitur pencarian film dari berbagai situs sebagai referensi sebelum memutuskan film yang akan ditonton. Karena setiap individu memiliki preferensi yang berbeda, mereka cenderung tertarik pada film yang mirip dengan yang mereka sukai. Untuk membantu dalam hal ini, sistem rekomendasi menjadi salah satu solusi yang efektif.

Sistem rekomendasi film dapat dibuat dengan metode content-based filtering, yang berfokus pada kemiripan antar konten film. Dalam pendekatan ini, informasi seperti genre, kata kunci (keywords), dan sinopsis (overview) digunakan untuk mewakili setiap film dalam bentuk data tekstual. Data ini kemudian diolah menggunakan teknik TF-IDF (Term Frequency–Inverse Document Frequency) untuk mengekstraksi bobot pentingnya kata-kata dalam tiap film, dan selanjutnya tingkat kemiripan antar film dihitung menggunakan cosine similarity.

## 1. Business Understanding
### 1.1 Problem Statements
1. Bagaimana pengguna dapat menemukan film yang mirip dengan film favorit mereka berdasarkan isi/konten film?

2. Apakah informasi tambahan seperti perusahaan produksi dapat digunakan untuk lebih memahami kesamaan antar film?

3. Apakah informasi berupa keywords dan overview, tanpa melibatkan genre, dapat menghasilkan perhitungan cosine similarity yang lebih akurat dalam merekomendasikan film?
   
### 1.2 Goals
1. Membantu pengguna menemukan film serupa berdasarkan kesamaan konten film, bukan hanya genre atau rating.

2. Menilai apakah informasi tambahan (seperti production company) berkontribusi terhadap penentuan kesamaan film.

3. Menguji apakah hanya menggunakan keywords dan overview (tanpa genre) bisa menghasilkan rekomendasi yang lebih relevan.
   
### 1.3 Solution Approach
1. Gunakan content-based filtering dengan representasi teks (genre_list, keyword, dan overview) menggunakan TF-IDF dan cosine similarity untuk menemukan film yang mirip.

2. Tambahkan fitur seperti production company ke dalam model dan uji kontribusinya terhadap peningkatan kualitas rekomendasi.

3. Bandingkan hasil cosine similarity dari model yang hanya menggunakan keywords dan overview dengan model yang juga menyertakan genre untuk mengevaluasi akurasi rekomendasi.
   
## 2. Data Understanding
### 2.1 Exploratory Data Analysis
#### 2.1.1 Mengetahui jumlah missing values, data duplicate dan outlier
#### 2.1.2 Distribusi Kolom Numerik
#### 2.1.3 Korelasi Antar Kolom Numerik
#### 2.1.4 Rata-rata rating per genre

## 3. Data Preparation
### 3.1 Data Cleaning
#### 3.1.1 Format Tidak Konsisten / JSON dalam String
#### 3.1.2 Menghapus kolom yang tidak relevan untuk Sistem Rekomendasi (Content-based Filtering)
#### 3.1.3 Menangani Missing Values

## 4. Modeling and Result
### 4.1 Model Development dengan Content Based Features
### 4.2 Penerapan

## 5. Evaluation

## Problem Answer
### 1. Bagaimana pengguna dapat menemukan film yang mirip dengan film favorit mereka berdasarkan isi/konten film?

### 2. Apakah informasi tambahan seperti perusahaan produksi dapat digunakan untuk lebih memahami kesamaan antar film?

### 3. Apakah informasi berupa keywords dan overview, tanpa melibatkan genre, dapat menghasilkan perhitungan cosine similarity yang lebih akurat dalam merekomendasikan film?

# Referensi 
1. Arfisko, H. H., & Wibowo, A. T. (2022). Sistem Rekomendasi Film Menggunakan Metode Hybrid Collaborative Filtering Dan Content-Based Filtering. eProceedings of Engineering, 9(3).

2. Fajriansyah, M., Adikara, P. P., & Widodo, A. W. (2021). Sistem Rekomendasi Film Menggunakan Content Based Filtering. Jurnal Pengembangan Teknologi Informasi Dan Ilmu Komputer, 5(6), 2188–2199. Diambil dari https://j-ptiik.ub.ac.id/index.php/j-ptiik/article/view/9163

3. Saputra, J. M. A., Huizen, L. M., & Arianto, D. B. (2024). Sistem Rekomendasi Film pada Platform Streaming Menggunakan Metode Content-Based Filtering. Jurnal Transformatika, 22(1), 10-21.
