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
Dataset yang saya gunakan diambil dari platform open source Kaggle dan dipublikasi oleh Ibtesan Ahmed, [Link download](https://www.kaggle.com/code/ibtesama/getting-started-with-a-movie-recommendation-system/input?select=tmdb_5000_movies.csv), saya memakai yang tmdb_5000_movies.csv, File ini berisi data tentang ±5.000 film dari database TMDB (The Movie Database). 

- Deskripsi Variabel Dataset

| **Variabel**         | **Deskripsi**                                                                                       |
| ---------------------- | --------------------------------------------------------------------------------------------------- |
| `budget`               | Anggaran produksi film (dalam USD)                                                                  |
| `genres`               | Daftar genre film (misalnya: Action, Comedy), dalam format JSON                                     |
| `homepage`             | URL website resmi film (jika ada)                                                                   |
| `id`                   | ID unik film dari TMDb                                                                              |
| `keywords`             | Kata kunci yang menggambarkan isi film, dalam format JSON                                           |
| `original_language`    | Bahasa asli film, menggunakan kode ISO 639-1. Contoh: `en` = Inggris, `fr` = Prancis, `ja` = Jepang |
| `original_title`       | Judul asli film (bisa berbeda dengan `title`)                                                       |
| `overview`             | Ringkasan atau sinopsis film                                                                        |
| `popularity`           | Skor popularitas film menurut TMDb (gabungan dari views, rating, dll)                               |
| `production_companies` | Daftar perusahaan produksi yang membuat film, dalam format JSON                                     |
| `release_date`         | Tanggal rilis film (format: YYYY-MM-DD)                                                             |
| `revenue`              | Pendapatan film secara global (dalam USD)                                                           |
| `runtime`              | Durasi film dalam menit                                                                             |
| `spoken_languages`     | Bahasa-bahasa yang digunakan dalam film, dalam format JSON                                          |
| `status`               | Status rilis film, misalnya: Released, Post Production, dll                                         |
| `tagline`              | Slogan atau tagline promosi film                                                                    |
| `title`                | Judul film (nama yang ditampilkan ke publik)                                                        |
| `vote_average`         | Rata-rata rating film dari pengguna (skala 1–10)                                                    |
| `vote_count`           | Jumlah total suara/rating yang diberikan pengguna                                                   |

- Info Dataset
<p align="center">
   <img src="Assets/info_dataset.png"width="500"/>
</p>

      Insight:
      1. Dataset ini terdiri dari 4803 baris dan 20 kolom
      2. Dataset ini terdiri dari 3 tipe data, yaitu: object, float64, dan int64
      3. Terdapat Missing Values di variabel **homepage, release_date, runtime, overview dan tagline

- Deskripsi Statistik Dataset
<p align="center">
   <img src="Assets/deskripsi.png"width="500"/>
</p>

Insight:
1. Banyak Nilai 0 yang Tidak Masuk Akal

| Kolom     | Masalah                                                         |
| --------- | --------------------------------------------------------------- |
| `budget`  | Min = 0 → banyak film tidak mencantumkan anggaran               |
| `revenue` | Min = 0 → banyak film tidak mencantumkan pendapatan             |
| `runtime` | Min = 0 → ada film yang tidak punya durasi (data error/missing) |

2. Data Sangat Tidak Merata (Skewed)
   - budget, revenue, dan vote_count menunjukkan perbedaan ekstrem antara nilai minimum dan maksimum.

3. Rata-rata Durasi & Rating Film

| Kolom          | Nilai Rata-rata                                          |
| -------------- | -------------------------------------------------------- |
| `runtime`      | \~107 menit (sekitar 1 jam 47 menit)                     |
| `vote_average` | \~6.1 dari 10                                            |
| `popularity`   | Skor rata-rata \~21, tapi sangat bervariasi (maks = 875) |

  - noted: Durasi film sebagian besar normal. Rating rata-rata cenderung di angka 6–7, artinya sebagian besar film "cukup baik".


### 2.1 Exploratory Data Analysis
#### 2.1.1 Mengetahui jumlah missing values, data duplicate dan outlier
- Missing Values
<p align="center">
   <img src="Assets/missing_values.png"width="500"/>
</p>

      insight:

      Terdapat missing values di dataset ini. Kemungkinan akan saya hapus karena tidak terpakai saat modeling

- Duplicate
<p align="center">
   <img src="Assets/duplicate.png"width="500"/>
</p>

      insight:

      Tidak terdapat duplikasi data di dataset ini.
  
- Outlier
<p align="center">
   <img src="Assets/cek_outlier.png"width="500"/>
   <img src="Assets/outlier.png"width="500"/>
</p>

Insight:
- Terdapat banyak outlier di sebagian besar fitur (kecuali vote_average dan sebagian runtime).

- Fitur-fitur seperti budget, revenue, popularity, dan vote_count menunjukkan bahwa industri film sangat tidak seimbang: hanya sebagian kecil film yang sangat sukses secara komersial atau populer.
  
#### Pembagian Fitur
<p align="center">
   <img src="Assets/pembagian.png"width="500"/>
</p>

- Yang tidak dimasukkan ke numeric atau categoric :
  - **id**
  - **title**
  - **original_title**
  - **tagline**
  - **homepage**
  - **overview**
  - **release_date**

| **Kolom**        | **Tipe Data**    | **Alasan Tidak Termasuk**                                                                                                                                    |
| ---------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `id`             | Integer          | Merupakan **ID unik** film, tidak punya makna statistik atau kategorikal. Tidak digunakan untuk analisis atau pemodelan.                                     |
| `title`          | String           | **Judul film**, bersifat unik per film. Tidak berguna sebagai fitur numerik/kategorikal.                                                                     |
| `original_title` | String           | Sama seperti `title`, bersifat unik dan bukan fitur yang representatif secara statistik.                                                                     |
| `homepage`       | String (URL)     | Merupakan **link URL resmi** film. Unik per film, tidak bermakna untuk pemodelan atau analisis.                                                              |
| `tagline`        | String           | Kalimat promosi (slogan), bersifat unik atau kosong. Jarang berguna langsung dalam analisis kategorikal atau numerik.                                        |
| `overview`       | String (teks)    | Deskripsi panjang (text bebas). Tidak termasuk `categoric` karena bukan kategori diskrit, dan tidak bisa diukur seperti `numeric`. Bisa digunakan untuk NLP. |
| `release_date`   | String (tanggal) | Format tanggal (`YYYY-MM-DD`), bukan numerik secara langsung, dan bukan kategorikal. Perlu diubah dulu (misalnya jadi tahun atau bulan) untuk digunakan.     |

#### 2.1.2 Distribusi Kolom Numerik
<p align="center">
   <img src="Assets/db.png"width="500"/>
   <img src="Assets/dr.png"width="500"/>
   <img src="Assets/druntime.png"width="500"/>
   <img src="Assets/dv.png"width="500"/>
   <img src="Assets/dp.png"width="500"/>
   <img src="Assets/dvc.png"width="500"/>
</p>

Insight:

- Budget & Revenue: Distribusinya sangat miring ke kanan; sebagian besar film punya nilai kecil, hanya sedikit yang sangat besar.

- Runtime: Distribusi mendekati normal, mayoritas film berdurasi sekitar 100 menit.

- Vote Average: Simetris, sebagian besar film punya rating sekitar 6–7.

- Popularity & Vote Count: Sama seperti budget dan revenue, miring ke kanan; hanya sebagian kecil film yang sangat populer dan banyak mendapat suara.

kesimpulan

Sebagian besar film dalam dataset memiliki karakteristik yang cukup umum—runtime sekitar 100 menit, rating rata-rata, serta nilai popularitas, budget, dan pendapatan yang relatif rendah. Hanya sebagian kecil film yang tergolong sangat sukses atau populer, terlihat dari adanya outlier pada distribusi budget, revenue, popularity, dan vote count. Pola ini menggambarkan bahwa industri film didominasi oleh beberapa blockbuster saja, sedangkan mayoritas film beroperasi pada skala yang lebih kecil.

#### 2.1.3 Korelasi Antar Kolom Numerik
<p align="center">
   <img src="Assets/korelasi_numerik.png"width="500"/>
</p>

Insight:

1. vote_count punya korelasi kuat dengan
    - revenue (0.78) → film yang banyak diberi suara (vote) cenderung menghasilkan pendapatan lebih tinggi.
    - popularity (0.78) → film yang populer cenderung lebih banyak diberi rating.
    - budget (0.59) → film dengan anggaran lebih besar umumnya juga mendapat lebih banyak vote.

2. budget dan revenue sangat berkorelasi (0.73)
    - Semakin besar anggaran produksi, cenderung semakin besar pendapatannya.

3. vote_average (rating rata-rata) tidak berkorelasi tinggi dengan apapun
    - Korelasi tertinggi hanya dengan runtime (0.38) dan vote_count (0.31) → artinya film yang panjang dan banyak di-vote sedikit lebih cenderung dapat rating bagus, tapi tidak signifikan.

4. runtime korelasinya lemah ke semua variabel:
    - Film berdurasi panjang tidak menjamin revenue, popularitas, atau rating tinggi.
      
#### 2.1.4 Rata-rata rating per genre
Karena genres berupa JSON string, perlu di-parse dulu:

<p align="center">
   <img src="Assets/parse.png"width="500"/>
</p>

lalu eksplor:

<p align="center">
   <img src="Assets/explore.png"width="500"/>
</p>

visualisainya:

<p align="center">
   <img src="Assets/visualisasi_rating.png"width="500"/>
</p>

## 3. Data Preparation
### 3.1 Data Cleaning
#### 3.1.1 Format Tidak Konsisten / JSON dalam String
Melakukan penanganan terhadap format yang tidak konsisten atau JSON dalam string itu penting karena beberapa alasan krusial dalam proses pengolahan data.
Alasan:
1. Parsing dan Ekstraksi Data Gagal:
     - Banyak kolom (misalnya genres, keywords, production_companies) disimpan dalam format string yang terlihat seperti JSON (misalnya: "[{'id': 28, 'name': 'Action'}]"). Jika tidak dikonversi ke struktur Python (seperti list atau dict), kita tidak bisa mengakses nilai-nilainya secara efektif.

2. Analisis Terhambat:
     - Jika struktur data tidak dibersihkan, kita tidak bisa menghitung jumlah genre, menghitung berapa film mengandung genre "Action", atau melakukan agregasi berdasarkan kategori tertentu.

3. Inconsistent Format = Error Saat Modeling:
     - Format yang tidak konsisten (misal kadang pakai kutip tunggal ', kadang kutip ganda ", atau bahkan format tidak lengkap) bisa menyebabkan error saat parsing, yang berdampak pada pipeline analisis atau training model machine learning.

<p align="center">
   <img src="Assets/json.png"width="500"/>
</p>

#### 3.1.2 Menghapus kolom yang tidak relevan untuk Sistem Rekomendasi (Content-based Filtering)
Content-Based Filtering adalah metode sistem rekomendasi yang memberikan saran berdasarkan kemiripan konten antar item (dalam hal ini, film).
Artinya, sistem ini melihat fitur-fitur yang menjelaskan isi atau karakteristik film, seperti:

- overview (sinopsis)

- genres (genre film)

- keywords (kata kunci)

jadi, saya akan menghapus variabel yang tidak relevan

<p align="center">
   <img src="Assets/hapus_kolom.png"width="500"/>
   <img src="Assets/info_baru.png"width="500"/>
</p>

Insight:

Masih terdapat missing values di kolom `overview`

#### 3.1.3 Menangani Missing Values

<p align="center">
   <img src="Assets/hapus_missing.png"width="500"/>
</p>

Insight:

Missing values sudah tidak ada

## 4. Modeling and Result
### 4.1 Model Development dengan Content Based Features
#### TF-IDF
Pada proyek saya ini perlu menggunakan TF-IDF (Term Frequency - Inverse Document Frequency) karena metode ini sangat efektif dalam mengolah data teks untuk sistem rekomendasi berbasis konten (content-based filtering)

TF-IDF digunakan karena bisa mengekstrak makna penting dari teks, mengabaikan kata tidak penting, dan mengubah teks menjadi format numerik yang siap untuk menghitung kemiripan antar item.

<p align="center">
   <img src="Assets/tf-idf.png"width="500"/>
</p>

Insight:

Tahap penggabungan fitur teks (`combined_feature`) sangat penting dalam proses TF-IDF karena memungkinkan berbagai sumber informasi teks seperti overview, keywords, dan genre_list disatukan ke dalam satu kolom. TF-IDF hanya dapat bekerja pada satu kolom teks, sehingga penyatuan ini diperlukan agar semua informasi penting bisa diproses bersama. Dengan menggabungkan fitur-fitur tersebut, representasi teks menjadi lebih kaya dan kontekstual. Misalnya, overview memberikan gambaran alur cerita, keywords mencerminkan tema atau topik spesifik, dan genre_list menambahkan klasifikasi film. Kombinasi ini membantu TF-IDF menghitung bobot kata secara lebih akurat. Selain itu, vektor teks yang lebih lengkap meningkatkan akurasi perhitungan cosine similarity, sehingga menghasilkan rekomendasi film yang lebih relevan. Penggabungan ini juga mencegah kehilangan informasi penting yang mungkin terjadi jika hanya satu fitur saja yang digunakan dalam analisis.

#### Cosine Similarity
<p align="center">
   <img src="Assets/cosine_similarity.png"width="500"/>
</p>

- Cosine Similarity antar `title`
<p align="center">
   <img src="Assets/df_cs.png"width="500"/>
   <img src="Assets/cs_antar_title.png"width="500"/>
</p>

Insight:

Cosine similarity matrix memungkinkan kita mengukur dan membandingkan kemiripan antar film secara numerik. Dari sampel data yang ditampilkan, tampak bahwa sebagian besar film memiliki kesamaan yang rendah satu sama lain, menandakan keberagaman tema dan genre. Namun, ada juga beberapa pasangan film dengan kemiripan relatif tinggi, yang menunjukkan bahwa model berhasil menangkap hubungan semantik atau genre tertentu antar film. Matriks ini sangat berguna untuk sistem rekomendasi berbasis konten (content-based filtering), di mana film yang mirip dengan film tertentu bisa direkomendasikan kepada pengguna.

### 4.2 Penerapan
<p align="center">
   <img src="Assets/membuat_fungsi.png"width="500"/>
</p>

#### 4.2.1 Penerapan 1
<p align="center">
   <img src="Assets/penerapan1.png"width="500"/>
</p>

Insight:
1. Kesamaan Genre

    - Hampir semua film yang direkomendasikan memiliki genre Science Fiction dan beberapa juga memiliki unsur Action dan Thriller, sama seperti Avatar.

    - Ini menunjukkan bahwa sistem berhasil menangkap kesamaan genre utama dari Avatar dengan film lain.

2. Kemiripan Kata Kunci & Tema Cerita:

    - Keyword seperti spacecraft, alien, space marine, android, extraterrestrial sangat mendekati dunia dan tema Avatar.

    - Banyak film juga mengambil latar luar angkasa atau eksplorasi luar dunia — cocok dengan dunia fiksi ilmiah imajinatif ala Avatar.

3. Variasi dan Relevansi Film:

    - Judul seperti Aliens, Moonraker, dan Mission to Mars memang berkutat di genre eksplorasi luar angkasa dan konflik antar spesies atau teknologi canggih.

    - Spaceballs muncul sebagai film dengan elemen sci-fi, namun bernuansa komedi parodi, yang bisa diperdebatkan relevansinya tergantung konteks pengguna.
      
#### 4.2.2 Penerapan 2
<p align="center">
   <img src="Assets/penerapan2.png"width="500"/>
</p>

Insight:
1. Genre yang Serupa

    - Film yang direkomendasikan mayoritas memiliki genre Fantasy, Family, Comedy, dan bahkan ada yang Animation seperti Shrek 2.

    - Ini menunjukkan bahwa sistem rekomendasi berhasil mengidentifikasi genre utama dari Tangled dan mencari film dengan genre sejenis.

2. Kemiripan Tema/Kata Kunci

    - Film seperti Ella Enchanted dan Enchanted memiliki keyword seperti "magic", "fairy tale", dan "princess" — sangat relevan dengan karakteristik Tangled.

    - Hal ini menunjukkan bahwa metode TF-IDF + Cosine Similarity cukup efektif menangkap konteks kata dan tema cerita.

3. Kualitas Hasil Rekomendasi

    - Judul seperti Into the Woods dan Enchanted memang punya kesamaan dunia fantasi dan musikal, cocok untuk penonton Tangled.

    - Namun ada juga judul seperti Out of Inferno (dengan genre Action) yang kurang relevan. Ini mungkin terjadi karena keterbatasan informasi pada fitur teks, atau karena elemen umum seperti "heroism" atau "rescue" membuatnya dianggap mirip.
      
#### 4.2.3 Penerapan 3
<p align="center">
   <img src="Assets/penerapan3.png"width="500"/>
</p>

Insight:

1. Nilai cosine score relatif rendah (maks ~0.12)
  - Ini menandakan bahwa tidak ada film yang sangat mirip secara konten dengan Inception dalam dataset ini. Namun, beberapa film memiliki cukup banyak elemen serupa untuk dijadikan rekomendasi.

2. Kemiripan berdasarkan genre dan tema kompleks
  - Film seperti Cypher dan Blood and Wine memiliki genre seperti Thriller, Science Fiction, dan Drama, yang juga dimiliki oleh Inception. Ini menunjukkan bahwa model menekankan pada genre dan elemen cerita (misalnya: "undercover", "robbery", "double life", dll) dalam menentukan kemiripan.

3. Tidak semua film memiliki keywords yang lengkap
  - Misalnya, The Helix... Loaded dan Duplex tidak memiliki data keywords, namun tetap masuk rekomendasi karena overview dan genre masih relevan.

4. Beberapa rekomendasi tampak kurang relevan secara tematik
  - Misalnya, Crouching Tiger, Hidden Dragon lebih ke arah aksi dan seni bela diri, berbeda secara substansi dengan tema psikologis dan mimpi seperti di Inception. Ini bisa terjadi karena kemiripan kata dalam deskripsi atau genre overlap seperti Action dan Drama.
    
## 5. Evaluation
Proyek ini bertujuan untuk melihat apakah rekomendasi film yang dihasilkan berdasarkan genre, keywords, dan overview mampu memberikan hasil yang akurat dan relevan. Namun, karena dataset yang digunakan tidak menyediakan data interaksi eksplisit antara pengguna dan film (seperti rating, klik, atau histori tontonan), evaluasi sistem rekomendasi tidak dapat dilakukan secara kuantitatif menggunakan metrik standar seperti Precision@K, Recall@K, MAP, atau NDCG.

Sebagai gantinya, evaluasi dilakukan secara kualitatif, yaitu dengan menganalisis hasil rekomendasi yang dihasilkan oleh model berbasis konten. Model ini menggunakan representasi teks gabungan dari overview, keywords, dan genre_list yang diolah menggunakan TF-IDF, serta perhitungan cosine similarity untuk menentukan kemiripan antar film.

Beberapa film dipilih sebagai input, dan daftar film yang direkomendasikan diamati secara manual. Penilaian dilakukan berdasarkan relevansi isi, kesamaan tema, genre, dan topik. Hasilnya menunjukkan bahwa model mampu merekomendasikan film-film yang secara konteks cukup serupa dengan film acuan, terutama jika teks pada overview, keywords, dan genre_list kaya dan deskriptif.

Catatan: TF-IDF dan cosine similarity digunakan sebagai metode perhitungan kemiripan konten, bukan sebagai metrik evaluasi performa sistem. (Kesalahan saya sebelumnya)

## Problem Answer (Untuk no 2 dan 3, perbandingan dengan 4.2 Penerapan)
### 1. Bagaimana pengguna dapat menemukan film yang mirip dengan film favorit mereka berdasarkan isi/konten film?
<p align="center">
   <img src="Assets/no1_fungsi.png"width="500"/>
   <img src="Assets/no1_jawab.png"width="500"/>
</p>

Intepretasi:

Berdasarkan gambar rekomendasi film yang mirip dengan Jurassic World, dapat disimpulkan bahwa sistem rekomendasi berbasis konten mampu memberikan hasil yang sangat relevan pada urutan teratas, terutama ketika film-film tersebut berasal dari waralaba atau franchise yang sama, seperti Jurassic Park, The Lost World, dan Jurassic Park III. Ketiga film ini memiliki skor kemiripan yang tinggi karena kesamaan yang kuat dalam genre, kata kunci, dan alur cerita.

Namun, setelah tiga besar, kualitas rekomendasi mulai menurun secara signifikan. Film-film seperti Vacation, The Nut Job, hingga Adventureland yang muncul selanjutnya memiliki tema, genre, dan konten yang tidak lagi selaras dengan Jurassic World, meskipun secara teknis masih memiliki elemen umum seperti “adventure” atau “comedy”.

Penurunan skor cosine similarity yang drastis ini menunjukkan bahwa model memiliki keterbatasan dalam mempertahankan relevansi ketika fitur-fitur yang digunakan terlalu umum. Oleh karena itu, untuk meningkatkan akurasi terutama di luar film-film yang sangat mirip, perlu dipertimbangkan penambahan fitur yang lebih mendalam seperti karakter, setting, atau analisis sentimen dari ulasan pengguna.

### 2. Apakah informasi tambahan seperti perusahaan produksi dapat digunakan untuk lebih memahami kesamaan antar film?
<p align="center">
   <img src="Assets/no2_fungsi.png"width="500"/>
   <img src="Assets/no2_tangled.png"width="500"/>
   <img src="Assets/no2_avatar.png"width="500"/>
</p>

Intepretasi:

Menambahkan informasi production_companies dapat meningkatkan akurasi rekomendasi, tetapi tergantung konteks film.

| Kondisi Film                                                 | Dampak Penambahan `production_companies`                                                                                          |
| ------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| **Film dengan gaya khas studio** (misal: *Tangled* - Disney) | **Meningkatkan akurasi** karena banyak film dari studio tersebut punya tone, genre, dan target audiens yang mirip.              |
| **Film dengan genre kuat dan lebih umum** (misal: *Avatar*)  | **Peningkatan akurasi terbatas**, karena genre seperti sci-fi lebih ditentukan oleh cerita, visual, dan tema, bukan studionya. |


Penambahan **production_companies** dapat meningkatkan akurasi, terutama untuk film yang dipengaruhi kuat oleh gaya studio (misalnya animasi Disney). Namun, untuk film dengan kekuatan utama di alur cerita atau genre luas seperti sci-fi, pengaruhnya lebih kecil.

### 3. Apakah informasi berupa keywords dan overview, tanpa melibatkan genre, dapat menghasilkan perhitungan cosine similarity yang lebih akurat dalam merekomendasikan film?
<p align="center">
   <img src="Assets/no3_fungsi.png"width="500"/>
   <img src="Assets/no3_jawab.png"width="500"/>
</p>

Intepretasi:

- Perbandingan Dua Pendekatan

| Pendekatan   | Fitur yang Digunakan               | Nilai Cosine Tertinggi | Judul Paling Mirip (Top 1) |
| ------------ | ---------------------------------- | ---------------------- | -------------------------- |
| Tanpa Genre  | `keywords + overview`              | 0.104511               | **Blood and Wine**         |
| Dengan Genre | `genre_list + keywords + overview` | 0.123394               | **Cypher**                 |

Insight Utama
- Dengan Genre Lebih Akurat

    - Nilai cosine similarity rata-rata lebih tinggi ketika genre disertakan, yang menunjukkan bahwa genre membantu memperkuat konteks tematik antar film.

    - Film seperti Cypher muncul di posisi atas pada kedua pendekatan, tapi skornya lebih tinggi saat genre digunakan.

- Tanpa Genre Lebih Bebas Tapi Kurang Tepat:

    - Saat hanya menggunakan keywords dan overview, hasil rekomendasi menjadi lebih bervariasi dan kurang relevan secara genre. Misalnya, Pitch Perfect 2 muncul meski jauh dari nuansa sci-fi/thriller seperti Inception.

    - Ini menunjukkan bahwa meskipun ada kemiripan deskripsi, genre tetap penting dalam menyaring konteks film yang tepat.

Kesimpulan

Informasi genre berperan penting dalam meningkatkan akurasi rekomendasi film. Tanpa genre, rekomendasi bisa melenceng ke film yang deskripsinya mirip secara permukaan, tapi tidak sejalan dari sisi tema atau pengalaman menonton.


# Referensi 
1. Arfisko, H. H., & Wibowo, A. T. (2022). Sistem Rekomendasi Film Menggunakan Metode Hybrid Collaborative Filtering Dan Content-Based Filtering. eProceedings of Engineering, 9(3).

2. Fajriansyah, M., Adikara, P. P., & Widodo, A. W. (2021). Sistem Rekomendasi Film Menggunakan Content Based Filtering. Jurnal Pengembangan Teknologi Informasi Dan Ilmu Komputer, 5(6), 2188–2199. Diambil dari https://j-ptiik.ub.ac.id/index.php/j-ptiik/article/view/9163

3. Saputra, J. M. A., Huizen, L. M., & Arianto, D. B. (2024). Sistem Rekomendasi Film pada Platform Streaming Menggunakan Metode Content-Based Filtering. Jurnal Transformatika, 22(1), 10-21.


© 2025 Firda Humaira
