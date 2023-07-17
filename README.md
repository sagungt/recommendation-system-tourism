# Laporan Proyek Machine Learning - Nama Anda

## Project Overview

Kamus Besar Bahasa Indonesia menjelaskan  bahwa pariwisata merupakan suatu kegiatan yang  berhubungan dengan perjalanan rekreasi. Berdasarkan UU No. 10 Tahun 2009 tentang Kepariwisataan, pariwisata memiliki pengertian berupa berbagai macam kegiatan wisata dan didukung berbagai fasilitas serta layanan yang disediakan oleh masyarakat, pengusaha, pemerintah, dan pemerintah daerah. Peningkatan devisa negara dari sektor pariwisata menjadikan kawasan pariwisata harus terus dibangun dan dijaga sebagai salah satu aset negara yang berpotensi. Selain itu, pariwisata juga merupakan sektor ekonomi alternatif yang dipandang mampu mempercepat penanggulangan kemiskinan di Indonesia (Yoeti, 2008). Pariwisata merupakan gejala yang kompleks dalam masyarakat, yang di dalamnya terdapat hotel, objek wisata, usaha souvenir, pramuwisata, angkutan wisata, biro perjalanan wisata, rumah makan dan banyak lainnya (Soekadijo, 1997). McIntoshi R. dan Gupta S. dalam (Muntasib & Rachmawati, 2014) menyatakan pariwisata sebagai berikut: 
_“Pariwisata adalah gabungan gejala dari hubungan yang timbul dari interaksi wisatawan, bisnis, pemerintah, tuan rumah, serta masyarakat tuan rumah dalam proses menarik dan melayani wisatawan-wisatawan serta para pengunjung lainnya.”_ [1]

Salah satu masalah yang terjadi di masyarakat yaitu kebingungan memilih tempat wisata yang ada di suatu daerah tertentu. Selain itu kurangnya informasi akan membuat wisatawan menjadi kurang tepat saat memilih tempat wisata yang akan dikunjungi. Penelitian ini dilakukan untuk membuat suatu sistem rekomendasi tempat wisata bagi pengguna yang kurang memiliki informasi tentang tempat wisata di daerah yang sedang dikunjungi berdasarkan kriteria yang diinginkan dengan hasil yang telah diberikan peringkat. Sistem rekomendasi merupakan sebuah teknik untuk memberikan informasi berupa saran yang sesuai kepada pengguna. [2]

Sistem rekomendasi memiliki beberapa keuntungan bagi pengguna maupun pihak pengelola tempat pariwisata contohnya, pengguna bisa mendapatkan rekomendasi tempat wisata yang relevan baginya tanpa harus melakukan riset atau pencarian yang memungkinkan memakan waktu. Sedangkan bagi pengelola pariwisata dapat secara tidak langsung promosi untuk tempat wisata yang dikelolanya tanpa mengeluarkan uang tambahan untuk promosi. Ada beberapa kauntungan lainnya juga yaitu tempat wistata yang jarang terekspos juga bisa mendapat promosi jika sesuai dengan preferensi pengguna.

## Business Understanding

### Problem Statements

Menjelaskan pernyataan masalah:
- Bagaimana cara memahami data ?
- Bagaimana cara mengolah data untuk model ?
- Bagaimana cara membuat model rekomendasi dengan performa yang bagus berdasarkan konten ?
- Bagaimana cara membuat model rekomendasi dengan performa yang bagus berdasarkan perilaku pengguna ?

### Goals

Menjelaskan tujuan proyek yang menjawab pernyataan masalah:
- Untuk memahami data, cara yang akan dilakukan yaitu dengan EDA (_Exploration Data Analysys_):
  - Memahami setiap fitur data dan beserta tipe data-nya.
  - Melakukan visualisasi data dengan bertujuan agar lebih mudah dipahami. Visualisasi yang akan digunakan yaitu _bar chart_ dan _line chart_.
- Untuk mengolah data, beberapa teknik yang dilakukan:
  - Memilih fitur apa yang akan digunakan.
  - Augmentasi fitur. Fitur yang akan diaugmentasi yaitu fitur Price yang semula numerik menjadi teks kategorikal.
  - Melakukan pembagian data untuk _train_ dan _validation_ dengan rasio 8:2.
  - Melakukan normalisasi data. Data yang akan di normalisasi yaitu rating yang semula dalam skala 1 sampai 5, menjadi skala 0 sampai 1.
- Untuk model rekomendasi berdasarkan konten, teknik yang akan digunakan:
  - Melakukan transformasi teks menjadi vektor menggunakan TFIDF (_Term Frequency-Inverse Document Frequency_)
```math
w_{i, j} = tf_{i, j} \times idf_{i}
```
  - Menggunakan _cosine similarity_ untuk mengukur kesamaan (_similarity_) antar tempat wisata.
```math
Cosine(x,y) = \frac{x \cdot y}{|x||y|}
```
  - Menggunakan metrik _precision_ untuk tolak ukur presisi rekomendasi.
```math
\begin{array}{rcl}
precision & = & \dfrac{\text{Relevant recommendations}}{\text{Top items recommended}}
\end{array}
``````
- Untuk model rekomendasi berdasarkan perilaku pengguna, teknik yang akan dilakukan:
  - Menggunakan teknik _embedding_ untuk membuat model bernama **RecommenderNet** dengan model _library_ [keras Model class](https://keras.io/api/models/model/).
  - Menggunakan metrik RMSE (_Root Mean Squared Error_)

```math
\begin{array}{rcl}
\text{RMSE}(y, \hat{y}) = \sqrt{\frac{\sum_{i=0}^{N - 1} (y_i - \hat{y}_i)^2}{N}}
\end{array}
```


## Data Understanding

Dataset ini berisi beberapa tempat wisata di 5 kota besar di Indonesia yaitu Jakarta, Bandung, Semarang, Suarabaya dan Yogyakarta. Dataset ini berisi 4 berkas berisikan tentang paket, info pariwisata, info user dan rating tempat dari user. [Indonesia Tourism Destination](https://www.kaggle.com/datasets/aprabowo/indonesia-tourism-destination).

- **package_tourism.csv**: merupakan dataset untuk paket-paket wisata berisikan 100 baris data dengan fitur:
  - **Package**: merupakan ID dari paket tour wisata
  - **City**: merupakan kota dari paket tempat wisata
  - **Place_Tourism1**: merupakan destinasi wisata pertama dalam paket
  - **Place_Tourism2**: merupakan destinasi wisata kedua dalam paket
  - **Place_Tourism3**: merupakan destinasi wisata ketiga dalam paket
  - **Place_Tourism4**: merupakan destinasi wisata keempat dalam paket
  - **Place_Tourism5**: merupakan destinasi wisata kelima dalam paket
- **tourism_with_id.csv**: merupakan dataset untuk tempat wisata berikikan 437 baris data dengan fitur:
  - **Place_Id**: merupakan ID dari tempat wisata
  - **Place_Name**: merupakan nama tempat wisata
  - **Description**: merupakan deskripsi dari tempat wisata
  - **Category**: merupakan kategori tempat wisata
  - **City**: merupakan kota dari tempat wisata
  - **Price**: merupakan biaya masuk ke tempat wisata
  - **Rating**: merupakan penilaian tempat wisata
  - **Time_Minutes**: -
  - **Coordinate**: merupakan titik koordinat dari tempat wisata
  - **Lat**: merupakan nilai _latitude_ dari tempat wisata
  - **Long**: merupakan nilai _longitude_ dari tempat wisata
  - **Unnamed: 11**: -
  - **Unnamed: 12**: -
- **user.csv**: merupakan dataset informasi pengguna berisikan 300 baris data dengan fitur:
  - **User_Id**: merupakan ID dari pengguna
  - **Location**: merupakan lokasi tempat tinggal pengguna (Kota, Provinsi)
  - **Age**: merupakan umur dari pengguna
- **tourism_rating.csv**: merupakan dataset penilaian-penilaian pengguna terhadap tempat wisata yang telah dikunjunginya berisikan 10000 baris data dengan fitur:
  - **Place_Id**: merupakan ID dari tempat wisata
  - **User_Id**: merupakan ID dari pengguna
  - **Place_Ratings**: merupakan penilaian tempat wisata yang telah dikunjungi pengguna

Informasi dataset paket
```
RangeIndex: 100 entries, 0 to 99
Data columns (total 7 columns):
 #   Column          Non-Null Count  Dtype 
---  ------          --------------  ----- 
 0   Package         100 non-null    int64 
 1   City            100 non-null    object
 2   Place_Tourism1  100 non-null    object
 3   Place_Tourism2  100 non-null    object
 4   Place_Tourism3  100 non-null    object
 5   Place_Tourism4  66 non-null     object
 6   Place_Tourism5  39 non-null     object
dtypes: int64(1), object(6)
```

Informasi dataset tempat wisata
```
RangeIndex: 437 entries, 0 to 436
Data columns (total 13 columns):
 #   Column        Non-Null Count  Dtype  
---  ------        --------------  -----  
 0   Place_Id      437 non-null    int64  
 1   Place_Name    437 non-null    object 
 2   Description   437 non-null    object 
 3   Category      437 non-null    object 
 4   City          437 non-null    object 
 5   Price         437 non-null    int64  
 6   Rating        437 non-null    float64
 7   Time_Minutes  205 non-null    float64
 8   Coordinate    437 non-null    object 
 9   Lat           437 non-null    float64
 10  Long          437 non-null    float64
 11  Unnamed: 11   0 non-null      float64
 12  Unnamed: 12   437 non-null    int64  
dtypes: float64(5), int64(3), object(5)
```
Informasi dataset pengguna
```
RangeIndex: 300 entries, 0 to 299
Data columns (total 3 columns):
 #   Column    Non-Null Count  Dtype 
---  ------    --------------  ----- 
 0   User_Id   300 non-null    int64 
 1   Location  300 non-null    object
 2   Age       300 non-null    int64 
dtypes: int64(2), object(1)
```
Informasi data rating
```
RangeIndex: 10000 entries, 0 to 9999
Data columns (total 3 columns):
 #   Column         Non-Null Count  Dtype
---  ------         --------------  -----
 0   User_Id        10000 non-null  int64
 1   Place_Id       10000 non-null  int64
 2   Place_Ratings  10000 non-null  int64
dtypes: int64(3)
```
Rata-rata harga tempat wisata per kategori bervariasi dari Rp 0 yang berarti gratis sampai Rp 800.000 dengan kategori paling mahal yaitu bahari dan taman hiburan.

Gambar 1. Rata-rata harga tempat wisata per kategori

Gambar 2. Jumlah _rating_ 1-5 tempat wisata per kategori

_Rating_ tempat wisata dan _rating_ dari user memiliki perbedaan rata-rata.

Gambar 3. Rata-rata _rating_ tempat wisata dan _rating_ dari user

Kategori tempat wisata yang paling banyak dikunjungi yaitu tempat wisata, budaya dan cagar alam.

Gambar 4. Jumlah pengunjung tempat wisata per kategori

Rata-rata tempat wisata dengan harga tertinggi yaitu Jakarta.

Gambar 5. Rata-rata harga tempat wisata per kota

Gambar 6. Jumlah _rating_ 1-5 tempat wisata per kota

Kota dengan pengunjung tempat wisata tertinggi yaitu Bandung dan Yogyakarta.

Gambar 7. Jumlah pengunjung tempat wisata per kota

Harga tempat wisata bervariasi, fitur harga dapat di-augmentasi untuk menghasilkan fitur baru yaitu Budget.

Gambar 8. Harga tempat wisata

## Data Preparation

Fitur yang akan digunakan yaitu Category, City, Description dan Budget yang akan diaugmentasi dari fitur Price.

Konten yang akan dijadikan sebagai _filtering_ yaitu gabungan dari keempat fitur diatas. Kumpulan data yang telah digabung ini bisa dinamakan dengan **Metadata**.

Fitur Price dapat diaugmentasi karena data numerik tidak bagus untuk _content based filtering_. Fitur Harga dapat dipetakan menjadi pengeluaran rata-rata yaitu Budget.

```
Price == 0 (Gratis)
Price > 0 & Price <= 100000 (Murah)
Price > 100000 & Price <= 200000 (Lumayan)
Price > 200000 (Mahal)
```

Kode dalan _syntax_ python:
```
def set_budget(price):
  if price == 0:
    return 'Gratis'
  elif price <= 100000:
    return 'Murah'
  elif price <= 20000:
    return 'Lumayan'
  else:
    return 'Mahal'
```

Tabel 1. Daftar sebaran Budget harga
|Budget|Jumlah|
|-|-|
|Gratis|137|
|Murah|276|
|Lumayan|16|
|Mahal|8|

_Feature augmentation_ merupakan strategi yang menggabungkan fitur asli dan fitur baru untuk mempertahankan informasi berguna dari data asli. Dalam konteks analisis data, augmentasi fitur digunakan untuk meningkatkan performa model pembelajaran mesin dengan menambahkan fitur baru ke data asli. [3]

Buat fitur baru bernama Metadata dengan perintah berikut:
```
features = ['Category', 'City', 'Budget', 'Description']

df_tourism['Metadata'] = ''

for feature in features:
  df_tourism['Metadata'] += (df_tourism[feature].str.lower() + ' ')

df_tourism['Metadata']
```

Pembagian dataset dapat dilakukan dengan pembagian berdasarkan index dilakukan dengan index _slicing_. Data untuk _test_ sebesar 80% dan untuk _validation_ sebesar 20%.

```
index = int(.8 * dataset.shape[0])
X_train, X_val, y_train, y_val = [
    X[:index],
    X[index:],
    y[:index],
    y[index:],
] 
```

Normalisasi dengan sederhana data dapat dilakukan tanpa menggunakan _library_. Dapat menggunakan formula:
```math
f(x) = \frac{x - min}{max - min}
```

## Modeling

- _Content Based Filtering_
  _Content Based Filtering_ adalah algoritma rekomendasi untuk menemukan saran serupa tentang sesuatu. Pada algoritma ini setiap nilai unik dalam kumpulan data diberikan kata kunci atau atribut yang membantu mereka untuk dikenali. Kemudian berdasarkan pola ini, informasi tentang suka dan tidak suka pengguna disimpan, merekomendasikan item yang relevan. [4]
  Untuk mengubah Metadata menjadi vektor, teknik yang akan digunakan yaitu TF-IDF (_Term Frequency-Inverse Document Frequency_). 
  TD-IDF dapat didefinisikan sebagai perhitungan seberapa relevan sebuah kata dalam rangkaian atau korpus dengan sebuah teks. Makna meningkat secara proporsional dengan berapa kali dalam teks sebuah kata muncul tetapi dikompensasi oleh frekuensi kata dalam korpus (kumpulan data). [5]
  Setelah matriks didapat, top-n rekomendasi berdasarkan konten dapat diproses dengan menghitung kesamaan/_similarity_. Teknik menghitung kesamaan/_similarity_ yang akan digunakan yaitu _cosine similarity_.
  _Cosine Similarity_ adalah metrik yang digunakan untuk mengukur seberapa mirip dokumen terlepas dari ukurannya. Secara matematis, ini mengukur kosinus sudut antara dua vektor yang diproyeksikan dalam ruang multidimensi. [6] 

  Gambar 9. _Cosine Similarity_

  **Kelebihan**:
  - Model tidak memerlukan data apa pun tentang pengguna lain, karena rekomendasinya khusus untuk pengguna ini. Ini membuatnya lebih mudah untuk menskalakan ke sejumlah besar pengguna.
  - Model tersebut dapat menangkap minat khusus pengguna, dan dapat merekomendasikan item khusus yang sangat sedikit diminati oleh pengguna lain.
  **Kekurangan**:
  - Karena representasi fitur dari item direkayasa dengan tangan sampai batas tertentu, teknik ini memerlukan banyak pengetahuan domain/_domain specific knowlegde_. Oleh karena itu, modelnya hanya bisa sebagus fitur rekayasa tangan.
  - Model hanya dapat membuat rekomendasi berdasarkan minat pengguna yang ada. Dengan kata lain, model memiliki kemampuan terbatas untuk memperluas minat pengguna yang ada.

- _Collaborative Filtering_
  _Colaborative Filtering_ adalah metode yang semata-mata didasarkan pada interaksi masa lalu yang telah direkam antara pengguna dan item, guna menghasilkan rekomendasi baru. _Collaborative Filtering_ cenderung menemukan apa yang diinginkan oleh pengguna yang serupa dan rekomendasi yang akan diberikan dan untuk mengklasifikasikan pengguna ke dalam kelompok jenis yang serupa dan merekomendasikan setiap pengguna sesuai dengan preferensi kelompoknya. [7]
  Teknik yang akan digunakan untuk membangun model rekomendasi _Collaborative Filtering_ adalah dengan teknik _Embedding_.
  _Embedding_ adalah ruang berdimensi relatif rendah tempat Anda dapat menerjemahkan vektor berdimensi tinggi. _Embedding_ mempermudah melakukan pembelajaran mesin pada input besar seperti vektor renggang yang mewakili kata. Idealnya, sebuah _embedding_ menangkap beberapa semantik input dengan menempatkan input semantik yang mirip secara berdekatan di ruang embedding. Penyematan dapat dipelajari dan digunakan kembali di seluruh model. [8]

  **Kelebihan**:
  - Tidak memerlukan pengetahuan domain/_domain specific knowledge_ karena model mempelajari sendiri secara otomatis.
  - Model tersebut dapat membantu pengguna menemukan minat baru. Secara terpisah, model mungkin tidak mengetahui bahwa pengguna tertarik pada item tertentu, tetapi model mungkin masih merekomendasikannya karena pengguna serupa tertarik pada item tersebut.
  - Sampai batas tertentu, sistem hanya membutuhkan matriks umpan balik untuk melatih model faktorisasi matriks. Secara khusus, sistem tidak memerlukan fitur kontekstual.
  **Kekurangan**:
  - Riwayat pengguna sebelumnya diperlukan atau data untuk produk diperlukan berdasarkan jenis metode kolaboratif yang digunakan.
  - Item baru tidak dapat direkomendasikan jika tidak ada pengguna yang berinteraksi dengan target item.
  - Sulit untuk memasukan fitur lain untuk rekomendasi.

## Evaluation
Evaluasi untuk _content based filtering_ yaitu dengan metrik _precision_ sedangakan untuk model _collaborative filtering_ yaitu dengan metrik RMSE (_Root Mean Squared Error_).

### Precision

_Precision_ adalah salah satu indikator kinerja model pembelajaran mesin – kualitas prediksi positif yang dibuat oleh model. Presisi mengacu pada jumlah positif sejati dibagi dengan jumlah total prediksi positif. [9]
Formula _precision_ yang digunakan:

```math
\begin{array}{rcl}
precision & = & \dfrac{\text{Relevant recommendations}}{\text{Top items recommended}}
\end{array}
``````

Tabel 2. Hasil top 10 rekomendasi _content based_ untuk tempat wisata Kota Tua
|Nama Tempat Wisata|Kategori|
|-|-|
|Alun-alun Utara Keraton Yogyakarta|Budaya|
|Alun Alun Selatan Yogyakarta|Tamah Hiburan|
|Museum Sonobudoyo Unit I|Budaya|
|Alun-Alun Kota Bandung|Taman Hiburan|
|Museum Bahari Jakarta|Budaya|
|Museum Seni Rupa dan Kramik|Budaya|
|Museum Wayang|Budaya|
|Masjid Agung Ungaran|Tempat Ibadah|
|Museum Fatahillah|Budaya|
|Museum Nasional|Budaya|

Hasil _precision_ dari rekomendasi diatas adalah $ p = \frac{7}{10} = 70\% $. Hasil rekomendasi sudah cukup memberikan hasil yang lumayan bagus.

### RMSE (_Root Mean Squared Error_)

RMSE mengukur perbedaan rata-rata antara nilai prediksi model statistik dan nilai sebenarnya. Secara matematis, ini adalah standar deviasi dari residual. Residu mewakili jarak antara garis regresi dan titik data. RMSE mengkuantifikasi seberapa tersebar residu ini, mengungkapkan seberapa erat cluster data yang diamati di sekitar nilai prediksi. [10]
Formula RMSE:
```math
\begin{array}{rcl}
\text{RMSE}(y, \hat{y}) = \sqrt{\frac{\sum_{i=0}^{N - 1} (y_i - \hat{y}_i)^2}{N}}
\end{array}
```

Gambar 10. Hasil _test loss_ dan _validation loss_ dari model

Hasil dari _test loss_ dan _validation loss_ terlihat ada _overfitting_. Untuk menghindari overfitting dalam _embedding_, salah satunya dengan menggunakan data pelatihan yang lebih lengkap, meningkatkan kompleksitas model, meningkatkan waktu pelatihan, hingga _cost function_ diminimalkan. Kumpulan data harus mencakup berbagai input yang diharapkan dapat ditangani oleh model. Data tambahan mungkin hanya berguna jika mencakup kasus baru dan menarik.
Tabel 3. Tempat dengan rating tertinggi dari user  14
|Nama Tempat Wisata|Kategori|
|-|-|
|Margasatwa Muara Angke|Cagar Alam|
|Situs Warungboto|Tamah Hiburan|
|Upside Down World Bandung|Taman Hiburan|
|Gua Pawon|Cagar Alam|
|Semarang Chinatown|Budaya|

Tabel 4. Top 10 Rekomendasi tempat wisata berdasarkan user 14
|Nama Tempat Wisata|Kategori|
|-|-|
|Taman Situ Lembang|Taman Hiburan|
|Monumen Batik Yogyakarta|Budaya|
|Gumuk Pasir Parangkusumo|Taman Hiburan|
|Water Park Bandung Indah|Taman Hiburan|
|Sendang Geulis Kahuripan|Cagar Alam|
|Taman Hutan Raya Ir. H. Juanda|Cagar Alam|
|Wisata Alam Wana Wisata Penggaron|Cagar Alam|
|Kampoeng Rawa|Cagar Alam|
|Benteng Pendem|Budaya|
|Semarang Chinatown|Budaya|

## Referensi

[1] Annisya Rakha Anandhyta. Rilus A. Kinseng "Hubungan Tingkat Partisipasi dengan Tingkat Kesejahteraan Masyarakat dalam Pengembangan Wisata Pesisir". Jurnal Nasional Pariwisata, Volume 12, Nomor 2, September 2020 ISSN Cetak: 1411 - 9862. [https://jurnal.ugm.ac.id/tourism_pariwisata/article/download/60398/29532](https://jurnal.ugm.ac.id/tourism_pariwisata/article/download/60398/29532). [Accessed July 16 2023]
[2] Agustina, Wina (2017) SISTEM REKOMENDASI TEMPAT WISATA MENGGUNAKAN METODE KNOWLEDGE-BASED RECOMMENDATION DAN AHP-TOPSIS. S1 thesis, Universitas Pendidikan Indonesia. [http://repository.upi.edu/31888/](http://repository.upi.edu/31888/). [Accessed July 16 2023]
[3] Zhang, Fan. Bales, Chris. Fleyeh, Hasan. Feature Augmentation of Classifiers Using Learning Time Series Shapelets Transformation for Night Setback Classification of District Heating Substations. Advances in Civil Engineering. 10.1155/2021/8887328. [https://doi.org/10.1155/2021/8887328](https://doi.org/10.1155/2021/8887328) [Accessed July 16 2023]
[4] BINAY KUMAR GUPTA. Content Based Filtering in Machine Learning. [https://www.scaler.com/topics/machine-learning/content-based-filtering/](https://www.scaler.com/topics/machine-learning/content-based-filtering/) [Accessed July 16 2023]
[5] riturajsaha. Understanding TF-IDF (Term Frequency-Inverse Document Frequency) [https://www.geeksforgeeks.org/understanding-tf-idf-term-frequency-inverse-document-frequency/](https://www.geeksforgeeks.org/understanding-tf-idf-term-frequency-inverse-document-frequency/) [Accessed July 16 2023]
[6] Selva Prabhakaran. Cosine Similarity – Understanding the math and how it works (with python codes) [https://www.machinelearningplus.com/nlp/cosine-similarity/](https://www.machinelearningplus.com/nlp/cosine-similarity/) [Accessed July 16 2023]
[7] Victor Dey. Collaborative Filtering Vs Content-Based Filtering for Recommender Systems. [https://analyticsindiamag.com/collaborative-filtering-vs-content-based-filtering-for-recommender-systems/](https://analyticsindiamag.com/collaborative-filtering-vs-content-based-filtering-for-recommender-systems/) [Accessed July 16 2023]
[8] Google Developers. Embeddings. [https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture](https://developers.google.com/machine-learning/crash-course/embeddings/video-lecture) [Accessed July 16 2023]
[9] C3.ai. Precision [https://c3.ai/glossary/machine-learning/precision/](https://c3.ai/glossary/machine-learning/precision/) [Accessed July 16 2023]
[10]