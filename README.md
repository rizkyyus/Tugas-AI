# Tugas-AI
# 1. https://www.megabagus.id/deep-learning-convolutional-neural-networks/

Convolutional Neural Network (CNN) adalah jenis jaringan saraf tiruan yang sangat populer dalam pemrosesan gambar dan video. CNN digunakan untuk mengenali pola, mengklasifikasi objek, dan mendeteksi fitur-fitur dalam gambar. Berikut adalah penjelasan tahapan utama dalam CNN:

1. **Konvolusi (Convolution)**
2. **Pooling (Max Pooling)**
3. **Flattening**
4. **Koneksi Penuh (Fully Connected Layer)**



## 1. Konvolusi

Tahap konvolusi bertujuan untuk mengenali pola atau fitur dalam gambar. Misalkan kita memiliki gambar wajah, CNN akan menggunakan filter (seperti "kaca pembesar") untuk melihat bagian-bagian gambar, seperti mata, hidung, atau mulut.

### Cara Kerjanya
- Gambar dipindai menggunakan filter kecil (misalnya ukuran 3x3 piksel).
- Filter ini melakukan perkalian elemen dengan bagian gambar yang dilihat, kemudian dijumlahkan menjadi satu nilai.
- Hasilnya adalah **feature map**, yaitu gambar yang sudah diproses dan berisi pola yang terdeteksi.

### Contoh
Misalkan gambar memiliki ukuran 7x7 piksel dan filter berukuran 3x3. Filter akan "menyapu" gambar dari kiri ke kanan dan atas ke bawah, menghasilkan peta fitur yang lebih kecil.

### Mengapa Penting?
- Konvolusi membantu menemukan fitur penting, seperti tepi dan tekstur.
- Setelah konvolusi, fungsi aktivasi **ReLU (Rectified Linear Unit)** digunakan untuk menghapus nilai negatif, sehingga model hanya fokus pada fitur yang relevan.



## 2. Max Pooling

Max pooling digunakan untuk mengurangi ukuran gambar setelah konvolusi tanpa kehilangan fitur penting. Proses ini seperti mengambil "ringkasan" data dengan memilih nilai terbesar dari beberapa bagian gambar.

### Cara Kerjanya
- Gambar dibagi menjadi kotak-kotak kecil (misalnya ukuran 2x2 piksel).
- Dari setiap kotak, hanya nilai terbesar yang diambil.

### Contoh
Jika kita memiliki kotak 2x2 dengan nilai `[1, 3, 2, 5]`, maka nilai yang diambil adalah `5`.

### Mengapa Penting?
- Mengurangi ukuran data sehingga komputasi menjadi lebih cepat.
- Mengabaikan detail kecil yang tidak terlalu penting, tetapi tetap mempertahankan informasi utama.


## 3. Flattening

Flattening adalah proses mengubah data hasil pooling layer (berbentuk matriks dua dimensi) menjadi satu baris panjang (vektor). Ini seperti meratakan gambar menjadi daftar angka-angka panjang.

### Mengapa Diperlukan?
- Proses ini memudahkan input untuk dimasukkan ke dalam lapisan jaringan saraf tiruan yang memerlukan bentuk vektor satu dimensi.


## 4. Fully Connected Layer

Pada tahap ini, semua neuron saling terhubung (fully connected). Di sinilah model melakukan klasifikasi gambar berdasarkan fitur yang telah dipelajari.

### Cara Kerjanya
- Vektor hasil flattening menjadi input bagi lapisan jaringan saraf tiruan.
- Setiap neuron terhubung ke neuron di lapisan berikutnya.
- Fungsi aktivasi seperti **softmax** digunakan di lapisan akhir untuk menghasilkan probabilitas kelas. Misalnya, apakah gambar tersebut adalah anjing atau kucing.

### Contoh
Jika kita memiliki dua kelas: kucing dan anjing, dan model menghasilkan probabilitas `[0.2, 0.8]`, artinya gambar tersebut lebih mungkin adalah kucing (karena probabilitasnya lebih tinggi).




