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

# 2. https://www.megabagus.id/deep-learning-convolutional-neural-networks-aplikasi/

## Penjelasan Kode

### Kode 1: Membangun Model CNN untuk Klasifikasi Gambar Kucing dan Anjing

#### 1. **Import Library yang Diperlukan**
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```
- Mengimpor library dari **TensorFlow Keras** untuk membangun model CNN dan memproses data gambar.

#### 2. **Inisialisasi Model CNN**
```python
MesinKlasifikasi = Sequential()
```
- Membuat model **Sequential** yang memungkinkan kita menambahkan layer secara berurutan.

#### 3. **Langkah 1 - Convolution**
```python
MesinKlasifikasi.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(128, 128, 3), activation='relu'))
```
- Menambahkan **Conv2D layer**:
  - `filters=32`: Menggunakan 32 filter konvolusi.
  - `kernel_size=(3, 3)`: Ukuran filter adalah 3x3.
  - `input_shape=(128, 128, 3)`: Gambar input berukuran 128x128 dengan 3 channel warna (RGB).
  - `activation='relu'`: Menggunakan fungsi aktivasi **ReLU** untuk menghilangkan nilai negatif.

#### 4. **Langkah 2 - Pooling**
```python
MesinKlasifikasi.add(MaxPooling2D(pool_size=(2, 2)))
```
- Menambahkan **MaxPooling2D layer** dengan ukuran pooling (2, 2) untuk mengurangi dimensi fitur dan mencegah overfitting.

#### 5. **Menambah Convolutional Layer Kedua**
```python
MesinKlasifikasi.add(Conv2D(32, (3, 3), activation='relu'))
MesinKlasifikasi.add(MaxPooling2D(pool_size=(2, 2)))
```
- Menambahkan convolutional layer kedua dengan spesifikasi yang sama (32 filter, kernel 3x3, dan aktivasi ReLU), diikuti dengan pooling layer.

#### 6. **Langkah 3 - Flattening**
```python
MesinKlasifikasi.add(Flatten())
```
- **Flatten layer** mengubah data fitur 2D menjadi bentuk 1D untuk input ke dalam dense layer.

#### 7. **Langkah 4 - Full Connection**
```python
MesinKlasifikasi.add(Dense(units=128, activation='relu'))
MesinKlasifikasi.add(Dense(units=1, activation='sigmoid'))
```
- Menambahkan dua dense layer:
  - **Dense layer pertama** dengan 128 unit dan aktivasi ReLU.
  - **Output layer** dengan 1 unit dan aktivasi **sigmoid** untuk klasifikasi **biner** (kucing atau anjing).

#### 8. **Kompilasi Model**
```python
MesinKlasifikasi.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```
- Menggunakan **optimizer Adam**, **loss binary crossentropy**, dan metrik **akurasi**.

#### 9. **Data Augmentasi Menggunakan ImageDataGenerator**
```python
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
```
- `ImageDataGenerator` digunakan untuk augmentasi data:
  - **rescale**: Normalisasi piksel gambar menjadi nilai antara 0 dan 1.
  - **shear_range, zoom_range, horizontal_flip**: Transformasi untuk memperluas variasi data dan mengurangi overfitting.

#### 10. **Menyiapkan Dataset untuk Training dan Testing**
```python
training_set = train_datagen.flow_from_directory('dataset/training_set', target_size=(128, 128), batch_size=32, class_mode='binary')
test_set = test_datagen.flow_from_directory('dataset/test_set', target_size=(128, 128), batch_size=32, class_mode='binary')
```
- Mengambil gambar dari direktori:
  - **target_size**: Mengubah ukuran gambar menjadi 128x128 piksel.
  - **batch_size**: Mengatur ukuran batch menjadi 32.
  - **class_mode='binary'**: Karena ini adalah klasifikasi biner (kucing atau anjing).

#### 11. **Melatih Model**
```python
MesinKlasifikasi.fit(training_set, steps_per_epoch=8000 // 32, epochs=50, validation_data=test_set, validation_steps=2000 // 32)
```
- Melatih model:
  - **steps_per_epoch**: Jumlah langkah per epoch dihitung sebagai total sampel dibagi ukuran batch.
  - **epochs=50**: Model dilatih selama 50 epoch.
  - **validation_data**: Menggunakan test set untuk validasi.
  - **validation_steps**: Jumlah langkah per validasi dihitung sebagai total sampel validasi dibagi ukuran batch.

---

### Kode 2: Menguji Model pada Dataset Test Set

#### 1. **Import Library yang Diperlukan**
```python
import numpy as np
from keras.preprocessing import image
```
- Mengimpor library untuk manipulasi array dan pemrosesan gambar.

#### 2. **Menginisialisasi Variabel untuk Menghitung Prediksi**
```python
count_dog = 0
count_cat = 0
```
- Menginisialisasi penghitung untuk jumlah prediksi **anjing** dan **kucing**.

#### 3. **Melakukan Prediksi pada Gambar Uji**
```python
for i in range(4001, 5001): 
    test_image = image.load_img('dataset/test_set/dogs/dog.' + str(i) + '.jpg', target_size=(128, 128))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    result = MesinKlasifikasi.predict(test_image)
    training_set.class_indices
    if result[0][0] == 0:
        prediction = 'cat'
        count_cat += 1
    else:
        prediction = 'dog'
        count_dog += 1
```
- Melakukan iterasi dari file gambar `dog.4001.jpg` hingga `dog.5000.jpg`:
  - Mengubah ukuran gambar menjadi **128x128**.
  - Mengonversi gambar menjadi array dan menambahkan **batch dimension**.
  - Membuat prediksi dengan model.
  - Jika hasil prediksi adalah `0`, maka prediksi adalah **kucing**; jika `1`, maka prediksi adalah **anjing**.

#### 4. **Mencetak Hasil Prediksi**
```python
print("count_dog:" + str(count_dog))    
print("count_cat:" + str(count_cat))
```
- Menampilkan jumlah gambar yang diprediksi sebagai **anjing** dan **kucing**.

---

### **Kesimpulan**
- **Kode 1** membangun dan melatih model CNN untuk mengklasifikasikan gambar sebagai **kucing** atau **anjing** menggunakan dataset yang diambil dari direktori lokal.
- **Kode 2** digunakan untuk menguji model pada dataset uji dan menghitung jumlah prediksi untuk masing-masing kelas.
- Teknik **augmentasi data** dan penggunaan **MaxPooling** membantu mengurangi risiko overfitting dan meningkatkan performa model.

# 3. https://modul-praktikum-ai.vercel.app/Materi/4-convolutional-neural-network

## Penjelasan Kode

### Kode 1: Pelatihan Model CNN dengan CIFAR-10

#### 1. **Import Library**
```python
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
```
- Mengimpor TensorFlow dan modul terkait untuk pemrosesan data dan pembangunan model.

#### 2. **Memuat Dataset CIFAR-10**
```python
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
```
- Memuat dataset **CIFAR-10**, yang terdiri dari gambar 32x32 piksel dengan 10 kelas (misalnya pesawat, mobil, kucing, anjing, dll.).

#### 3. **Normalisasi Data Gambar**
```python
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0
```
- Mengubah tipe data gambar menjadi float32 dan melakukan normalisasi (nilai antara 0 dan 1) agar model lebih cepat konvergen.

#### 4. **Mengonversi Label ke Bentuk Kategorikal**
```python
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)
```
- Mengonversi label yang semula berupa angka menjadi **one-hot encoding** untuk klasifikasi multi-kelas (10 kelas).

#### 5. **Membangun Model CNN**
```python
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])
```
- **Sequential model** dibangun dengan beberapa layer:
  - **Conv2D**: Layer konvolusi dengan filter 32, 64, dan 128 serta ukuran kernel (3, 3).
  - **MaxPooling2D**: Layer pooling dengan ukuran (2, 2) untuk mengurangi dimensi fitur.
  - **Flatten**: Mengubah data 2D menjadi 1D sebelum masuk ke layer Dense.
  - **Dense**: Layer fully connected dengan 128 unit dan Dropout 50% untuk mencegah overfitting.
  - **Output layer**: Menggunakan aktivasi **softmax** karena ada 10 kelas.

#### 6. **Menampilkan Ringkasan Model**
```python
model.summary()
```
- Menampilkan arsitektur model dan jumlah parameter pada setiap layer.

#### 7. **Kompilasi Model**
```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```
- Menggunakan **optimizer Adam**, **loss categorical crossentropy**, dan **metrik akurasi**.

#### 8. **Pelatihan Model**
```python
history = model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_data=(test_images, test_labels))
```
- Melatih model dengan:
  - **epochs** = 10 (jumlah iterasi)
  - **batch_size** = 64
  - Menggunakan data validasi dari data tes untuk memonitor performa model.

#### 9. **Evaluasi Model**
```python
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
```
- Mengevaluasi model pada data tes dan mencetak akurasi.

---

### Kode 2: Prediksi Gambar Menggunakan Model yang Telah Dilatih

#### 1. **Import Library**
```python
from google.colab import files
from keras.models import load_model
from PIL import Image
import numpy as np
```
- Mengimpor library yang diperlukan:
  - `files`: untuk mengunggah file gambar.
  - `load_model`: untuk memuat model yang telah disimpan.
  - `Image`: untuk memproses gambar.
  - `numpy`: untuk manipulasi array.

#### 2. **Fungsi untuk Memuat dan Memproses Gambar**
```python
def load_and_prepare_image(file_path):
    img = Image.open(file_path)
    img = img.resize((32, 32))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img
```
- Membaca gambar menggunakan **PIL**, mengubah ukurannya menjadi **32x32**, melakukan normalisasi, dan menambahkan **batch dimension**.

#### 3. **Daftar Nama Kelas CIFAR-10**
```python
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
```
- Mendefinisikan nama kelas sesuai dengan urutan pada dataset CIFAR-10.

#### 4. **Memuat Model yang Telah Disimpan**
```python
# model = load_model('path_to_your_model.h5')
```
- Memuat model yang telah disimpan. (Baris ini dikomentari, ganti dengan path model yang benar).

#### 5. **Mengunggah dan Memproses File Gambar**
```python
uploaded = files.upload()
```
- Mengunggah file gambar dari perangkat pengguna.

#### 6. **Membuat Prediksi**
```python
for filename in uploaded.keys():
    img = load_and_prepare_image(filename)
    prediction = model.predict(img)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    predicted_class_name = class_names[predicted_class_index]
    print(f'File: {filename}, Predicted Class Index: {predicted_class_index}, Predicted Class Name: {predicted_class_name}')
```
- Untuk setiap gambar yang diunggah:
  - Memuat gambar menggunakan fungsi `load_and_prepare_image`.
  - Membuat prediksi menggunakan model yang telah dilatih.
  - Mendapatkan indeks kelas dengan nilai probabilitas tertinggi.
  - Mengonversi indeks kelas menjadi nama kelas dan mencetak hasil prediksi.

---

### **Kesimpulan**
- **Kode 1** membangun dan melatih model CNN menggunakan dataset CIFAR-10 untuk klasifikasi gambar.
- **Kode 2** digunakan untuk memprediksi kelas gambar baru menggunakan model yang telah dilatih.




