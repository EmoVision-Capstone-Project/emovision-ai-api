# EmoVision AI API

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/Lisensi-MIT-green?style=for-the-badge"/>
</p>

<p align="center">
  REST API Multimodal berperforma tinggi untuk deteksi emosi secara real-time, mengombinasikan analisis ekspresi wajah (Computer Vision via FastAPI & TensorFlow) dan analisis teks jurnal (NLP via mBERT di Hugging Face). Mampu mengenali 7 kelas emosi secara holistik beserta skor kepercayaan dan rincian distribusi probabilitas lengkap.
</p>

---

## Daftar Isi

- [Gambaran Umum](#gambaran-umum)
- [Arsitektur Multimodal](#arsitektur-multimodal)
- [Fitur](#fitur)
- [Kelas Emosi](#kelas-emosi)
- [Teknologi yang Digunakan](#teknologi-yang-digunakan)
- [Memulai & Instalasi](#memulai--instalansi)
  - [1. Setup API Deteksi Wajah (Lokal)](#1-setup-api-deteksi-wajah-lokal)
  - [2. Setup API Deteksi Teks NLP (Lokal / Hugging Face)](#2-setup-api-deteksi-teks-nlp-lokal--hugging-face)
- [Referensi API](#referensi-api)
  - [1. API Deteksi Wajah (Image)](#1-api-deteksi-wajah-image)
  - [2. API Deteksi Teks Jurnal (NLP)](#2-api-deteksi-teks-jurnal-nlp)
- [Struktur Proyek](#struktur-proyek)

---

## Gambaran Umum

EmoVision AI adalah inti machine learning dari **EmoVision Capstone Project** yang mengusung pendekatan multimodal (Teks & Gambar). Sistem ini mengintegrasikan dua layanan utama: API analisis teks yang memproses curhatan atau jurnal pengguna menggunakan arsitektur Transformer mBERT via Hugging Face Spaces, serta API deteksi ekspresi wajah yang menerima file gambar, mendeteksi dan memotong wajah secara otomatis menggunakan Haar Cascade OpenCV, lalu menjalankan inferensi melalui model TensorFlow (SavedModel) lokal. Kedua subsistem ini menyelaraskan hasil untuk mengembalikan prediksi 7 kelas emosi secara holistik beserta skor kepercayaan dan distribusi probabilitas yang lengkap.

---

## Fitur

- **Deteksi Wajah Otomatis** — Menggunakan Haar Cascade untuk menemukan dan memotong wajah terbesar dalam gambar; jika tidak ada wajah yang terdeteksi, dilakukan center crop secara otomatis.
- - **Analisis Sentimen Jurnal** — Memahami konteks bahasa (NLP) dari tulisan curhatan/jurnal pengguna menggunakan mBERT.
- **Pengenalan 7 Kelas Emosi** — Mengklasifikasikan ekspresi wajah ke dalam 7 kategori emosi yang berbeda.
- **Output Kepercayaan & Probabilitas** — Mengembalikan emosi teratas beserta rincian probabilitas untuk semua kelas.
- **Cepat & Ringan** — Penanganan request asinkron melalui FastAPI dengan overhead minimal.
- **Mudah Diintegrasikan** — REST API standar yang kompatibel dengan klien HTTP, aplikasi mobile, maupun web frontend.

---

## Kelas Emosi

Model mengenali 7 kategori ekspresi wajah berikut:

| Label | Deskripsi |
|-------|-----------|
| 😠 Angry | Ekspresi marah atau frustrasi |
| 🤢 Disgust | Ekspresi jijik atau muak |
| 😨 Fear | Ekspresi takut atau cemas |
| 😊 Happy | Ekspresi bahagia atau senang |
| 😐 Neutral | Ekspresi netral atau diam |
| 😢 Sad | Ekspresi sedih atau murung |
| 😲 Surprise | Ekspresi terkejut atau kaget |

---

## Teknologi yang Digunakan

| Teknologi | Peran |
|-----------|-------|
| **FastAPI** | Framework web & routing API |
| **Uvicorn** | ASGI server untuk menjalankan FastAPI |
| **TensorFlow** | Inferensi model (format SavedModel) |
| **tf-keras** | Kompatibilitas loading model Keras dengan TensorFlow |
| **Transformers (HuggingFace)** | Arsitektur & tokenizer mBERT untuk klasifikasi teks emosi |
| **Multilingual BERT (mBERT)** | Model bahasa pra-latih berbasis Transformer untuk pemrosesan teks multibahasa |
| **OpenCV** | Deteksi wajah & preprocessing gambar |
| **NumPy** | Manipulasi array |
| **Google Gemini API** | Generative AI untuk menghasilkan insight emosi dari hasil prediksi teks |
| **Python Dotenv** | Manajemen environment variable & API key |
| **Docker** | Containerisasi API NLP untuk deployment di Hugging Face Spaces |
| **Hugging Face Spaces** | Platform hosting API NLP berbasis cloud |
| **PyTorch** | Konversi weight mBERT dari format PyTorch ke TensorFlow saat load model |

---

## Memulai

### Prasyarat

- Python 3.10 atau lebih tinggi
- Package manager `pip`
- (Disarankan) Virtual environment seperti `venv` atau `conda`

### Instalasi

1. **Clone repositori**

   ```bash
   git clone https://github.com/EmoVision-Capstone-Project/emovision-ai-api.git
   cd emovision-ai-api
   ```

2. **Buat dan aktifkan virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux / macOS
   venv\Scripts\activate           # Windows
   ```

3. **Install dependensi**

   ```bash
   pip install -r requirements.txt
   ```

4. **Pastikan model tersedia**

   Model SavedModel yang telah dilatih harus berada di lokasi berikut:

   ```
   emotion_savedmodel/
   ├── saved_model.pb
   ├── fingerprint.pb
   └── variables/
       ├── variables.data-00000-of-00001
       └── variables.index
   ```

### Menjalankan Server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

API akan tersedia di `http://localhost:8000`.

Dokumentasi API interaktif (Swagger UI) dapat diakses di `http://localhost:8000/docs`.

---

## Referensi API

### `GET /`

Endpoint health check untuk memastikan layanan berjalan dengan baik.

**Respons**

```json
{
  "message": "API Emotion Detection jalan"
}
```

---

### `POST /predict`

Menerima file gambar dan mengembalikan prediksi emosi beserta skor kepercayaan.

**Request**

| Parameter | Tipe | Wajib | Deskripsi |
|-----------|------|-------|-----------|
| `file` | `multipart/form-data` | ✅ | File gambar (JPEG, PNG, dll.) |

**Contoh — cURL**

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/gambar.jpg"
```

**Contoh — Python**

```python
import requests

url = "http://localhost:8000/predict"
with open("gambar.jpg", "rb") as f:
    response = requests.post(url, files={"file": f})

print(response.json())
```

**Respons — Berhasil (`200 OK`)**

```json
{
  "emotion": "Happy",
  "confidence": 94.37,
  "probabilities": {
    "Angry": 0.21,
    "Disgust": 0.05,
    "Fear": 0.13,
    "Happy": 94.37,
    "Neutral": 4.82,
    "Sad": 0.31,
    "Surprise": 0.11
  }
}
```

**Respons — Gagal (gambar tidak valid)**

```json
{
  "error": "File gambar tidak valid"
}
```

**Keterangan Field Respons**

| Field | Tipe | Deskripsi |
|-------|------|-----------|
| `emotion` | `string` | Label emosi yang diprediksi |
| `confidence` | `float` | Skor kepercayaan prediksi teratas (0–100) |
| `probabilities` | `object` | Persentase probabilitas untuk setiap 7 kelas emosi |

---

## Struktur Proyek

```
emovision-ai-api/
├── app.py                      # Aplikasi FastAPI & definisi endpoint
├── requirements.txt            # Daftar dependensi Python
├── README.md                   # Dokumentasi proyek
└── emotion_savedmodel/         # Model TensorFlow yang telah dilatih
    ├── saved_model.pb
    ├── fingerprint.pb
    └── variables/
        ├── variables.data-00000-of-00001
        └── variables.index
```
