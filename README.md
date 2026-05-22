# EmoVision AI API

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=for-the-badge&logo=fastapi&logoColor=white"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white"/>
  <img src="https://img.shields.io/badge/Transformers-4.44-F9AB00?style=for-the-badge&logo=huggingface&logoColor=white"/>
  <img src="https://img.shields.io/badge/mBERT-Multilingual-4B0082?style=for-the-badge&logo=bert&logoColor=white"/>
  <img src="https://img.shields.io/badge/Gemini-AI-4285F4?style=for-the-badge&logo=google&logoColor=white"/>
  <img src="https://img.shields.io/badge/Hugging%20Face-Spaces-F9AB00?style=for-the-badge&logo=huggingface&logoColor=white"/>
  <img src="https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white"/>
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white"/>
  <img src="https://img.shields.io/badge/Lisensi-MIT-green?style=for-the-badge"/>
</p>

<p align="center">
  REST API Multimodal berperforma tinggi untuk deteksi emosi secara real-time, mengombinasikan <b>Analisis Ekspresi Wajah</b> (Computer Vision via FastAPI & TensorFlow) dan <b>Analisis Teks Jurnal</b> (NLP via mBERT di Hugging Face). Mampu mengenali 7 kelas emosi secara holistik beserta skor kepercayaan dan rincian distribusi probabilitas lengkap.
</p>

---

## Daftar Isi

- [Gambaran Umum](#gambaran-umum)
- [Arsitektur Multimodal](#arsitektur-multimodal)
- [Fitur](#fitur)
- [Kelas Emosi](#kelas-emosi)
- [Teknologi yang Digunakan](#teknologi-yang-digunakan)
- [Memulai dan Instalasi](#memulai-dan-instalasi)
  - [Prasyarat Umum](#prasyarat-umum)
  - [1. Setup API Deteksi Wajah (Lokal)](#1-setup-api-deteksi-wajah-lokal)
  - [2. Setup API Deteksi Teks NLP (Lokal / Hugging Face)](#2-setup-api-deteksi-teks-nlp-lokal--hugging-face)
- [Referensi API](#referensi-api)
  - [1. API Deteksi Wajah (Image)](#1-api-deteksi-wajah-image)
  - [2. API Deteksi Teks Jurnal (NLP)](#2-api-deteksi-teks-jurnal-nlp)
- [Struktur Proyek](#struktur-proyek)
  - [1. Struktur API Deteksi Wajah (Lokal)](#1-struktur-api-deteksi-wajah-lokal)
  - [2. Struktur API Deteksi Teks NLP (HuggingFace)](#2-struktur-api-deteksi-teks-nlp-huggingface)
---

## Gambaran Umum

EmoVision AI adalah inti machine learning dari **EmoVision Capstone Project** yang mengusung pendekatan multimodal (Teks & Gambar). Sistem ini mengintegrasikan dua layanan utama: API analisis teks yang memproses curhatan atau jurnal pengguna menggunakan arsitektur Transformer mBERT via Hugging Face Spaces, serta API deteksi ekspresi wajah yang menerima file gambar, mendeteksi dan memotong wajah secara otomatis menggunakan Haar Cascade OpenCV, lalu menjalankan inferensi melalui model TensorFlow (SavedModel) lokal. Kedua subsistem ini menyelaraskan hasil untuk mengembalikan prediksi 7 kelas emosi secara holistik beserta skor kepercayaan dan distribusi probabilitas yang lengkap.

---

## Arsitektur Multimodal

Proyek ini terbagi menjadi dua *service* utama:
* **Service A (Image - Local/Server):** Menerima file gambar, mendeteksi wajah menggunakan Haar Cascade OpenCV, lalu menjalankan inferensi melalui TensorFlow SavedModel.
* **Service B (Text - Cloud/Hugging Face):** Di-*deploy* di platform Hugging Face (`tasyacac05-emovision.hf.space`). Menerima *input* teks dan mengklasifikasikan sentimen emosional tulisan tersebut menggunakan arsitektur Transformer (mBERT).

---

## Fitur

- **Deteksi Wajah Otomatis** — Menggunakan Haar Cascade untuk menemukan dan memotong wajah terbesar dalam gambar; jika tidak ada wajah yang terdeteksi, dilakukan center crop secara otomatis.
- **Analisis Sentimen Jurnal** — Memahami konteks bahasa (NLP) dari tulisan curhatan/jurnal pengguna menggunakan mBERT.
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

## Memulai dan Instalasi

### Prasyarat Umum

- Python 3.10 atau lebih tinggi
- Package manager `pip`
- (Disarankan) Virtual environment seperti `venv` atau `conda`

### 1. Setup API Deteksi Wajah (Lokal)

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

### 2. Setup API Deteksi Teks NLP (Lokal / Hugging Face)

Jika ingin melakukan pengembangan atau menjalankan API model mBERT (Teks) secara lokal sebelum didorong ke Hugging Face Spaces:

1. **Clone/Masuk ke direktori proyek NLP**
```bash
   cd emovision-nlp-api
```

2. **Buat dan aktifkan virtual environment (Python 3.11)**
```bash
   py -3.11 -m venv venv311
   venv311\Scripts\activate        # Windows
   source venv311/bin/activate     # Linux / macOS
```

3. **Install dependensi NLP**
```bash
   pip install -r requirements.txt
```

4. **Buat file `.env` dan isi Gemini API Key**
```env
   GEMINI_API_KEY=isi_api_key_kamu
```

5. **Jalankan Server API Teks Lokal**
```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
   *Swagger UI lokal dapat diakses di `http://localhost:8000/docs`.*

---

### Deploy ke Hugging Face Spaces

API Deteksi Teks di-*hosting* di Hugging Face Spaces menggunakan Docker. Untuk memperbarui deployment:

1. **Pastikan sudah login dan punya Access Token** di [huggingface.co](https://huggingface.co) dengan role **Write**

2. **Push perubahan ke Space**
```bash
   git remote set-url origin https://USERNAME:TOKEN@huggingface.co/spaces/TasyaCAC05/EmoVision
   git add .
   git commit -m "update"
   git push
```

3. **Akses API cloud setelah build selesai**
   https://tasyacac05-emovision.hf.space/docs

> **Catatan:** Free tier Hugging Face Spaces akan otomatis *sleep* setelah tidak ada aktivitas. Request pertama setelah *sleep* membutuhkan waktu 30 detik untuk *wake up*.

> **Gemini API Key** disimpan sebagai *Secret* di Settings Space, bukan di dalam kode atau repo.
---

## Referensi API

### 1. API Deteksi Wajah (Image)
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

| Parameter | Tipe |  Deskripsi |
|-----------|------|-----------|
| `file` | `multipart/form-data` | File gambar (JPEG, PNG, dll.) |

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

### 2. API Deteksi Teks Jurnal (NLP)

API berbasis cloud yang menangani input berupa string teks curhatan/jurnal pengguna.

- **Base URL Cloud:** `https://tasyacac05-emovision.hf.space`
- **Dokumentasi Swagger:** `https://tasyacac05-emovision.hf.space/docs`

---

#### `GET /`
Health check untuk memastikan API berjalan.

**Contoh Respons:**
```json
{
  "message": "EmoVision NLP API is running",
  "status": "ok"
}
```
---

#### `GET /health`
Mengecek status model yang sedang berjalan.

**Contoh Respons:**
```json
{
  "status": "healthy",
  "model": "emovision_nlp_savedmodel"
}
```
---

#### `POST /predict`
Menerima input teks jurnal dan mengembalikan hasil klasifikasi emosi beserta insight opsional dari Gemini AI.

**Request Body:**
| Field | Tipe | Deskripsi |
|-------|------|-------|-----------|
| `text` | `string` | Teks jurnal/curhatan pengguna |
| `with_insight` | `boolean` | Jika `true`, mengembalikan insight empati dari Gemini AI (default: `false`) |

**Contoh Request (Python):**
```python
import requests

url = "https://tasyacac05-emovision.hf.space/predict"

# Tanpa insight
payload = {
    "text": "Hari ini aku senang sekali karena modelku berhasil di-deploy!",
    "with_insight": False
}
response = requests.post(url, json=payload)
print(response.json())

# Dengan insight Gemini
payload_with_insight = {
    "text": "Aku merasa sangat cemas dengan presentasi besok.",
    "with_insight": True
}
response = requests.post(url, json=payload_with_insight)
print(response.json())
```

**Contoh Respons (tanpa insight):**
```json
{
  "text": "Hari ini aku senang sekali karena modelku berhasil di-deploy!",
  "predicted_label": "happy",
  "confidence": 0.9812,
  "all_scores": {
    "angry": 0.0001,
    "disgust": 0.0000,
    "fear": 0.0002,
    "happy": 0.9812,
    "neutral": 0.0150,
    "sad": 0.0030,
    "surprise": 0.0005
  },
  "insight": null
}
```

**Contoh Respons (dengan insight):**
```json
{
  "text": "Aku merasa sangat cemas dengan presentasi besok.",
  "predicted_label": "fear",
  "confidence": 0.8743,
  "all_scores": {
    "angry": 0.0021,
    "disgust": 0.0005,
    "fear": 0.8743,
    "happy": 0.0012,
    "neutral": 0.0634,
    "sad": 0.0573,
    "surprise": 0.0012
  },
  "insight": "Wajar sekali merasa cemas sebelum presentasi, perasaanmu sangat valid. Coba tarik napas dalam dan ingat bahwa kamu sudah mempersiapkan ini dengan baik. Kamu pasti bisa!"
}
```

**Kode Error:**
| Kode | Deskripsi |
|------|-----------|
| `400` | Text tidak boleh kosong |
| `422` | Format request body tidak valid |
| `500` | Internal server error |

---

## Struktur Proyek

### 1. Struktur API Deteksi Wajah (Lokal)

```text
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

### 2. Struktur API Deteksi Teks NLP (HuggingFace)

```text
emovision-nlp-api/
├── main.py                         # Aplikasi FastAPI & definisi endpoint NLP
├── requirements.txt                # Daftar dependensi Python
├── Dockerfile                      # Konfigurasi Docker untuk Hugging Face Spaces
├── .env                            # Gemini API Key (tidak di-commit ke Git)
├── .gitignore                      # Mengecualikan venv, __pycache__, .env
└── model/
    ├── emovision_nlp_savedmodel/   # Model TensorFlow mBERT yang telah dilatih
    │   ├── saved_model.pb
    │   ├── fingerprint.pb
    │   └── variables/
    │       ├── variables.data-00000-of-00001
    │       └── variables.index
    ├── tokenizer/                  # mBERT Tokenizer
    │   ├── tokenizer.json
    │   ├── tokenizer_config.json
    │   ├── special_tokens_map.json
    │   └── vocab.txt
    └── model_metadata.json         # Metadata label & konfigurasi model
```
