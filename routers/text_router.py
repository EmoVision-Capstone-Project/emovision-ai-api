import os
import json
import zipfile
import numpy as np
import tensorflow as tf
from pathlib import Path
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from transformers import BertTokenizer
from google import genai
from huggingface_hub import snapshot_download

# Config
load_dotenv()

router = APIRouter()

BASE_DIR         = Path(__file__).parent.parent 
SAVEDMODEL_PATH  = BASE_DIR / "model" / "emovision_nlp_savedmodel"
TOKENIZER_ZIP    = BASE_DIR / "model" / "tokenizer_export.zip"
TOKENIZER_DIR    = BASE_DIR / "model" / "tokenizer"
METADATA_PATH    = BASE_DIR / "model" / "model_metadata.json"
MODEL_DIR        = BASE_DIR / "model"
HF_REPO_ID       = "fadidinna/emovision-nlp-model"
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")
MAX_LEN          = 64
NUM_LABELS       = 7

if not os.path.exists(MODEL_DIR / "emovision-nlp-savedmodel"):
    print("Mengunduh model dari Hugging Face... Mohon tunggu.")
    snapshot_download(
        repo_id=HF_REPO_ID, 
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False
    )
    print("Download selesai!")

# Extract tokenizer
if not (TOKENIZER_DIR / "tokenizer_config.json").exists():
    print("Extracting tokenizer...")
    with zipfile.ZipFile(TOKENIZER_ZIP, "r") as z:
        z.extractall(TOKENIZER_DIR)
    print("Tokenizer extracted.")

# Load metadata
with open(METADATA_PATH, "r") as f:
    metadata = json.load(f)

LABEL_MAP = metadata["label_map"]
ID2LABEL  = {int(k): v for k, v in LABEL_MAP.items()}

# Load SavedModel
print("Loading SavedModel...")
loaded_model = tf.saved_model.load(str(SAVEDMODEL_PATH))
infer        = loaded_model.signatures["serving_default"]

# Auto-detect output key dari model
_dummy_ids = tf.zeros((1, MAX_LEN), dtype=tf.int32)
_dummy_out = infer(input_ids=_dummy_ids, attention_mask=_dummy_ids, token_type_ids=_dummy_ids)
OUTPUT_KEY = list(_dummy_out.keys())[0]
print(f"Model loaded. Output key: '{OUTPUT_KEY}'")

# Load Tokenizer
tokenizer = BertTokenizer.from_pretrained(str(TOKENIZER_DIR))
print("Tokenizer loaded. Ready!")

# Gemini setup
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

class PredictRequest(BaseModel):
    text        : str
    with_insight: bool = False

class PredictResponse(BaseModel):
    text            : str
    predicted_label : str
    confidence      : float
    all_scores      : dict
    insight         : str | None = None

# Helper: predict
def predict_emotion(text: str) -> dict:
    encoding = tokenizer(
        text,
        max_length     = MAX_LEN,
        padding        = "max_length",
        truncation     = True,
        return_tensors = "tf"
    )

    output = infer(
        input_ids      = tf.cast(encoding["input_ids"],      tf.int32),
        attention_mask = tf.cast(encoding["attention_mask"], tf.int32),
        token_type_ids = tf.cast(
            encoding.get("token_type_ids", tf.zeros_like(encoding["input_ids"])),
            tf.int32
        )
    )

    logits     = output[OUTPUT_KEY]
    probs      = tf.nn.softmax(logits, axis=-1).numpy()[0]
    pred_idx   = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    all_scores = {ID2LABEL[i]: float(probs[i]) for i in range(len(probs))}

    return {
        "predicted_label": ID2LABEL[pred_idx],
        "confidence"     : confidence,
        "all_scores"     : all_scores
    }

# Helper: Gemini insight
def get_emotion_insight(journal_text: str, emotion_result: dict) -> str:
    if not gemini_client:
        return "[GEMINI_API_KEY tidak ditemukan di file .env]"

    emotion    = emotion_result["predicted_label"]
    confidence = emotion_result["confidence"] * 100
    top3_str   = ", ".join([
        f"{k}: {v*100:.1f}%"
        for k, v in sorted(emotion_result["all_scores"].items(), key=lambda x: -x[1])[:3]
    ])

    prompt = f"""Kamu adalah asisten kesehatan mental yang empatik dan suportif.

Seorang pengguna menulis di jurnal emosi mereka:
\"\"\"{journal_text}\"\"\"

Model AI mendeteksi emosi dominan: **{emotion}** (confidence: {confidence:.1f}%)
Distribusi emosi teratas: {top3_str}

Berikan respons dalam Bahasa Indonesia yang:
1. Validasi perasaan pengguna dengan empati (1 kalimat)
2. Berikan 1 saran konkret & positif untuk membantu mengelola emosi ini
3. Penutup yang menyemangati (1 kalimat)

Gunakan bahasa yang hangat, tidak menghakimi, dan mudah dipahami.
Maksimal 250 karakter."""

    try:
        response = gemini_client.models.generate_content(
            model    = "gemini-2.5-flash-lite",
            contents = prompt
        )
        return response.text
    except Exception as e:
        return f"[Gemini error: {str(e)}]"


# Endpoints 
@router.get("/")
def root():
    return {"message": "EmoVision NLP API is running", "status": "ok"}

@router.get("/health")
def health():
    return {"status": "healthy", "model": "emovision_nlp_savedmodel"}

@router.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text tidak boleh kosong.")
    result  = predict_emotion(request.text)
    insight = get_emotion_insight(request.text, result) if request.with_insight else None
    return PredictResponse(
        text            = request.text,
        predicted_label = result["predicted_label"],
        confidence      = result["confidence"],
        all_scores      = result["all_scores"],
        insight         = insight
    )