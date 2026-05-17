from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.face_router import router as face_router
from routers.text_router import router as text_router

# Inisialisasi API
app = FastAPI(
    title="EmoVision AI API",
    description="API Tunggal untuk Deteksi Emosi Berbasis Wajah dan Teks Jurnal",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoint
app.include_router(face_router, prefix="/api/face", tags=["Face Emotion Detection"])
app.include_router(text_router, prefix="/api/text", tags=["Text Emotion NLP"])

@app.get("/")
def root_api():
    return {"message": "Server EmoVision AI API Aktif. Akses /docs untuk Dokumentasi Swagger."}