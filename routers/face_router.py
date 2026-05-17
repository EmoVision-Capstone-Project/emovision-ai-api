from fastapi import APIRouter, UploadFile, File
import tensorflow as tf
import numpy as np
import cv2

router = APIRouter()

# Load SavedModel 
model = tf.saved_model.load("emotion_savedmodel")
infer = model.signatures["serving_default"]

class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

@router.get("/")
def home():
    return {"message": "API Emotion Detection jalan"}

def crop_face_from_array(img):
    if img is None:
        raise ValueError("Gambar tidak ditemukan atau file tidak valid")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(50, 50)
    )

    if len(faces) == 0:
        h, w = img.shape[:2]
        size = min(h, w)
        start_x = (w - size) // 2
        start_y = (h - size) // 2
        return img[start_y:start_y + size, start_x:start_x + size]

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])

    margin = 0.25
    x1 = max(0, int(x - w * margin))
    y1 = max(0, int(y - h * margin))
    x2 = min(img.shape[1], int(x + w * (1 + margin)))
    y2 = min(img.shape[0], int(y + h * (1 + margin)))

    return img[y1:y2, x1:x2]

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "File gambar tidak valid"}

    face = crop_face_from_array(img)
    face = cv2.resize(face, (224, 224))

    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    img_array = tf.convert_to_tensor(face_rgb, dtype=tf.float32)
    img_array = tf.expand_dims(img_array, axis=0)

    outputs = infer(img_array)
    preds = list(outputs.values())[0].numpy()[0]

    percentages = preds * 100

    idx = int(np.argmax(percentages))
    label = class_names[idx]
    confidence = percentages[idx]

    return {
        "emotion": label,
        "confidence": float(confidence),
        "probabilities": {
            class_names[i]: float(percentages[i])
            for i in range(len(class_names))
        }
    }