from fastapi import FastAPI, File, UploadFile, HTTPException
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import io
import tensorflow as tf
from pathlib import Path

app = FastAPI(
    title="API Détection COVID-19",
    description="API Binaire : COVID-19 vs Autres",
    version="1.0"
)

# Variables globales
model = None
class_names = []

@app.on_event("startup")
def load_ai_model():
    global model, class_names
    model_path = Path('./model/covid_detector.h5')
    labels_path = Path('./model/class_names.txt')
    
    try:
        print("🔄 Chargement du modèle...")
        if not model_path.is_file() or not labels_path.is_file():
            raise FileNotFoundError("Modèle ou fichier model/class_names.txt absent.")
        model = load_model(model_path)
        class_names = [line.strip() for line in labels_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        num_classes = int(model.output_shape[-1])
        if len(class_names) != num_classes:
            raise ValueError("Le nombre d'étiquettes ne correspond pas aux sorties du modèle.")
            
        print(f"✅ Configuration terminée. Classes actives : {class_names}")
        
    except Exception as e:
        print(f"❌ Erreur critique : {e}")

def preprocess_image(image: Image.Image):
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    normalized_image_array = (img_array.astype(np.float32) / 255.0)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    return data

@app.get("/")
def home():
    return {"status": "online", "message": "API COVID-19 prête."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé.")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Fichier image requis.")

    try:
        contents = await file.read(10 * 1024 * 1024 + 1)
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image trop volumineuse (maximum 10 Mio).")
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        processed_data = preprocess_image(image)
        prediction = model.predict(processed_data)
        index = np.argmax(prediction)
        
        confidence = float(prediction[0][index])
        
        # Récupération du nom de la classe
        if index < len(class_names):
            raw_class_name = class_names[index]
        else:
            raw_class_name = "Inconnu"

        return {
            "filename": file.filename,
            "prediction": raw_class_name,
            "confidence_score": confidence,
            "confidence_percentage": f"{confidence*100:.2f}%",
            "medical_disclaimer": "Prototype académique non validé pour le diagnostic clinique."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur analyse : {str(e)}")
