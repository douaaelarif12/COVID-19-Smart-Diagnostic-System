from fastapi import FastAPI, File, UploadFile, HTTPException
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import io
import tensorflow as tf
from pathlib import Path

from model_config import load_and_validate_model_configuration

app = FastAPI(
    title="API Détection COVID-19",
    description="API académique de classification thoracique multiclasse",
    version="1.0"
)

# Variables globales
model = None
class_names = []
model_configuration = None

@app.on_event("startup")
def load_ai_model():
    global model, class_names, model_configuration
    model_path = Path('./model/covid_detector.h5')
    labels_path = Path('./model/class_names.txt')
    metadata_path = Path('./model/model_metadata.json')
    
    try:
        print("🔄 Chargement du modèle...")
        if not model_path.is_file() or not labels_path.is_file():
            raise FileNotFoundError("Modèle ou fichier model/class_names.txt absent.")
        loaded_model = load_model(model_path, compile=False)
        loaded_class_names = [line.strip() for line in labels_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        num_classes = int(loaded_model.output_shape[-1])
        if len(loaded_class_names) != num_classes:
            raise ValueError("Le nombre d'étiquettes ne correspond pas aux sorties du modèle.")

        loaded_configuration = load_and_validate_model_configuration(
            loaded_model, model_path, loaded_class_names, metadata_path
        )

        model = loaded_model
        class_names = loaded_class_names
        model_configuration = loaded_configuration
            
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
    return {
        "status": "online",
        "model_loaded": model is not None,
        "message": "API COVID-19 disponible.",
    }

@app.get("/health")
def health():
    if model is None or model_configuration is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé.")
    return {
        "status": "healthy",
        "model_loaded": True,
        "classes": class_names,
        "output_count": int(model.output_shape[-1]),
        "model_sha256": model_configuration["model_sha256"],
        "configuration_verified": model_configuration["configuration_verified"],
        "configuration_warnings": model_configuration["warnings"],
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None or model_configuration is None:
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
            "configuration_verified": model_configuration["configuration_verified"],
            "configuration_warning": (
                None
                if model_configuration["configuration_verified"]
                else "Ordre des classes et prétraitement non confirmés par les artefacts d'entraînement."
            ),
            "medical_disclaimer": "Prototype académique non validé pour le diagnostic clinique."
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur analyse : {str(e)}")
