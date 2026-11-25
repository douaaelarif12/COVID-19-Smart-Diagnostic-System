from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from keras.models import load_model
from pathlib import Path

from inference import InvalidImageError, check_image_contrast, decode_image, predict_image
from model_config import load_and_validate_model_configuration

# Variables globales
model = None
class_names = []
model_configuration = None

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
        model = None
        class_names = []
        model_configuration = None
        raise RuntimeError(f"Impossible de charger le modèle : {e}") from e


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_ai_model()
    yield


app = FastAPI(
    title="API de classification thoracique expérimentale",
    description=(
        "API académique multiclasse. Elle ne constitue pas un dispositif médical "
        "et ne fournit pas de diagnostic."
    ),
    version="1.1.0",
    lifespan=lifespan,
)

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

    accepted_content_types = {"image/jpeg", "image/png"}
    if file.content_type not in accepted_content_types:
        raise HTTPException(status_code=415, detail="Format requis : JPEG ou PNG.")

    try:
        contents = await file.read(10 * 1024 * 1024 + 1)
        image = decode_image(contents)
        contrast_valid, contrast_value, contrast_message = check_image_contrast(image)
        if not contrast_valid:
            raise HTTPException(status_code=422, detail=contrast_message)
        result = predict_image(image, model, class_names)

        return {
            "filename": file.filename,
            "prediction": result.class_name,
            "class_index": result.class_index,
            "model_score": result.score,
            "confidence_score": result.score,
            "confidence_percentage": f"{result.score * 100:.2f}%",
            "scores_by_class": result.scores,
            "input_validation": {
                "format_valid": True,
                "contrast_standard_deviation": contrast_value,
                "chest_xray_verified": False,
                "warning": (
                    "Le serveur valide le fichier et son contraste, mais ne peut pas "
                    "confirmer qu'il s'agit d'une radiographie thoracique."
                ),
            },
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
    except InvalidImageError as e:
        status_code = 413 if "volumineuse" in str(e) else 422
        raise HTTPException(status_code=status_code, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Erreur interne pendant l'analyse.") from e
