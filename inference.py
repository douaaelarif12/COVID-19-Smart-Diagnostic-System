"""Pipeline d'inférence partagé entre Streamlit, FastAPI et l'évaluation."""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

INPUT_SIZE = (224, 224)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 25_000_000
MIN_IMAGE_SIDE = 64
SUPPORTED_FORMATS = {"JPEG", "PNG"}


class InvalidImageError(ValueError):
    """Image absente, corrompue ou incompatible avec le pipeline."""


@dataclass(frozen=True)
class PredictionResult:
    class_index: int
    class_name: str
    score: float
    scores: dict[str, float]


def decode_image(contents: bytes) -> Image.Image:
    """Valide les octets reçus puis retourne une image RGB orientée correctement."""
    if not contents:
        raise InvalidImageError("Le fichier image est vide.")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise InvalidImageError("Image trop volumineuse (maximum 10 Mio).")

    try:
        with Image.open(BytesIO(contents)) as probe:
            image_format = probe.format
            width, height = probe.size
            probe.verify()

        if image_format not in SUPPORTED_FORMATS:
            raise InvalidImageError("Format non pris en charge. Utilisez JPEG ou PNG.")
        if width * height > MAX_IMAGE_PIXELS:
            raise InvalidImageError("Résolution trop élevée (maximum 25 mégapixels).")
        if min(width, height) < MIN_IMAGE_SIDE:
            raise InvalidImageError("Résolution insuffisante (minimum 64 pixels par côté).")

        with Image.open(BytesIO(contents)) as source:
            return ImageOps.exif_transpose(source).convert("RGB")
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("Le fichier n'est pas une image JPEG/PNG valide.") from exc


def prepare_image(image: Image.Image) -> np.ndarray:
    """Applique le prétraitement historique déclaré pour le modèle."""
    rgb_image = ImageOps.exif_transpose(image).convert("RGB")
    resized = ImageOps.fit(rgb_image, INPUT_SIZE, Image.Resampling.LANCZOS)
    normalized = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)


def check_image_contrast(image: Image.Image) -> tuple[bool, float, str]:
    """Contrôle technique de contraste, sans validation du contenu médical."""
    grayscale = ImageOps.grayscale(image)
    standard_deviation = float(np.asarray(grayscale, dtype=np.float32).std())
    if standard_deviation < 10.0:
        return (
            False,
            standard_deviation,
            "L'image est trop uniforme pour une analyse fiable.",
        )
    return (
        True,
        standard_deviation,
        "Contraste suffisant pour le traitement. Le contenu médical n'est pas vérifié.",
    )


def predict_image(
    image: Image.Image,
    model: Any,
    class_names: list[str],
) -> PredictionResult:
    """Exécute une prédiction et contrôle la cohérence de la sortie."""
    raw_prediction = np.asarray(model.predict(prepare_image(image), verbose=0))
    if raw_prediction.shape != (1, len(class_names)):
        raise ValueError(
            "Sortie du modèle invalide : "
            f"forme attendue (1, {len(class_names)}), obtenue {raw_prediction.shape}."
        )
    if not np.all(np.isfinite(raw_prediction)):
        raise ValueError("La sortie du modèle contient une valeur non finie.")

    values = raw_prediction[0].astype(float)
    if np.any(values < 0.0) or np.any(values > 1.0) or not np.isclose(
        values.sum(), 1.0, atol=1e-3
    ):
        raise ValueError("La sortie Softmax du modèle est invalide.")
    index = int(np.argmax(values))
    return PredictionResult(
        class_index=index,
        class_name=class_names[index],
        score=float(values[index]),
        scores={name: float(values[i]) for i, name in enumerate(class_names)},
    )


def load_image_file(path: Path) -> Image.Image:
    """Charge une image locale avec les mêmes contrôles que les uploads."""
    return decode_image(path.read_bytes())
