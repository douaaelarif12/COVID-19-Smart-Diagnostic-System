"""Chargement et validation de la configuration associée au modèle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Calcule l'empreinte SHA-256 d'un fichier sans le charger en mémoire."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate_model_configuration(
    model: Any,
    model_path: Path,
    class_names: list[str],
    metadata_path: Path,
) -> dict[str, Any]:
    """Valide le modèle chargé contre les métadonnées versionnées.

    Les divergences mesurables empêchent le démarrage. Le statut de vérification
    scientifique reste distinct : il n'est vrai que si les artefacts
    d'entraînement confirment les classes et le prétraitement.
    """
    if not metadata_path.is_file():
        raise FileNotFoundError(
            "Métadonnées absentes : model/model_metadata.json est requis."
        )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    expected_filename = metadata.get("model_file")
    if expected_filename != model_path.name:
        errors.append(
            f"fichier attendu {expected_filename!r}, fichier chargé {model_path.name!r}"
        )

    actual_hash = sha256_file(model_path)
    expected_hash = str(metadata.get("sha256", "")).lower()
    if actual_hash != expected_hash:
        errors.append("l'empreinte SHA-256 du modèle ne correspond pas aux métadonnées")

    expected_input = metadata.get("input", {}).get("shape")
    actual_input = list(model.input_shape)
    if expected_input != actual_input:
        errors.append(
            f"forme d'entrée attendue {expected_input}, forme chargée {actual_input}"
        )

    expected_output_count = metadata.get("output", {}).get("count")
    actual_output_count = int(model.output_shape[-1])
    if expected_output_count != actual_output_count:
        errors.append(
            f"nombre de sorties attendu {expected_output_count}, obtenu {actual_output_count}"
        )

    expected_activation = metadata.get("output", {}).get("activation")
    actual_activation = getattr(model.layers[-1].activation, "__name__", None)
    if expected_activation != actual_activation:
        errors.append(
            f"activation de sortie attendue {expected_activation!r}, obtenue {actual_activation!r}"
        )

    expected_classes = metadata.get("output", {}).get("class_names")
    if expected_classes != class_names:
        errors.append(
            "model/class_names.txt ne correspond pas à l'ordre déclaré dans les métadonnées"
        )

    if errors:
        raise ValueError("Configuration du modèle invalide : " + "; ".join(errors) + ".")

    verification = metadata.get("verification", {})
    configuration_verified = bool(
        verification.get("class_mapping_from_training")
        and verification.get("preprocessing_from_training")
    )

    return {
        "metadata": metadata,
        "model_sha256": actual_hash,
        "configuration_verified": configuration_verified,
        "warnings": [str(item) for item in metadata.get("warnings", [])],
    }
