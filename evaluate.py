"""Évalue le modèle sur un jeu de test organisé par dossiers de classes."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from inference import InvalidImageError, load_image_file, predict_image
from model_config import load_and_validate_model_configuration


def safe_divide(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def compute_metrics(
    expected: list[int], predicted: list[int], class_names: list[str]
) -> dict[str, Any]:
    """Calcule matrice de confusion et métriques par classe sans dépendance externe."""
    if not expected or len(expected) != len(predicted):
        raise ValueError("Les prédictions et les références doivent être non vides et alignées.")

    count = len(class_names)
    matrix = np.zeros((count, count), dtype=int)
    for actual_index, predicted_index in zip(expected, predicted):
        matrix[actual_index, predicted_index] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    weighted_f1 = 0.0
    for index, name in enumerate(class_names):
        true_positive = int(matrix[index, index])
        support = int(matrix[index, :].sum())
        predicted_count = int(matrix[:, index].sum())
        precision = safe_divide(true_positive, predicted_count)
        recall = safe_divide(true_positive, support)
        f1_score = safe_divide(2 * precision * recall, precision + recall)
        per_class[name] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "support": support,
        }
        f1_values.append(f1_score)
        weighted_f1 += f1_score * support

    total = len(expected)
    return {
        "sample_count": total,
        "accuracy": safe_divide(int(np.trace(matrix)), total),
        "macro_f1": float(np.mean(f1_values)),
        "weighted_f1": weighted_f1 / total,
        "class_names": class_names,
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
    }


def iter_dataset(dataset_path: Path, class_names: list[str]):
    extensions = {".jpg", ".jpeg", ".png"}
    for class_index, class_name in enumerate(class_names):
        class_dir = dataset_path / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Dossier de classe absent : {class_dir}. "
                "Les noms doivent correspondre exactement à model/class_names.txt."
            )
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in extensions:
                yield class_index, class_name, image_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Évalue le modèle sur un jeu de test indépendant organisé par classe."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/evaluation"))
    parser.add_argument("--model", type=Path, default=Path("model/covid_detector.h5"))
    return parser.parse_args()


def main() -> None:
    from keras.models import load_model

    args = parse_args()
    class_names = [
        line.strip()
        for line in Path("model/class_names.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    model = load_model(args.model, compile=False)
    configuration = load_and_validate_model_configuration(
        model,
        args.model,
        class_names,
        Path("model/model_metadata.json"),
    )

    expected: list[int] = []
    predicted: list[int] = []
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for expected_index, expected_name, image_path in iter_dataset(args.dataset, class_names):
        try:
            result = predict_image(load_image_file(image_path), model, class_names)
        except (InvalidImageError, OSError, ValueError) as exc:
            failures.append({"file": str(image_path), "error": str(exc)})
            continue

        expected.append(expected_index)
        predicted.append(result.class_index)
        rows.append(
            {
                "file": str(image_path),
                "expected": expected_name,
                "predicted": result.class_name,
                "model_score": result.score,
                "correct": expected_index == result.class_index,
            }
        )

    metrics = compute_metrics(expected, predicted, class_names)
    report = {
        "dataset_path": str(args.dataset.resolve()),
        "model_sha256": configuration["model_sha256"],
        "configuration_verified": configuration["configuration_verified"],
        "failed_file_count": len(failures),
        "failures": failures,
        "metrics": metrics,
        "medical_disclaimer": "Résultats expérimentaux, sans validation clinique.",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "predictions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
