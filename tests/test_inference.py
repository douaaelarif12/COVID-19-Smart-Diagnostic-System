import io
import unittest

import numpy as np
from PIL import Image

from inference import (
    InvalidImageError,
    check_image_contrast,
    decode_image,
    predict_image,
    prepare_image,
)


def image_bytes(size=(128, 96), image_format="PNG"):
    buffer = io.BytesIO()
    Image.new("RGB", size, (32, 96, 160)).save(buffer, format=image_format)
    return buffer.getvalue()


class FakeModel:
    def __init__(self, output):
        self.output = np.asarray([output], dtype=np.float32)

    def predict(self, batch, verbose=0):
        self.batch = batch
        return self.output


class InferenceTests(unittest.TestCase):
    def test_decode_and_prepare_image(self):
        image = decode_image(image_bytes())
        batch = prepare_image(image)
        self.assertEqual(image.mode, "RGB")
        self.assertEqual(batch.shape, (1, 224, 224, 3))
        self.assertEqual(batch.dtype, np.float32)
        self.assertGreaterEqual(float(batch.min()), 0.0)
        self.assertLessEqual(float(batch.max()), 1.0)

    def test_rejects_corrupt_and_tiny_images(self):
        with self.assertRaises(InvalidImageError):
            decode_image(b"not an image")
        with self.assertRaises(InvalidImageError):
            decode_image(image_bytes(size=(32, 32)))

    def test_prediction_exposes_all_scores(self):
        model = FakeModel([0.1, 0.7, 0.2])
        result = predict_image(
            decode_image(image_bytes()), model, ["Normal", "Pneumonie", "COVID-19"]
        )
        self.assertEqual(result.class_index, 1)
        self.assertEqual(result.class_name, "Pneumonie")
        self.assertAlmostEqual(result.score, 0.7, places=6)
        self.assertEqual(set(result.scores), {"Normal", "Pneumonie", "COVID-19"})

    def test_contrast_check_does_not_claim_medical_validation(self):
        valid, value, message = check_image_contrast(decode_image(image_bytes()))
        self.assertFalse(valid)
        self.assertGreaterEqual(value, 0.0)
        self.assertNotIn("radiographie valide", message.lower())

    def test_rejects_unexpected_model_output(self):
        with self.assertRaises(ValueError):
            predict_image(
                decode_image(image_bytes()), FakeModel([0.5, 0.5]), ["A", "B", "C"]
            )


if __name__ == "__main__":
    unittest.main()
