import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from model_config import load_and_validate_model_configuration


def softmax(value):
    return value


class FakeLayer:
    activation = softmax


class FakeModel:
    input_shape = (None, 224, 224, 3)
    output_shape = (None, 3)
    layers = [FakeLayer()]


class ModelConfigurationTests(unittest.TestCase):
    def test_validates_hash_shape_and_classes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            model_path = directory / "model.h5"
            model_path.write_bytes(b"model fixture")
            metadata_path = directory / "metadata.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "model_file": "model.h5",
                        "sha256": hashlib.sha256(b"model fixture").hexdigest(),
                        "input": {"shape": [None, 224, 224, 3]},
                        "output": {
                            "count": 3,
                            "activation": "softmax",
                            "class_names": ["A", "B", "C"],
                        },
                        "verification": {
                            "class_mapping_from_training": False,
                            "preprocessing_from_training": False,
                        },
                        "warnings": ["fixture"],
                    }
                ),
                encoding="utf-8",
            )

            result = load_and_validate_model_configuration(
                FakeModel(), model_path, ["A", "B", "C"], metadata_path
            )
            self.assertFalse(result["configuration_verified"])
            self.assertEqual(result["warnings"], ["fixture"])

            with self.assertRaises(ValueError):
                load_and_validate_model_configuration(
                    FakeModel(), model_path, ["C", "B", "A"], metadata_path
                )


if __name__ == "__main__":
    unittest.main()
