import unittest

from evaluate import compute_metrics


class EvaluationMetricTests(unittest.TestCase):
    def test_multiclass_metrics(self):
        metrics = compute_metrics(
            expected=[0, 0, 1, 1, 2, 2],
            predicted=[0, 1, 1, 1, 2, 0],
            class_names=["A", "B", "C"],
        )
        self.assertEqual(metrics["sample_count"], 6)
        self.assertEqual(metrics["confusion_matrix"], [[1, 1, 0], [0, 2, 0], [1, 0, 1]])
        self.assertAlmostEqual(metrics["accuracy"], 4 / 6)
        self.assertEqual(metrics["per_class"]["B"]["support"], 2)

    def test_rejects_empty_or_misaligned_values(self):
        with self.assertRaises(ValueError):
            compute_metrics([], [], ["A"])
        with self.assertRaises(ValueError):
            compute_metrics([0], [], ["A"])


if __name__ == "__main__":
    unittest.main()
