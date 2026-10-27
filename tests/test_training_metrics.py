import unittest

import numpy as np

from scripts.train_bilstm import calibration_metrics


class TrainingMetricsTests(unittest.TestCase):
    def test_calibration_metrics_match_known_probabilities(self):
        result = calibration_metrics(
            [0, 1],
            np.asarray([[0.9, 0.1], [0.2, 0.8]], dtype=np.float64),
            bins=10,
        )
        self.assertAlmostEqual(result["ece"], 0.15)
        self.assertAlmostEqual(result["multiclass_brier_score"], 0.05)
        self.assertEqual(result["method"], "top-label ECE with equal-width bins")

    def test_calibration_rejects_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "sample_count"):
            calibration_metrics([0], np.asarray([[0.4, 0.6], [0.2, 0.8]]))


if __name__ == "__main__":
    unittest.main()
