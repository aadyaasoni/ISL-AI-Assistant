import json
import unittest
from pathlib import Path

import numpy as np

from scripts.recognition_adapter import RecognitionAdapter


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "training_outputs/bilstm_baseline/best_model.pt"
LABEL_MAP = ROOT / "training_outputs/bilstm_baseline/label_mapping.json"


def adapter():
    return RecognitionAdapter(CHECKPOINT, LABEL_MAP, device="cpu")


class RecognitionAdapterTests(unittest.TestCase):
    def test_contract_fields(self):
        result = adapter().predict(np.zeros((4, 258), dtype=np.float32), np.ones((4, 75), dtype=np.float32), 123.0)
        self.assertEqual(set(result), {"gloss", "confidence", "timestamp"})


    def test_timestamp_is_preserved(self):
        result = adapter().predict(np.zeros((4, 258)), np.ones((4, 75)), 123.0)
        self.assertEqual(result["timestamp"], 123.0)


    def test_gloss_is_in_label_map(self):
        labels = json.loads(LABEL_MAP.read_text())
        self.assertIn(adapter().predict(np.zeros((4, 258)), np.ones((4, 75)), 123.0)["gloss"], labels)


    def test_invalid_feature_dimension_rejected(self):
        with self.assertRaisesRegex(ValueError, "features"):
            adapter().predict(np.zeros((4, 257)), np.ones((4, 75)), 123.0)


    def test_invalid_mask_shape_rejected(self):
        with self.assertRaisesRegex(ValueError, "mask"):
            adapter().predict(np.zeros((4, 258)), np.ones((4, 74)), 123.0)


    def test_confidence_is_probability(self):
        result = adapter().predict(np.zeros((4, 258)), np.ones((4, 75)), 123.0)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()