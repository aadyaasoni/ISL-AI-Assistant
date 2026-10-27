import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.recognition_adapter import RecognitionAdapter, resolve_checkpoint_paths


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "training_outputs/bilstm_baseline/best_model.pt"
LABEL_MAP = ROOT / "training_outputs/bilstm_baseline/label_mapping.json"


def adapter():
    return RecognitionAdapter(CHECKPOINT, LABEL_MAP, device="cpu")


class RecognitionAdapterTests(unittest.TestCase):
    def test_legacy_checkpoint_remains_default(self):
        with patch.dict(os.environ, {}, clear=True):
            checkpoint, label_map = resolve_checkpoint_paths()
        self.assertEqual(checkpoint, CHECKPOINT)
        self.assertEqual(label_map, LABEL_MAP)

    def test_checkpoint_directory_environment_selects_checkpoint_and_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {"ISL_CHECKPOINT_DIR": directory}):
                checkpoint, label_map = resolve_checkpoint_paths()
        self.assertEqual(checkpoint, Path(directory) / "best_model.pt")
        self.assertEqual(label_map, Path(directory) / "label_mapping.json")

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

    def test_predict_with_candidates_returns_top_two_probabilities(self):
        prediction, candidates = adapter().predict_with_candidates(
            np.zeros((4, 258), dtype=np.float32),
            np.ones((4, 75), dtype=np.float32),
            123.0,
        )

        self.assertEqual(prediction["gloss"], candidates[0]["gloss"])
        self.assertEqual(prediction["confidence"], candidates[0]["confidence"])
        self.assertEqual(len(candidates), 2)
        self.assertGreaterEqual(candidates[0]["confidence"], candidates[1]["confidence"])
        self.assertLessEqual(sum(item["confidence"] for item in candidates), 1.0)


if __name__ == "__main__":
    unittest.main()