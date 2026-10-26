import unittest

import numpy as np

from scripts.landmark_preprocessing import sequence_from_json


def coordinates(count, x, y, z, visibility=None):
    points = []
    for index in range(count):
        item = {
            "x": x + index * 0.001,
            "y": y + index * 0.001,
            "z": z + index * 0.001,
        }
        if visibility is not None:
            item["visibility"] = visibility
        points.append(item)
    return points


def example_frame():
    pose = coordinates(33, 0.5, 0.3, 0.0, visibility=0.9)
    pose[11] = {"x": 0.4, "y": 0.5, "z": 0.0, "visibility": 1.0}
    pose[12] = {"x": 0.6, "y": 0.5, "z": 0.0, "visibility": 1.0}
    return {
        "hands": [
            {"handedness": "Right", "landmarks": coordinates(21, 0.8, 0.3, 0.0)},
            {"handedness": "Left", "landmarks": coordinates(21, 0.2, 0.3, 0.0)},
        ],
        "pose_landmarks": pose,
    }


class LandmarkPreprocessingTests(unittest.TestCase):
    def test_json_sequence_matches_258_feature_and_75_mask_training_schema(self):
        features, masks = sequence_from_json([example_frame()])

        self.assertEqual(features.shape, (1, 258))
        self.assertEqual(masks.shape, (1, 75))
        self.assertEqual(features.dtype, np.float32)
        self.assertEqual(masks.dtype, np.float32)
        np.testing.assert_allclose(features[0, :3], [-1.5, -1.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(features[0, 63:66], [1.5, -1.0, 0.0], atol=1e-6)
        self.assertTrue(np.all(masks[0, :42] == 1))
        self.assertTrue(np.all(masks[0, 42:] == 1))

    def test_right_only_hand_uses_right_hand_slots_and_mask(self):
        frame = example_frame()
        frame["hands"] = [frame["hands"][0]]
        features, masks = sequence_from_json([frame])

        self.assertTrue(np.all(features[0, :63] == 0))
        self.assertTrue(np.all(masks[0, :21] == 0))
        self.assertTrue(np.all(masks[0, 21:42] == 1))
        self.assertNotEqual(float(features[0, 63]), 0.0)

    def test_rejects_frames_without_a_detected_hand(self):
        frame = example_frame()
        frame["hands"] = []

        with self.assertRaisesRegex(ValueError, "at least one detected hand"):
            sequence_from_json([frame])

    def test_rejects_non_finite_coordinates(self):
        frame = example_frame()
        frame["hands"][0]["landmarks"][0]["x"] = float("nan")

        with self.assertRaisesRegex(ValueError, "must be finite"):
            sequence_from_json([frame])


if __name__ == "__main__":
    unittest.main()
