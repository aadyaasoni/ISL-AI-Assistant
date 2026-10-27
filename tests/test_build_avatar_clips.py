import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.build_avatar_clips import build_clips


class BuildAvatarClipsTests(unittest.TestCase):
    def test_builds_a_clip_for_each_mapped_label(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            label_map = root / "label_mapping.json"
            label_map.write_text(json.dumps({"200. greeting": 0, "201. thank you": 1}))
            samples = []
            for index, label in enumerate(("200. greeting", "201. thank you")):
                sample_path = root / f"sample-{index}.npz"
                np.savez_compressed(
                    sample_path,
                    features=np.zeros((2, 258), dtype=np.float32),
                    mask=np.ones((2, 75), dtype=np.float32),
                )
                samples.append({
                    "sample_id": f"sample-{index}",
                    "split": "train",
                    "label": label,
                    "valid": "True",
                    "processed_file_path": str(sample_path),
                })
            metadata = root / "metadata.csv"
            with metadata.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(samples[0]))
                writer.writeheader()
                writer.writerows(samples)
            clips_dir = root / "clips"

            index = build_clips(metadata, label_map, clips_dir)

            self.assertEqual(set(index), {"200. greeting", "201. thank you"})
            self.assertTrue((clips_dir / "index.json").is_file())
            for clip_file in index.values():
                payload = json.loads((clips_dir / clip_file).read_text())
                self.assertEqual(payload["coordinate_space"], "normalized-landmarks")


if __name__ == "__main__":
    unittest.main()
