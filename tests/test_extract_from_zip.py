import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.extract_from_zip import process_zip, scan_zip, split_class_rows


class FakeLandmarker:
    def close(self):
        pass


class FakeCapture:
    def get(self, _):
        return 25.0

    def release(self):
        pass


class FakeExtractor:
    cv2 = SimpleNamespace(VideoCapture=lambda _: FakeCapture(), CAP_PROP_FPS=1)
    calls = []

    @classmethod
    def build_landmarkers(cls, *_):
        return FakeLandmarker(), FakeLandmarker()

    @classmethod
    def extract_video(cls, video_path, *_args, **_kwargs):
        cls.calls.append(Path(video_path).read_bytes())
        features = np.ones((4, 258), dtype=np.float32)
        mask = np.ones((4, 75), dtype=np.float32)
        return features, mask


class PartialFailureExtractor(FakeExtractor):
    @classmethod
    def extract_video(cls, video_path, *args, **kwargs):
        if Path(video_path).read_bytes().startswith(b"b"):
            raise RuntimeError("synthetic_decode_failure")
        return super().extract_video(video_path, *args, **kwargs)


class ExtractFromZipTests(unittest.TestCase):
    def make_zip(self, path):
        with zipfile.ZipFile(path, "w") as archive:
            for index in range(5):
                archive.writestr(f"INCLUDE/Adjectives/class-a/video-{index}.mp4", f"a{index}")
            for index in range(2):
                archive.writestr(f"INCLUDE/Adjectives/class-b/video-{index}.mp4", f"b{index}")
            archive.writestr("__MACOSX/._junk.mp4", "ignored")
            archive.writestr("readme.txt", "ignored")

    def test_scan_lists_videos_without_extracting(self):
        with tempfile.TemporaryDirectory() as directory:
            zip_path = Path(directory) / "input.zip"
            self.make_zip(zip_path)
            videos, counts = scan_zip(zip_path)
        self.assertEqual(counts, {"class-a": 5, "class-b": 2})
        self.assertEqual(len(videos), 7)

    def test_split_assigns_all_splits_for_included_classes(self):
        rows = [{"sample_id": str(index), "label": "class-a"} for index in range(15)]
        split_class_rows(rows, seed=42)
        self.assertEqual({row["split"] for row in rows}, {"train", "val", "test"})

    def test_extraction_resumes_and_reports_excluded_class(self):
        FakeExtractor.calls = []
        with tempfile.TemporaryDirectory() as directory:
            zip_path = Path(directory) / "input.zip"
            output_root = Path(directory) / "output"
            self.make_zip(zip_path)

            result = process_zip(
                zip_path,
                output_root,
                "hand.task",
                "pose.task",
                min_samples=5,
                flush_every=2,
                extractor=FakeExtractor,
            )
            self.assertEqual(len(FakeExtractor.calls), 7)
            first_run_calls = len(FakeExtractor.calls)
            process_zip(
                zip_path,
                output_root,
                "hand.task",
                "pose.task",
                min_samples=5,
                flush_every=2,
                extractor=FakeExtractor,
            )
            self.assertEqual(len(FakeExtractor.calls), first_run_calls)
            progress = [json.loads(line) for line in (output_root / "progress.jsonl").read_text().splitlines()]
            self.assertEqual(len(progress), 14)
            self.assertTrue(all(event["status"] == "skipped_existing_output" for event in progress[7:]))

            report = json.loads(result["quality_report"].read_text())
            self.assertEqual(report["class_counts"], {"class-a": 5, "class-b": 2})
            self.assertEqual(report["archive_video_counts"], {"class-a": 5, "class-b": 2})
            self.assertEqual(report["failed_or_unprocessed_counts"], {"class-a": 0, "class-b": 0})
            self.assertTrue(report["extraction_complete"])
            self.assertEqual(report["excluded_classes"]["class-b"]["valid_samples"], 2)
            self.assertFalse(report["signer_independent_split_verified"])
            self.assertTrue(result["training_metadata"].is_file())
            self.assertTrue((output_root / "progress.jsonl").is_file())
            self.assertEqual(list((output_root / "landmarks").glob("*.npz")).__len__(), 7)
            self.assertFalse(list(Path(tempfile.gettempdir()).glob("isl_video_*")))

    def test_quality_report_explains_class_with_failed_videos(self):
        with tempfile.TemporaryDirectory() as directory:
            zip_path = Path(directory) / "input.zip"
            output_root = Path(directory) / "output"
            self.make_zip(zip_path)
            result = process_zip(
                zip_path,
                output_root,
                "hand.task",
                "pose.task",
                min_samples=5,
                flush_every=3,
                extractor=PartialFailureExtractor,
            )
            report = json.loads(result["quality_report"].read_text())
            self.assertEqual(report["class_counts"]["class-b"], 0)
            self.assertEqual(report["failed_or_unprocessed_counts"]["class-b"], 2)
            self.assertIn("below minimum threshold", report["excluded_classes"]["class-b"]["reason"])


if __name__ == "__main__":
    unittest.main()
