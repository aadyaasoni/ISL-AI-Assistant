import json
import tempfile
import unittest
from pathlib import Path

from scripts.validate_avatar_assets import validate_manifest


class AvatarAssetValidationTests(unittest.TestCase):
    def write_manifest(self, directory, payload):
        manifest_path = Path(directory) / "asset_manifest.json"
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")
        return manifest_path

    def test_pending_manifest_is_valid_without_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.write_manifest(
                directory,
                {"status": "pending-assets", "model": None, "animations": [], "license": None},
            )
            self.assertEqual(validate_manifest(manifest), [])

    def test_ready_manifest_requires_files_and_license(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = self.write_manifest(
                directory,
                {
                    "status": "ready",
                    "model": "avatar.glb",
                    "animations": [{"name": "hello", "path": "hello.glb"}],
                    "license": None,
                },
            )
            errors = validate_manifest(manifest)
            self.assertIn("missing avatar model: avatar.glb", errors)
            self.assertIn("missing animation clip: hello.glb", errors)
            self.assertIn("ready manifest requires license source, name, and redistribution metadata", errors)

    def test_skeleton_manifest_requires_clip_index_without_claiming_3d_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "clips").mkdir()
            (root / "clips/loud.json").write_text("{}", encoding="utf-8")
            (root / "clips/index.json").write_text(
                '{"gloss_to_clip": {"1. loud": "loud.json"}}', encoding="utf-8"
            )
            manifest = self.write_manifest(
                directory,
                {
                    "status": "skeleton-avatar",
                    "model": None,
                    "animations": [],
                    "clip_index": "clips/index.json",
                    "supported_glosses": ["1. loud"],
                    "license": None,
                },
            )

            self.assertEqual(validate_manifest(manifest), [])


if __name__ == "__main__":
    unittest.main()
