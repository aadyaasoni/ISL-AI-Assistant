import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.lead_pipeline.text_to_sign import _aliases_for_path, text_to_sign


class DynamicTextToSignTests(unittest.TestCase):
    def test_selected_checkpoint_aliases_are_used_and_ambiguous_terms_are_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            aliases_path = Path(directory) / "synonyms.json"
            aliases_path.write_text(json.dumps({
                "alias_to_canonical": {
                    "greeting sign": ["200. greeting"],
                    "sign": ["200. greeting", "201. other"],
                }
            }), encoding="utf-8")
            _aliases_for_path.cache_clear()
            with patch.dict(os.environ, {"ISL_CHECKPOINT_DIR": directory}):
                result = text_to_sign("greeting sign")
        self.assertEqual(result, {"glosses": ["200. greeting"], "unmatched": []})

    def test_legacy_aliases_remain_the_default(self):
        with patch.dict(os.environ, {}, clear=True):
            result = text_to_sign("loud")
        self.assertEqual(result, {"glosses": ["1. loud"], "unmatched": []})

    def test_all_ambiguous_dynamic_aliases_do_not_fall_back_to_legacy_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            aliases_path = Path(directory) / "synonyms.json"
            aliases_path.write_text(json.dumps({"alias_to_canonical": {"loud": ["200. loud", "201. loud"]}}), encoding="utf-8")
            _aliases_for_path.cache_clear()
            with patch.dict(os.environ, {"ISL_CHECKPOINT_DIR": directory}):
                result = text_to_sign("loud")
        self.assertEqual(result, {"glosses": [], "unmatched": ["loud"]})


if __name__ == "__main__":
    unittest.main()
