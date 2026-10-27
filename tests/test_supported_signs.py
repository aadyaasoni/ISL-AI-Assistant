import json
import unittest

from web.server import LABELS, LABEL_MAP_PATH


class SupportedSignsTests(unittest.TestCase):
    def test_supported_sign_vocabulary_comes_from_selected_label_mapping(self):
        mapping = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
        expected = [label for label, _ in sorted(mapping.items(), key=lambda item: item[1])]
        self.assertEqual(LABELS, expected)


if __name__ == "__main__":
    unittest.main()
