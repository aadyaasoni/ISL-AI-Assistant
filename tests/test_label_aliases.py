import unittest

from scripts.build_label_aliases import build_aliases


class LabelAliasTests(unittest.TestCase):
    def test_aliases_are_derived_from_labels_and_collisions_preserved(self):
        result = build_aliases({"83. big large": 0, "84. small large": 1})
        self.assertIn("big large", result["canonical_to_aliases"]["83. big large"])
        self.assertEqual(result["alias_to_canonical"]["large"], ["83. big large", "84. small large"])
        self.assertEqual(result["ambiguous_aliases"]["large"], ["83. big large", "84. small large"])
        self.assertIn("no semantic synonyms", result["alias_policy"])

    def test_label_order_follows_mapping_index(self):
        result = build_aliases({"second": 1, "first": 0})
        self.assertEqual(list(result["canonical_to_aliases"]), ["first", "second"])


if __name__ == "__main__":
    unittest.main()
