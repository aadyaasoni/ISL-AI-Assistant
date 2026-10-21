import unittest

from src.lead_pipeline import Orchestrator
from src.lead_pipeline.avatar import GlossClipResolver
from src.lead_pipeline.phrase_evaluation import (
    PHRASE_EVALUATION_CASES,
    PhraseEvaluationCase,
    evaluate_phrase_cases,
)


class PhraseEvaluationTests(unittest.TestCase):
    def test_phrase_set_contains_25_cases_with_expected_outputs(self) -> None:
        self.assertEqual(len(PHRASE_EVALUATION_CASES), 25)
        self.assertTrue(
            all(
                case.gloss_sequence
                and case.expected_meaning
                and case.expected_response
                for case in PHRASE_EVALUATION_CASES
            )
        )

    def test_phrase_evaluation_classifies_required_failure_categories(self) -> None:
        cases = [
            PhraseEvaluationCase(
                "misrecognition",
                ("1. loud",),
                ("loud",),
                "I understood the property: loud.",
                (0.2,),
            ),
            PhraseEvaluationCase(
                "meaning-ambiguity",
                ("unseen gloss",),
                ("unmapped",),
                "I do not recognize that sign yet. Please repeat it.",
            ),
            PhraseEvaluationCase(
                "avatar-gap",
                ("1. loud",),
                ("loud",),
                "I understood the property: loud.",
            ),
        ]
        resolver = GlossClipResolver({}, lambda clip_path: clip_path)

        report = evaluate_phrase_cases(cases, Orchestrator(), resolver)

        self.assertEqual(report["total"], 3)
        self.assertEqual(report["failed"], 3)
        self.assertEqual(
            report["failure_categories"],
            {
                "misrecognition": 1,
                "meaning_ambiguity": 1,
                "avatar_mapping_gap": 1,
            },
        )

    def test_confidence_count_must_match_gloss_count(self) -> None:
        case = PhraseEvaluationCase(
            "invalid-confidence-count",
            ("1. loud", "2. quiet"),
            ("loud", "quiet"),
            "I understood the property: loud. I understood the property: quiet.",
            (0.95,),
        )
        resolver = GlossClipResolver({}, lambda clip_path: clip_path)

        with self.assertRaisesRegex(ValueError, "confidence count"):
            evaluate_phrase_cases([case], Orchestrator(), resolver)

    def test_complete_phrase_round_trip_passes_when_clip_is_mapped(self) -> None:
        case = PhraseEvaluationCase(
            "mapped-loud",
            ("1. loud",),
            ("loud",),
            "I understood the property: loud.",
        )
        resolver = GlossClipResolver(
            {"1. loud": "clips/loud.glb"},
            lambda clip_path: clip_path,
        )

        report = evaluate_phrase_cases([case], Orchestrator(), resolver)

        self.assertEqual(report["passed"], 1)
        self.assertTrue(report["cases"][0]["passed"])
        self.assertEqual(report["cases"][0]["actual_meaning"], ["loud"])
        self.assertEqual(
            report["cases"][0]["actual_response"],
            "I understood the property: loud.",
        )


if __name__ == "__main__":
    unittest.main()
