import unittest

from src.lead_pipeline import Orchestrator, RecognitionRuntime


class FakeRecognizer:
    def __init__(self, gloss: str, confidence: float) -> None:
        self.gloss = gloss
        self.confidence = confidence

    def predict(self, features, mask, timestamp):
        return {
            "gloss": self.gloss,
            "confidence": self.confidence,
            "timestamp": timestamp,
        }


class LeadPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.orchestrator = Orchestrator()

    def test_high_confidence_prediction_reaches_response(self) -> None:
        result = self.orchestrator.route(
            {"gloss": "HELLO", "confidence": 0.92, "timestamp": 1.5}
        )

        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["meaning"]["intent"], "greeting")
        self.assertIn("Hello", result["response_text"])

    def test_low_confidence_prediction_requests_clarification(self) -> None:
        result = self.orchestrator.route(
            {"gloss": "hello", "confidence": 0.42, "timestamp": 2.0}
        )

        self.assertEqual(result["status"], "clarification_required")
        self.assertEqual(result["reason"], "low_confidence")

    def test_unknown_gloss_requests_clarification(self) -> None:
        result = self.orchestrator.route(
            {"gloss": "unseen", "confidence": 0.95, "timestamp": 3.0}
        )

        self.assertEqual(result["status"], "clarification_required")
        self.assertEqual(result["reason"], "unknown_gloss")

    def test_invalid_prediction_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.orchestrator.route(
                {"gloss": "hello", "confidence": 1.2, "timestamp": 0}
            )

    def test_runtime_routes_recognizer_output_to_response(self) -> None:
        runtime = RecognitionRuntime(FakeRecognizer("hello", 0.91))

        result = runtime.process("features", "mask", 4.5)

        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["meaning"]["source_gloss"], "hello")
        self.assertEqual(result["timestamp"], 4.5)

    def test_runtime_preserves_recognizer_fallback(self) -> None:
        runtime = RecognitionRuntime(FakeRecognizer("hello", 0.41))

        result = runtime.process("features", "mask", 5.0)

        self.assertEqual(result["status"], "clarification_required")
        self.assertEqual(result["reason"], "low_confidence")


if __name__ == "__main__":
    unittest.main()