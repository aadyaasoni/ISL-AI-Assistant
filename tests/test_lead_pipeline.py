import unittest

from src.lead_pipeline import (
    ConversationAgent,
    EvaluationCase,
    GlossClipResolver,
    Orchestrator,
    RecognitionRuntime,
    evaluate_cases,
)


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

    def test_conversation_agent_keeps_bounded_state(self) -> None:
        agent = ConversationAgent()
        meaning = {"intent": "greeting", "entities": {}, "slots": {}}

        for _ in range(8):
            agent.respond(meaning)

        self.assertEqual(len(agent.state.turns), 6)

    def test_conversation_agent_uses_constrained_responder(self) -> None:
        seen_turns = []

        def responder(meaning, turns):
            seen_turns.append(len(turns))
            return "Constrained response"

        agent = ConversationAgent(response_generator=responder)
        meaning = {"intent": "greeting", "entities": {}, "slots": {}}

        self.assertEqual(agent.respond(meaning), "Constrained response")
        self.assertEqual(agent.respond(meaning), "Constrained response")
        self.assertEqual(seen_turns, [0, 1])

    def test_conversation_agent_rejects_empty_responder_output(self) -> None:
        agent = ConversationAgent(response_generator=lambda meaning, turns: " ")

        with self.assertRaises(ValueError):
            agent.respond({"intent": "greeting", "entities": {}, "slots": {}})

    def test_avatar_resolver_caches_repeated_glosses(self) -> None:
        loaded = []

        def load_clip(path):
            loaded.append(path)
            return f"loaded:{path}"

        resolver = GlossClipResolver(
            {"hello": "clips/hello.glb", "dry": "clips/dry.glb"},
            load_clip,
        )

        result = resolver.resolve(["HELLO", "dry", "hello"])

        self.assertEqual(result, [
            "loaded:clips/hello.glb",
            "loaded:clips/dry.glb",
            "loaded:clips/hello.glb",
        ])
        self.assertEqual(loaded, ["clips/hello.glb", "clips/dry.glb"])
        self.assertEqual(resolver.cache_size, 2)

    def test_avatar_resolver_rejects_unmapped_gloss(self) -> None:
        resolver = GlossClipResolver({}, lambda path: path)

        with self.assertRaises(KeyError):
            resolver.resolve(["unknown"])

    def test_evaluation_reports_meaning_preservation(self) -> None:
        cases = [
            EvaluationCase(
                "greeting-1",
                {"gloss": "hello", "confidence": 0.95, "timestamp": 1.0},
                "greeting",
            ),
            EvaluationCase(
                "unknown-1",
                {"gloss": "unknown", "confidence": 0.95, "timestamp": 2.0},
                "greeting",
            ),
        ]

        report = evaluate_cases(cases, self.orchestrator.route)

        self.assertEqual(report["total"], 2)
        self.assertEqual(report["passed"], 1)
        self.assertEqual(report["failure_categories"], {"meaning_layer_ambiguity": 1})

    def test_evaluation_classifies_low_confidence_as_misrecognition(self) -> None:
        case = EvaluationCase(
            "uncertain-1",
            {"gloss": "hello", "confidence": 0.2, "timestamp": 3.0},
            "greeting",
        )

        report = evaluate_cases([case], self.orchestrator.route)

        self.assertEqual(report["failure_categories"], {"misrecognition": 1})


if __name__ == "__main__":
    unittest.main()