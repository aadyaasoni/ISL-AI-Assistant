import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.lead_pipeline import (
    ConversationAgent,
    CommunicationPipeline,
    EvaluationCase,
    GlossClipResolver,
    Orchestrator,
    OrchestratorConfig,
    RecognitionRuntime,
    evaluate_cases,
    meaning_for_gloss,
    read_failure_events,
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


class FakeClipResolver:
    def __init__(self, clips=None):
        self.clips = clips or {}

    def resolve(self, gloss_sequence):
        return [self.clips[gloss] for gloss in gloss_sequence]


class LeadPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.orchestrator = self.make_orchestrator()

    def make_orchestrator(self, **kwargs) -> Orchestrator:
        directory = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(directory.cleanup)
        relative_path = Path(directory.name).relative_to(Path.cwd()) / "events.jsonl"
        config = kwargs.pop(
            "config",
            OrchestratorConfig(failure_log_path=relative_path),
        )
        return Orchestrator(config=config, **kwargs)

    def test_high_confidence_prediction_reaches_response(self) -> None:
        result = self.orchestrator.route(
            {"gloss": "HELLO", "confidence": 0.92, "timestamp": 1.5}
        )

        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["meaning"]["intent"], "greeting")
        self.assertIn("Hello", result["response_text"])

    def test_meaning_for_loud_gloss(self) -> None:
        self.assertEqual(meaning_for_gloss("1. loud")["slots"]["property"], "loud")

    def test_meaning_for_quiet_gloss(self) -> None:
        self.assertEqual(meaning_for_gloss("2. quiet")["slots"]["property"], "quiet")

    def test_meaning_for_happy_gloss(self) -> None:
        self.assertEqual(meaning_for_gloss("3. happy")["slots"]["property"], "happy")

    def test_meaning_for_long_gloss(self) -> None:
        self.assertEqual(meaning_for_gloss("78. long")["slots"]["property"], "long")

    def test_meaning_for_short_gloss(self) -> None:
        self.assertEqual(meaning_for_gloss("79. short")["slots"]["property"], "short")

    def test_meaning_for_big_large_gloss(self) -> None:
        self.assertEqual(meaning_for_gloss("83. big large")["slots"]["property"], "large")

    def test_meaning_for_small_little_gloss(self) -> None:
        self.assertEqual(meaning_for_gloss("84. small little")["slots"]["property"], "small")

    def test_meaning_for_hot_gloss(self) -> None:
        self.assertEqual(meaning_for_gloss("87. hot")["slots"]["property"], "hot")

    def test_meaning_for_new_gloss(self) -> None:
        self.assertEqual(meaning_for_gloss("91. new")["slots"]["property"], "new")

    def test_meaning_for_good_gloss(self) -> None:
        self.assertEqual(meaning_for_gloss("94. good")["slots"]["property"], "good")

    def test_meaning_for_dry_gloss(self) -> None:
        self.assertEqual(meaning_for_gloss("97. dry")["slots"]["property"], "dry")

    def test_unknown_gloss_is_explicitly_unmapped(self) -> None:
        meaning = meaning_for_gloss("unseen")

        self.assertEqual(meaning["intent"], "unknown")
        self.assertEqual(meaning["mapping_status"], "unmapped")

    def test_orchestrator_uses_injected_conversation_agent(self) -> None:
        agent = ConversationAgent(
            response_generator=lambda meaning, turns: "Injected response"
        )
        orchestrator = self.make_orchestrator(conversation_agent=agent)

        result = orchestrator.route(
            {"gloss": "hello", "confidence": 0.92, "timestamp": 1.0}
        )

        self.assertEqual(result["response_text"], "Injected response")
        self.assertEqual(len(agent.state.turns), 1)

    def test_low_confidence_prediction_requests_clarification(self) -> None:
        result = self.orchestrator.route(
            {"gloss": "hello", "confidence": 0.42, "timestamp": 2.0}
        )

        self.assertEqual(result["status"], "clarification_required")
        self.assertEqual(result["reason"], "low_confidence")

    def test_low_confidence_failure_is_logged(self) -> None:
        logged = []
        orchestrator = self.make_orchestrator(failure_logger=logged.append)

        orchestrator.route({"gloss": "hello", "confidence": 0.42, "timestamp": 2.0})

        self.assertEqual(orchestrator.failure_events, logged)
        self.assertEqual(logged[0]["reason"], "low_confidence")

    def test_failures_append_to_configured_jsonl_and_keep_callback(self) -> None:
        callbacks = []
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            relative_path = Path(directory).relative_to(Path.cwd()) / "events.jsonl"
            config = OrchestratorConfig(failure_log_path=relative_path)
            orchestrator = Orchestrator(
                failure_logger=callbacks.append,
                config=config,
            )

            orchestrator.route(
                {"gloss": "hello", "confidence": 0.42, "timestamp": 2.0}
            )
            orchestrator.route(
                {"gloss": "unseen", "confidence": 0.95, "timestamp": 3.0}
            )
            Orchestrator(config=config).route(
                {"gloss": "unseen", "confidence": 0.95, "timestamp": 4.0}
            )

            events = [
                json.loads(line)
                for line in (Path(directory) / "events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        self.assertEqual(len(events), 3)
        self.assertEqual(events[0]["reason"], "low_confidence")
        self.assertEqual(events[1]["reason"], "unknown_gloss")
        self.assertEqual(events[2]["reason"], "unknown_gloss")
        self.assertEqual(callbacks, events[:2])

    def test_failure_log_config_reads_relative_environment_path(self) -> None:
        with patch.dict(os.environ, {"ISL_FAILURE_LOG_PATH": "tmp/events.jsonl"}):
            config = OrchestratorConfig()

        self.assertEqual(config.failure_log_path, Path("tmp/events.jsonl"))

    def test_failure_log_config_reads_absolute_container_path(self) -> None:
        with patch.dict(
            os.environ,
            {"ISL_FAILURE_LOG_PATH": str(Path.cwd() / "tmp" / "events.jsonl")},
        ):
            config = OrchestratorConfig()

        self.assertEqual(
            config.failure_log_path,
            Path.cwd() / "tmp" / "events.jsonl",
        )

    def test_failure_events_reader_returns_recent_jsonl_records(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            log_path = Path(directory) / "events.jsonl"
            log_path.write_text(
                '\n'.join(
                    json.dumps({"reason": str(index)}) for index in range(3)
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                read_failure_events(log_path, limit=2),
                [{"reason": "1"}, {"reason": "2"}],
            )

    def test_failure_events_reader_treats_missing_log_as_empty(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            self.assertEqual(
                read_failure_events(Path(directory) / "missing.jsonl"),
                [],
            )

    def test_failure_events_reader_rejects_nonpositive_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "limit must be positive"):
            read_failure_events(Path("unused.jsonl"), limit=0)

    def test_unknown_gloss_failure_is_logged(self) -> None:
        orchestrator = self.make_orchestrator()

        orchestrator.route({"gloss": "unseen", "confidence": 0.95, "timestamp": 3.0})

        self.assertEqual(orchestrator.failure_events[0]["reason"], "unknown_gloss")

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

    def test_communication_pipeline_prepares_avatar_clip(self) -> None:
        pipeline = CommunicationPipeline(
            RecognitionRuntime(FakeRecognizer("hello", 0.91)),
            FakeClipResolver({"hello": "hello.glb"}),
        )

        result = pipeline.process("features", "mask", 4.5)

        self.assertEqual(result["avatar_status"], "ready")
        self.assertEqual(result["avatar_clips"], ["hello.glb"])

    def test_communication_pipeline_reports_pending_avatar(self) -> None:
        pipeline = CommunicationPipeline(
            RecognitionRuntime(FakeRecognizer("hello", 0.91)),
            FakeClipResolver(),
        )

        result = pipeline.process("features", "mask", 4.5)

        self.assertEqual(result["avatar_status"], "fallback_required")
        self.assertEqual(result["avatar_clips"], [])

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

    def test_avatar_resolver_uses_configured_fallback(self) -> None:
        loaded = []
        resolver = GlossClipResolver(
            {"hello": "clips/hello.glb"},
            lambda path: loaded.append(path) or f"loaded:{path}",
            fallback_clip="clips/repeat.glb",
        )

        result = resolver.resolve(["unknown", "another-unknown"])

        self.assertEqual(result, [
            "loaded:clips/repeat.glb",
            "loaded:clips/repeat.glb",
        ])
        self.assertEqual(loaded, ["clips/repeat.glb"])

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