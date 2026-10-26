import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from scripts.landmark_preprocessing import sequence_from_json
from src.lead_pipeline import Orchestrator, OrchestratorConfig
from tests.test_landmark_preprocessing import example_frame
from web import server as web_server


class FakeAdapter:
    def __init__(self, confidence=0.82):
        self.confidence = confidence

    def predict_with_candidates(self, features, mask):
        assert features.shape == (56, 258)
        assert mask.shape == (56, 75)
        prediction = {
            "gloss": "3. happy",
            "confidence": self.confidence,
            "timestamp": 123.0,
        }
        return prediction, [
            {"gloss": "3. happy", "confidence": self.confidence},
            {"gloss": "94. good", "confidence": (1.0 - self.confidence) * 0.5},
        ]


@pytest.fixture
def api(monkeypatch, tmp_path):
    failure_log = tmp_path / "failures.jsonl"
    orchestrator = Orchestrator(
        config=OrchestratorConfig(failure_log_path=failure_log)
    )
    adapter = FakeAdapter()
    monkeypatch.setattr(
        web_server,
        "get_runtime",
        lambda: SimpleNamespace(orchestrator=orchestrator),
    )
    monkeypatch.setattr(web_server, "get_adapter", lambda: adapter)
    monkeypatch.setattr(web_server, "apply_llm_response", lambda route, _ip: route)
    monkeypatch.setattr(
        web_server,
        "api_request_limiter",
        web_server.RequestLimiter(limit=60, window_seconds=60),
    )
    http_server = ThreadingHTTPServer(("127.0.0.1", 0), web_server.AppHandler)
    thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    thread.start()
    yield (
        f"http://127.0.0.1:{http_server.server_port}",
        failure_log,
        orchestrator,
    )
    http_server.shutdown()
    thread.join()
    http_server.server_close()


def json_post(url, path, payload):
    request = urllib.request.Request(
        url + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.load(response)


def test_infer_landmarks_returns_sentence_candidates_and_route(api):
    url, _, _ = api
    frame = example_frame()
    frames, _ = sequence_from_json([frame])
    payload = {
        "frames": [frame] * 56,
        "context_glosses": ["1. loud"],
    }

    result = json_post(url, "/api/infer_landmarks", payload)

    assert result["decision"] == "confident"
    assert result["gloss"] == "3. happy"
    assert len(result["candidates"]) == 2
    assert result["route"]["status"] == "accepted"
    assert result["sentence"] == "The recognized words are loud and happy."
    assert result["reply_glosses"] == ["1. loud", "3. happy"]
    assert result["avatar"]["status"] == "ready"
    assert frames.shape == (1, 258)


def test_low_confidence_returns_clarify_candidates_and_selection_is_logged(api):
    url, failure_log, orchestrator = api
    web_server.get_adapter().confidence = 0.42
    payload = {"frames": [example_frame()] * 56}

    result = json_post(url, "/api/infer_landmarks", payload)

    assert result["decision"] == "clarify"
    assert result["route"]["status"] == "clarification_required"
    assert len(result["candidates"]) == 2
    selected = json_post(
        url,
        "/api/clarification",
        {
            "gloss": result["candidates"][0]["gloss"],
            "confidence": result["confidence"],
            "context_glosses": [],
        },
    )

    event = json.loads(failure_log.read_text(encoding="utf-8").splitlines()[-1])
    assert selected["status"] == "logged"
    assert event == {
        "reason": "user_clarification",
        "gloss": "3. happy",
        "confidence": 0.42,
    }
    assert all("landmarks" not in key for key in event)
    assert len(orchestrator.failure_events) == 2


def test_compose_sentence_accepts_recognized_sequence_and_rejects_non_string(api):
    url, _, _ = api
    result = json_post(
        url,
        "/api/compose_sentence",
        {"glosses": ["1. loud", "3. happy"]},
    )

    assert result["sentence"] == "The recognized words are loud and happy."
    assert result["reply_glosses"] == ["1. loud", "3. happy"]

    request = urllib.request.Request(
        url + "/api/compose_sentence",
        data=json.dumps({"glosses": [3]}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request, timeout=5)
    assert error.value.code == 400


def test_landmark_endpoint_rejects_body_above_cap(api):
    url, _, _ = api
    body = b" " * (web_server.MAX_UPLOAD_BYTES + 1)
    request = urllib.request.Request(
        url + "/api/infer_landmarks",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request, timeout=5)

    assert error.value.code == 413


@pytest.mark.parametrize(
    "model_name",
    ["hand_landmarker.task", "pose_landmarker_lite.task"],
)
def test_mediapipe_models_are_served_from_repository(api, model_name):
    url, _, _ = api

    with urllib.request.urlopen(
        f"{url}/models/mediapipe/{model_name}",
        timeout=5,
    ) as response:
        assert response.status == 200
        assert response.headers["Content-Length"] != "0"
