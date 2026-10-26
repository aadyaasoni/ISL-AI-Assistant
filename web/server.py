import csv
import json
import logging
import mimetypes
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
AVATAR_CLIPS_ROOT = ROOT / "avatar" / "clips"
METADATA_PATH = ROOT / "data/include_metadata/metadata.csv"
CHECKPOINT_PATH = ROOT / "training_outputs/bilstm_baseline/best_model.pt"
LABEL_MAP_PATH = ROOT / "training_outputs/bilstm_baseline/label_mapping.json"
MEDIAPIPE_MODEL_ROOT = ROOT / "models" / "mediapipe"
HAND_MODEL_NAME = "hand_landmarker.task"
POSE_MODEL_NAME = "pose_landmarker_lite.task"

sys.path.insert(0, str(ROOT))
from scripts.landmark_preprocessing import sequence_from_json
from src.lead_pipeline import (
    GlossClipResolver,
    LLMResponder,
    Orchestrator,
    OrchestratorConfig,
    RequestLimiter,
    RecognitionRuntime,
    meaning_for_gloss_sequence,
    text_to_sign,
)
from src.lead_pipeline.failure_log import read_failure_events

with LABEL_MAP_PATH.open(encoding="utf-8") as file:
    LABEL_TO_INDEX = json.load(file)
LABELS = [
    label
    for label, _ in sorted(LABEL_TO_INDEX.items(), key=lambda item: item[1])
]
adapter = None
runtime = None
llm_responder = LLMResponder()
llm_request_limiter = RequestLimiter(limit=20, window_seconds=60)
with (AVATAR_CLIPS_ROOT / "index.json").open(encoding="utf-8") as file:
    AVATAR_CLIPS = json.load(file)["gloss_to_clip"]
avatar_resolver = GlossClipResolver(
    mapping={
        gloss.strip().lower(): str(AVATAR_CLIPS_ROOT / filename)
        for gloss, filename in AVATAR_CLIPS.items()
    },
    clip_loader=lambda clip_path: f"/avatar/clips/{Path(clip_path).name}",
)
with METADATA_PATH.open(newline="", encoding="utf-8") as file:
    SAMPLE_ROWS = [
        row for row in csv.DictReader(file) if row["split"] == "test" and row["valid"] == "True"
    ]
SAMPLES_BY_ID = {row["sample_id"]: row for row in SAMPLE_ROWS}
MAX_UPLOAD_BYTES = 2 * 1024 * 1024
api_request_limiter = RequestLimiter(limit=60, window_seconds=60)


def avatar_playback(glosses):
    clips = []
    unmatched = []
    for gloss in glosses:
        try:
            clips.extend(avatar_resolver.resolve([gloss]))
        except KeyError:
            unmatched.append(gloss)
    return {
        "status": "ready" if clips else "no_clip",
        "glosses": list(glosses),
        "clips": clips,
        "unmatched": unmatched,
    }


def apply_llm_response(routed, client_ip):
    if routed["status"] != "accepted":
        return routed
    llm_allowed = (
        not os.environ.get("ISL_LLM_API_KEY")
        or llm_request_limiter.allow(client_ip)
    )
    routed["response_text"] = llm_responder.respond(
        routed["meaning"],
        routed["response_text"],
        allowed=llm_allowed,
    )
    return routed


def json_response(handler, payload, status=200):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def get_adapter():
    global adapter
    if adapter is None:
        from scripts.recognition_adapter import RecognitionAdapter

        adapter = RecognitionAdapter(CHECKPOINT_PATH, LABEL_MAP_PATH, device="cpu")
    return adapter


def get_runtime():
    global runtime
    if runtime is None:
        runtime = RecognitionRuntime(
            get_adapter(),
            orchestrator=Orchestrator(config=OrchestratorConfig()),
        )
    return runtime


class AppHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            return json_response(self, {"status": "ready", "labels": len(LABELS)})
        if parsed.path == "/api/samples":
            return json_response(
                self,
                {
                    "samples": [
                        {"sample_id": row["sample_id"], "label": row["label"]}
                        for row in SAMPLE_ROWS
                    ]
                },
            )
        if parsed.path == "/api/failures":
            try:
                events = read_failure_events(
                    OrchestratorConfig().failure_log_path
                )
            except (OSError, json.JSONDecodeError):
                logging.exception("Could not read the failure-event log")
                return json_response(
                    self,
                    {"error": "Failure log is unavailable."},
                    503,
                )
            return json_response(self, {"events": events})
        if parsed.path == "/api/infer":
            if not self.check_rate_limit("infer"):
                return
            return self.infer(parse_qs(parsed.query))
        return self.serve_static(parsed.path)

    def do_POST(self):
        path = urlparse(self.path).path
        if not self.check_upload_size():
            return
        if path == "/api/frame":
            return json_response(
                self,
                {"error": "Landmark extraction now runs in the browser."},
                410,
            )
        if path == "/api/text_to_sign":
            if not self.check_rate_limit("text_to_sign"):
                return
            return self.text_to_sign()
        if path in {
            "/api/infer",
            "/api/infer_landmarks",
            "/api/clarification",
            "/api/compose_sentence",
        }:
            if not self.check_rate_limit("infer"):
                return
        if path == "/api/infer_landmarks":
            return self.infer_landmarks()
        if path == "/api/clarification":
            return self.record_clarification()
        if path == "/api/compose_sentence":
            return self.compose_sentence()
        if path != "/api/infer":
            return json_response(self, {"error": "Not found"}, 404)
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length))
            features = np.asarray(payload["features"], dtype=np.float32)
            mask = np.asarray(payload["mask"], dtype=np.float32)
            timestamp = float(payload.get("timestamp", 0.0))
            current_runtime = get_runtime()
            prediction = get_adapter().predict(features, mask, timestamp)
            routed = current_runtime.orchestrator.route(prediction)
            routed = apply_llm_response(routed, self.client_address[0])
            response_glosses = (
                [routed["meaning"]["source_gloss"]]
                if routed["status"] == "accepted"
                else ["please repeat"]
            )
            return json_response(
                self,
                {
                    "prediction": prediction,
                    "route": routed,
                    "avatar": avatar_playback(response_glosses),
                },
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            return json_response(self, {"error": str(error)}, 400)

    def check_upload_size(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            json_response(self, {"error": "Invalid Content-Length"}, 400)
            return False
        if content_length < 0:
            json_response(self, {"error": "Invalid Content-Length"}, 400)
            return False
        if content_length > MAX_UPLOAD_BYTES:
            if content_length <= MAX_UPLOAD_BYTES * 2:
                self.discard_request_body(content_length)
            else:
                self.close_connection = True
            json_response(self, {"error": "Request body is too large"}, 413)
            return False
        self.request_content_length = content_length
        return True

    def discard_request_body(self, content_length):
        remaining = content_length
        while remaining:
            chunk = self.rfile.read(min(65536, remaining))
            if not chunk:
                break
            remaining -= len(chunk)

    def check_rate_limit(self, endpoint):
        client_id = f"{self.client_address[0]}:{endpoint}"
        if not api_request_limiter.allow(client_id):
            if self.command == "POST":
                self.discard_request_body(self.request_content_length)
            json_response(
                self,
                {"error": "Rate limit exceeded; retry shortly."},
                429,
            )
            return False
        return True

    def text_to_sign(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length))
            if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
                raise ValueError("text must be a string")
            return json_response(self, text_to_sign(payload["text"]))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            return json_response(self, {"error": str(error)}, 400)

    def infer_landmarks(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            features, mask = sequence_from_json(payload.get("frames"))
            prediction, candidates = get_adapter().predict_with_candidates(
                features,
                mask,
            )
            orchestrator = get_runtime().orchestrator
            prior_glosses = payload.get("context_glosses", [])
            if not isinstance(prior_glosses, list) or len(prior_glosses) > 15:
                raise ValueError("context_glosses must be an array of at most 15 glosses")
            glosses = [
                value.strip().lower()
                for value in prior_glosses
                if isinstance(value, str)
            ]
            if len(glosses) != len(prior_glosses):
                raise ValueError("context_glosses must contain only strings")
            if glosses:
                meaning_for_gloss_sequence(glosses)
            if prediction["confidence"] < orchestrator.confidence_threshold:
                route = orchestrator.route(prediction)
                return json_response(
                    self,
                    {
                        "gloss": prediction["gloss"],
                        "confidence": prediction["confidence"],
                        "candidates": candidates,
                        "decision": "clarify",
                        "route": route,
                    },
                )

            glosses.append(prediction["gloss"])
            try:
                meaning = meaning_for_gloss_sequence(glosses)
            except ValueError:
                route = orchestrator.route(prediction)
                return json_response(
                    self,
                    {
                        "gloss": prediction["gloss"],
                        "confidence": prediction["confidence"],
                        "candidates": candidates,
                        "decision": "clarify",
                        "route": route,
                    },
                )
            route = {
                "status": "accepted",
                "meaning": meaning,
                "response_text": orchestrator.conversation_agent.respond(meaning),
                "confidence": prediction["confidence"],
                "timestamp": prediction["timestamp"],
            }
            route = apply_llm_response(route, self.client_address[0])
            return json_response(
                self,
                {
                    "gloss": prediction["gloss"],
                    "confidence": prediction["confidence"],
                    "candidates": candidates,
                    "decision": "confident",
                    "route": route,
                    "sentence": route["response_text"],
                    "reply_glosses": glosses,
                    "avatar": avatar_playback(glosses),
                },
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            return json_response(self, {"error": str(error)}, 400)

    def record_clarification(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            gloss = payload.get("gloss")
            confidence = payload.get("confidence")
            if not isinstance(gloss, str):
                raise ValueError("gloss must be a string")
            orchestrator = get_runtime().orchestrator
            prior_glosses = payload.get("context_glosses", [])
            if not isinstance(prior_glosses, list) or len(prior_glosses) > 15:
                raise ValueError("context_glosses must be an array of at most 15 glosses")
            if any(not isinstance(value, str) for value in prior_glosses):
                raise ValueError("context_glosses must contain only strings")
            glosses = [value.strip().lower() for value in prior_glosses]
            glosses.append(gloss.strip().lower())
            meaning = meaning_for_gloss_sequence(glosses)
            orchestrator.record_user_clarification(gloss, confidence)
            response_text = orchestrator.conversation_agent.respond(meaning)
            route = {
                "status": "accepted",
                "meaning": meaning,
                "response_text": response_text,
                "confidence": float(confidence),
            }
            route = apply_llm_response(route, self.client_address[0])
            return json_response(
                self,
                {
                    "status": "logged",
                    "sentence": route["response_text"],
                    "reply_glosses": glosses,
                    "avatar": avatar_playback(glosses),
                },
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            return json_response(self, {"error": str(error)}, 400)

    def compose_sentence(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            glosses = payload.get("glosses")
            if not isinstance(glosses, list) or len(glosses) > 16:
                raise ValueError("glosses must be an array of at most 16 items")
            if any(not isinstance(gloss, str) for gloss in glosses):
                raise ValueError("glosses must contain only strings")
            meaning = meaning_for_gloss_sequence(glosses)
            orchestrator = get_runtime().orchestrator
            response_text = orchestrator.conversation_agent.respond(meaning)
            route = {
                "status": "accepted",
                "meaning": meaning,
                "response_text": response_text,
            }
            route = apply_llm_response(route, self.client_address[0])
            return json_response(
                self,
                {
                    "sentence": route["response_text"],
                    "reply_glosses": meaning["source_glosses"],
                    "avatar": avatar_playback(meaning["source_glosses"]),
                },
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            return json_response(self, {"error": str(error)}, 400)

    def infer(self, query):
        sample_id = query.get("sample_id", [""])[0]
        row = SAMPLES_BY_ID.get(sample_id)
        if row is None:
            return json_response(self, {"error": "Unknown sample_id"}, 404)
        with np.load(ROOT / row["processed_file_path"]) as sample:
            features = sample["features"]
            mask = sample["mask"]
        timestamp = 0.0
        current_runtime = get_runtime()
        prediction = get_adapter().predict(features, mask, timestamp)
        routed = current_runtime.orchestrator.route(prediction)
        routed = apply_llm_response(routed, self.client_address[0])
        response_glosses = (
            [routed["meaning"]["source_gloss"]]
            if routed["status"] == "accepted"
            else ["please repeat"]
        )
        return json_response(
            self,
            {
                "sample_id": sample_id,
                "expected_gloss": row["label"],
                "prediction": prediction,
                "route": routed,
                "avatar": avatar_playback(response_glosses),
            },
        )

    def serve_static(self, path):
        if path.startswith("/models/mediapipe/"):
            model_name = unquote(path.removeprefix("/models/mediapipe/"))
            if model_name not in {HAND_MODEL_NAME, POSE_MODEL_NAME}:
                return json_response(self, {"error": "Not found"}, 404)
            return self.serve_file(MEDIAPIPE_MODEL_ROOT / model_name)
        if path.startswith("/avatar/clips/"):
            relative_clip = unquote(path.removeprefix("/avatar/clips/"))
            candidate = (AVATAR_CLIPS_ROOT / relative_clip).resolve()
            if (
                AVATAR_CLIPS_ROOT not in candidate.parents
                or not candidate.is_file()
            ):
                return json_response(self, {"error": "Not found"}, 404)
            return self.serve_file(candidate)
        relative = "index.html" if path in ("", "/") else unquote(path.lstrip("/"))
        candidate = (WEB_ROOT / relative).resolve()
        if WEB_ROOT not in candidate.parents or not candidate.is_file():
            return json_response(self, {"error": "Not found"}, 404)

        return self.serve_file(candidate)

    def serve_file(self, candidate):
        content_type = mimetypes.guess_type(candidate.name)[0] or "application/octet-stream"
        body = candidate.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else int(os.environ.get("PORT", "7860"))
    host = os.environ.get("HOST", "127.0.0.1")
    server = ThreadingHTTPServer((host, port), AppHandler)
    print(f"ISL Communication Assistant: http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
