import csv
import base64
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

sys.path.insert(0, str(ROOT))
from src.lead_pipeline import (
    GlossClipResolver,
    Orchestrator,
    OrchestratorConfig,
    RecognitionRuntime,
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
HAND_MODEL_PATH = ROOT / "models/mediapipe/hand_landmarker.task"
POSE_MODEL_PATH = ROOT / "models/mediapipe/pose_landmarker_lite.task"
camera_landmarkers = None
camera_timestamp_ms = 0


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
            return self.infer(parse_qs(parsed.query))
        return self.serve_static(parsed.path)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/frame":
            return self.frame()
        if path == "/api/text_to_sign":
            return self.text_to_sign()
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

    def text_to_sign(self):
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length))
            if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
                raise ValueError("text must be a string")
            return json_response(self, text_to_sign(payload["text"]))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            return json_response(self, {"error": str(error)}, 400)

    def frame(self):
        global camera_landmarkers, camera_timestamp_ms
        try:
            import cv2
            import mediapipe as mp
            from scripts.extract_landmarks import build_landmarkers, frame_features

            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length))
            encoded_image = payload["image_base64"].split(",")[-1]
            image_bytes = base64.b64decode(encoded_image, validate=True)
            image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("image_base64 is not a decodable image")
            if camera_landmarkers is None:
                camera_landmarkers = build_landmarkers(HAND_MODEL_PATH, POSE_MODEL_PATH)
            rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            camera_timestamp_ms = max(camera_timestamp_ms + 1, int(float(payload.get("timestamp_ms", 0))))
            hand_result = camera_landmarkers[0].detect_for_video(mp_image, camera_timestamp_ms)
            pose_result = camera_landmarkers[1].detect_for_video(mp_image, camera_timestamp_ms)
            features, mask = frame_features(hand_result, pose_result)
            return json_response(
                self,
                {
                    "features": features.tolist(),
                    "mask": mask.tolist(),
                    "has_landmarks": bool(np.any(mask)),
                    "timestamp_ms": camera_timestamp_ms,
                },
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, base64.binascii.Error, ImportError) as error:
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
