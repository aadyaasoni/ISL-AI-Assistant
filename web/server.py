import csv
import base64
import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = ROOT / "web"
METADATA_PATH = ROOT / "data/include_metadata/metadata.csv"
CHECKPOINT_PATH = ROOT / "training_outputs/bilstm_baseline/best_model.pt"
LABEL_MAP_PATH = ROOT / "training_outputs/bilstm_baseline/label_mapping.json"

sys.path.insert(0, str(ROOT))
from scripts.recognition_adapter import RecognitionAdapter
from scripts.extract_landmarks import build_landmarkers, frame_features
from src.lead_pipeline import GlossClipResolver, RecognitionRuntime

adapter = RecognitionAdapter(CHECKPOINT_PATH, LABEL_MAP_PATH, device="cpu")
runtime = RecognitionRuntime(adapter)
avatar_resolver = GlossClipResolver(
    mapping={},
    clip_loader=lambda clip_path: clip_path,
    fallback_clip="pending://avatar-assets",
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


def json_response(handler, payload, status=200):
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class AppHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/health":
            return json_response(
                self,
                {
                    "status": "ready",
                    "labels": len(adapter.labels),
                    "failure_events": len(runtime.orchestrator.failure_events),
                },
            )
        if parsed.path == "/api/failures":
            return json_response(
                self,
                {
                    "count": len(runtime.orchestrator.failure_events),
                    "events": runtime.orchestrator.failure_events,
                },
            )
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
        if parsed.path == "/api/infer":
            return self.infer(parse_qs(parsed.query))
        return self.serve_static(parsed.path)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/frame":
            return self.frame()
        if path != "/api/infer":
            return json_response(self, {"error": "Not found"}, 404)
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(content_length))
            features = np.asarray(payload["features"], dtype=np.float32)
            mask = np.asarray(payload["mask"], dtype=np.float32)
            timestamp = float(payload.get("timestamp", 0.0))
            prediction = adapter.predict(features, mask, timestamp)
            routed = runtime.orchestrator.route(prediction)
            avatar_clip = avatar_resolver.resolve([prediction["gloss"]])[0]
            return json_response(
                self,
                {
                    "prediction": prediction,
                    "route": routed,
                    "avatar": {"clip": avatar_clip, "status": "pending-assets"},
                },
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            return json_response(self, {"error": str(error)}, 400)

    def frame(self):
        global camera_landmarkers, camera_timestamp_ms
        try:
            import cv2
            import mediapipe as mp

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
        prediction = adapter.predict(features, mask, timestamp)
        routed = runtime.orchestrator.route(prediction)
        avatar_clip = avatar_resolver.resolve([prediction["gloss"]])[0]
        return json_response(
            self,
            {
                "sample_id": sample_id,
                "expected_gloss": row["label"],
                "prediction": prediction,
                "route": routed,
                "avatar": {"clip": avatar_clip, "status": "pending-assets"},
            },
        )

    def serve_static(self, path):
        relative = "index.html" if path in ("", "/") else unquote(path.lstrip("/"))
        candidate = (WEB_ROOT / relative).resolve()
        if WEB_ROOT not in candidate.parents or not candidate.is_file():
            return json_response(self, {"error": "Not found"}, 404)
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
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = ThreadingHTTPServer(("127.0.0.1", port), AppHandler)
    print(f"ISL Communication Assistant: http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
