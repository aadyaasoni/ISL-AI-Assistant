import csv
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
            return json_response(self, {"status": "ready", "labels": len(adapter.labels)})
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
