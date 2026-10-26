# Runtime Architecture

## Current flow

```text
Browser camera preview
        |
        v
web/index.html + app.js
        |
        +--> POST /api/frame {image_base64}
        |       (MediaPipe hand/pose)
        |                              |
        +--> POST /api/infer {features, mask, timestamp}
        |       or /api/infer?sample_id  v
                               RecognitionAdapter
                                             |
                                             v
                                      RecognitionRuntime
                                             |
                                             v
                                      Orchestrator
                                      /          \
                         low confidence       accepted meaning
                              |                       |
                       clarification          ConversationAgent
                                                     |
                                                     v
                                             GlossClipResolver
                                                     |
                                             pending://avatar-assets
                                                     
Browser failure table <--- GET /api/failures <--- append-only JSONL log
```

The local dashboard runs with `python web/server.py 8000`. It uses the real checkpoint for processed `.npz` samples and exposes the same `gloss`, `confidence`, and `timestamp` contract as the integration adapter.

## Boundaries

- Landmark preprocessing remains Python-side and accepts `(T, 258)` features plus `(T, 75)` masks.
- `python web/server.py 8000` runs the camera-capable server with the repository's MediaPipe and OpenCV dependencies.
- `POST /api/frame` converts one browser JPEG into the existing normalized 258-feature and 75-mask schema using the same extractor functions as dataset preprocessing, and marks frames without detections as `has_landmarks: false`.
- The browser ignores landmark-free frames, accumulates 24 valid frame responses, and sends them to `POST /api/infer`, which validates the adapter shapes before running inference.
- Confidence below the orchestrator threshold produces a clarification response.
- Unknown meanings are logged as `unknown_gloss` and do not enter the conversation response path.
- Avatar resolution is wired through `GlossClipResolver`, but the manifest has no licensed clips. The resolver therefore returns an explicit pending fallback token rather than a fake asset path.
- Evaluation reports preserve per-case outcomes so model errors, low confidence, meaning gaps, and avatar mapping gaps remain distinguishable.
- The orchestrator retains structured `failure_events` for `low_confidence` and `unknown_gloss` outcomes and can forward them through an injected `failure_logger` callback.
- Failure events are appended to a configurable relative JSONL path and exposed in the browser's recent-failures table.
- `scripts/run_demo_evidence.py` captures a reproducible three-sample evidence run; a human-recorded backup video remains a separate presentation task.
- The HTTP service can bind to a configured host and port. Docker Compose exposes port 8000, persists the failure log, and checks `/api/health`.

## Next integration boundary

The current browser camera path and recognition API are wired, but the avatar still has no licensed assets or renderer. Add only assets with verified redistribution rights, then connect the manifest to a web-compatible glTF runtime. An LLM-backed responder, natural-language/text-to-sign input, signer-independent data evaluation, and a recorded human demo also remain future work.
