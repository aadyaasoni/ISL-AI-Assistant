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
                                             optional LLM responder
                                                     |
                                             deterministic fallback
                                                     |
                                      canvas skeleton clip playback
                                                     
Browser failure table <--- GET /api/failures <--- append-only JSONL log

Hearing-user text box ---> POST /api/text_to_sign ---> glosses + unmatched words
                                                     |
                                                     +--> cached canvas clips
```

The local dashboard runs with `python web/server.py 8000`. It uses the real checkpoint for processed `.npz` samples and exposes the same `gloss`, `confidence`, and `timestamp` contract as the integration adapter.

## Boundaries

- Landmark preprocessing remains Python-side and accepts `(T, 258)` features plus `(T, 75)` masks.
- `python web/server.py 8000` runs the camera-capable server with the repository's MediaPipe and OpenCV dependencies.
- `POST /api/frame` converts one browser JPEG into the existing normalized 258-feature and 75-mask schema using the same extractor functions as dataset preprocessing, and marks frames without detections as `has_landmarks: false`.
- The browser ignores landmark-free frames, accumulates 24 valid frame responses, and sends them to `POST /api/infer`, which validates the adapter shapes before running inference.
- `POST /api/text_to_sign` tokenizes text deterministically against the supported dataset gloss labels and label synonyms, returning canonical glosses plus unmatched words. No LLM is used on this path.
- Confidence below the orchestrator threshold produces a clarification response.
- Unknown meanings are logged as `unknown_gloss` and do not enter the conversation response path.
- Accepted semantic frames may be sent to an optional urllib-based OpenAI-compatible responder. It is disabled without `ISL_LLM_API_KEY`; any failure falls back to the deterministic response.
- The browser loads generated landmark skeleton clips from `avatar/clips/`, caches them by filename, and draws pose and hand landmarks on a canvas. This is not a rigged 3D avatar; dataset redistribution terms remain unverified.
- Evaluation reports preserve per-case outcomes so model errors, low confidence, meaning gaps, and avatar mapping gaps remain distinguishable.
- The orchestrator retains structured `failure_events` for `low_confidence` and `unknown_gloss` outcomes and can forward them through an injected `failure_logger` callback.
- Failure events are appended to a configurable JSONL path and expose only gloss, confidence, and fallback reason; raw video is not stored or logged.
- `/api/infer` and `/api/text_to_sign` are rate-limited per client IP, and POST bodies are capped at 2 MiB.
- `scripts/run_demo_evidence.py` captures a reproducible three-sample evidence run; a human-recorded backup video remains a separate presentation task.
- The HTTP service can bind to a configured host and port. Docker Compose exposes the configured app port and checks `/api/health`.

## Next integration boundary

Verify dataset redistribution terms before distributing the generated skeleton clips. A rigged 3D avatar, richer language translation, signer-independent evaluation, and a recorded human demo remain future work.
