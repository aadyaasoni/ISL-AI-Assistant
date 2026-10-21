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
```

The local dashboard runs with `python3 web/server.py 8000`. It uses the real checkpoint for processed `.npz` samples and exposes the same `gloss`, `confidence`, and `timestamp` contract as the integration adapter.

## Boundaries

- Landmark preprocessing remains Python-side and accepts `(T, 258)` features plus `(T, 75)` masks.
- `venv/bin/python web/server.py 8000` runs the camera-capable server with the repository's MediaPipe and OpenCV dependencies.
- `POST /api/frame` converts one browser JPEG into the existing normalized 258-feature and 75-mask schema using the same extractor functions as dataset preprocessing, and marks frames without detections as `has_landmarks: false`.
- The browser ignores landmark-free frames, accumulates 24 valid frame responses, and sends them to `POST /api/infer`, which validates the adapter shapes before running inference.
- Confidence below the orchestrator threshold produces a clarification response.
- Unknown meanings are logged as `unknown_gloss` and do not enter the conversation response path.
- Avatar resolution is wired through `GlossClipResolver`, but the manifest has no licensed clips. The resolver therefore returns an explicit pending fallback token rather than a fake asset path.
- Evaluation reports preserve per-case outcomes so model errors, low confidence, meaning gaps, and avatar mapping gaps remain distinguishable.
- The orchestrator retains structured `failure_events` for `low_confidence` and `unknown_gloss` outcomes and can forward them through an injected `failure_logger` callback.
- `scripts/run_demo_evidence.py` captures a reproducible three-sample evidence run; a human-recorded backup video remains a separate presentation task.

## Next integration boundary

Add a licensed avatar package and populate `avatar/asset_manifest.json`. Then replace the pending fallback with a loader for the selected glTF/GLB runtime and add browser-frame landmark extraction that produces the existing model input contract.
