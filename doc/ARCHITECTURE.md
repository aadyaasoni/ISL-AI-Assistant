# Runtime Architecture

## Current flow

```text
Browser camera + MediaPipe Tasks Vision (pinned, loaded on demand)
        |
        v
web/index.html + app.js
        |
        +--> browser segments hand motion and sends landmark JSON
        |       POST /api/infer_landmarks
        |                         |
        +--> recorded sample ----> RecognitionAdapter
                                  (CPU BiLSTM)
                                        |
                                        v
                                  Orchestrator
                                  /          \
                     low confidence       accepted gloss sequence
                          |                        |
                  candidate choice          ConversationAgent
                          |                        |
                          +--> clarification       v
                             event log       deterministic sentence
                                                 |
                                                 v
                                      canvas skeleton clip playback

Browser failure table <--- GET /api/failures <--- append-only JSONL log

Hearing-user text box ---> POST /api/text_to_sign ---> glosses + unmatched words
                                                     |
                                                     +--> cached canvas clips
```

The local dashboard runs with `python web/server.py 8000`. Live MediaPipe detection, segmentation, and overlay drawing run in the browser; the server receives only landmark coordinates and applies the existing model to them. The browser resamples segments to 56 frames, a representative size rather than a fixed model requirement.

## Boundaries

- Browser hand and pose task files are served from `models/mediapipe/`; the pinned Tasks Vision JavaScript and WASM runtime are loaded from jsDelivr. Camera mirroring is display-only; handedness is retained as returned by MediaPipe.
- `POST /api/infer_landmarks` converts JSON frames through the same pure-NumPy normalization and layout used by `scripts/extract_landmarks.py`, then passes `(T, 258)` features and `(T, 75)` masks to the CPU recognizer.
- The browser starts a segment when visible hands move, ends it after 600 ms of no movement or no hand, ignores very short segments, and resamples accepted segments to 56 frames.
- The server returns top-two candidates and a `confident` or `clarify` routing decision. User candidate selection is recorded by `POST /api/clarification`; accepted glosses build a deterministic sentence through `POST /api/compose_sentence`.
- `POST /api/infer` and `GET /api/infer?sample_id=...` retain their recorded-feature and sample recognition flows. The legacy image-based `POST /api/frame` is deprecated (410); camera processing no longer runs in the server.
- `POST /api/text_to_sign` tokenizes text deterministically against the supported dataset gloss labels and label synonyms, returning canonical glosses plus unmatched words. No LLM is used on this path.
- Confidence below the orchestrator threshold produces a clarification response.
- Unknown meanings are logged as `unknown_gloss` and do not enter the conversation response path.
- Accepted semantic frames may be sent to an optional urllib-based OpenAI-compatible responder. It is disabled without `ISL_LLM_API_KEY`; any failure falls back to the deterministic response.
- The browser loads generated landmark skeleton clips from `avatar/clips/`, caches them by filename, and draws pose and hand landmarks on a canvas. This is not a rigged 3D avatar; dataset redistribution terms remain unverified.
- Evaluation reports preserve per-case outcomes so model errors, low confidence, meaning gaps, and avatar mapping gaps remain distinguishable.
- The orchestrator retains structured `failure_events` for `low_confidence` and `unknown_gloss` outcomes and can forward them through an injected `failure_logger` callback.
- Failure events are appended to a configurable JSONL path and expose only gloss, confidence, and fallback reason; raw video is not stored or logged.
- `/api/infer`, `/api/infer_landmarks`, `/api/clarification`, `/api/compose_sentence`, and `/api/text_to_sign` are rate-limited per client IP, and POST bodies are capped at 2 MiB.
- `scripts/run_demo_evidence.py` captures a reproducible three-sample evidence run; a human-recorded backup video remains a separate presentation task.
- The HTTP service can bind to a configured host and port. Docker Compose exposes the configured app port and checks `/api/health`.

## Next integration boundary

Verify dataset redistribution terms before distributing the generated skeleton clips. A rigged 3D avatar, richer language translation, signer-independent evaluation, and a recorded human demo remain future work.
