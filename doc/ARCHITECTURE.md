# Runtime Architecture

## Current flow

```text
Browser camera preview
        |
        v
web/index.html + app.js
        |
        +--> POST /api/infer {features, mask, timestamp}
        |                              |
        +--> /api/infer?sample_id=...  v
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
- The browser camera currently provides a local preview only. A future camera bridge must convert browser frames to the existing landmark schema before calling the recognizer.
- The server accepts model-ready browser landmarks through `POST /api/infer`; it validates the adapter shapes before running inference. A future camera bridge still must produce those arrays from browser frames.
- Confidence below the orchestrator threshold produces a clarification response.
- Unknown meanings are logged as `unknown_gloss` and do not enter the conversation response path.
- Avatar resolution is wired through `GlossClipResolver`, but the manifest has no licensed clips. The resolver therefore returns an explicit pending fallback token rather than a fake asset path.
- Evaluation reports preserve per-case outcomes so model errors, low confidence, meaning gaps, and avatar mapping gaps remain distinguishable.

## Next integration boundary

Add a licensed avatar package and populate `avatar/asset_manifest.json`. Then replace the pending fallback with a loader for the selected glTF/GLB runtime and add browser-frame landmark extraction that produces the existing model input contract.
