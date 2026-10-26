# ISL Communication Assistant

## Live demo

**Hugging Face Space:** `https://<your-space-name>.hf.space` (placeholder — replace after deployment).

A deployable prototype for a constrained Indian Sign Language (ISL) communication flow:

```text
camera frame -> MediaPipe hand/pose landmarks -> BiLSTM gloss prediction
             -> confidence-aware routing -> semantic frame -> response text
```

The application provides browser-side camera landmarks, motion-based sign segmentation, a recorded-sample fallback, explicit clarification for uncertain glosses, text-to-sign for supported words, a skeleton avatar, and a recent-failure dashboard.

> **Scope:** this is an 11-gloss isolated-sign research/demo prototype, not a general-purpose ISL translator or a production accessibility service. It does not translate continuous signing, arbitrary sentences, or audio.

## What works / what doesn't

| Feature | Status |
|---|---|
| Live camera recognition | Browser-side MediaPipe detection, motion segmentation, and 11-gloss isolated-sign classification are implemented. Webcam accuracy is not guaranteed and has not been measured. |
| Meaning and response | Deterministic semantic frames, confidence-aware clarification, and bounded conversation history. |
| Skeleton avatar | Plays derived 2D landmark stick-figure clips for the supported glosses. This is not a rigged 3D avatar. |
| Text-to-sign | Deterministically maps supported words to glosses and plays matching clips; unmatched words are reported. |
| Optional LLM | Disabled unless `ISL_LLM_API_KEY` is configured; errors/timeouts fall back to the deterministic responder. |
| Evaluation | Separate 24-sample recognition run and phrase evaluation. Signer-independent performance is unverified. |
| Deployment | Docker and Hugging Face Spaces instructions are in [`docs/DEPLOY.md`](docs/DEPLOY.md). |

**Manual webcam smoke (2026-10-26):** in one local browser session, camera start showed the live preview and landmark overlay (1/1 start); a movement segment reached low-confidence clarification. No labeled signs were tested (0 accuracy trials), so this is a UI/pipeline smoke check, not an accuracy measurement.

## Model evaluation

The checked-in test split contains 44 samples. Its reported macro-F1 is **0.685** and weighted F1 is **0.717**. The `97. dry` gloss has no test examples and is excluded from the reported test macro-F1. Signer IDs are unavailable, so signer-independent evaluation cannot be verified. See [`training_outputs/bilstm_baseline/test_evaluation.json`](training_outputs/bilstm_baseline/test_evaluation.json) and [`reports/real_sample_validation.md`](reports/real_sample_validation.md) for per-class results and caveats. These figures describe this fixed split, not expected performance on new signers or environments.

## Run locally

Use Python 3.10 and a webcam-enabled browser. The server loads the checked-in model, label map, MediaPipe task models, and processed sample metadata from repository-relative paths.

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-preprocessing.txt -r requirements-training.txt
python -m pip install pytest
python -m pytest -q
python web/server.py 8000
```

### macOS / Linux

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-preprocessing.txt -r requirements-training.txt
python -m pip install pytest
python -m pytest -q
python web/server.py 8000
```

Open <http://127.0.0.1:8000>. Allow camera access when prompted. The browser camera API requires a secure context: `localhost` is allowed locally, while a remotely hosted instance must be served over HTTPS.

The server supports `HOST` and `PORT` environment variables. It binds to loopback by default for local use; container configuration sets `HOST=0.0.0.0`.

The runtime image includes `training_outputs/bilstm_baseline/best_model.pt`, its label map, both MediaPipe task models, test metadata, and the extracted landmark samples needed by the sample inference endpoint. These tracked files are not excluded by `.dockerignore`.

## Deploy with Docker

Requirements: Docker Engine and the Docker Compose plugin. The first build downloads the pinned Python runtime dependencies and CPU-only PyTorch; expect a large image and allow several minutes for the initial build.

```bash
docker compose up --build
```

Open <http://localhost:7860>. The failure log defaults to `/tmp/failure_events.jsonl` inside the container and is ephemeral. Set `ISL_FAILURE_LOG_PATH` to a writable path to change it. For a cloud container host, build from this repository, expose port `7860`, and configure the host's HTTPS endpoint so browser camera access works.

Check readiness at `/api/health`. The container health check uses the same endpoint. The process runs as a non-root user.

## Evaluation commands

Run recognition on the fixed test samples:

```bash
python scripts/run_round_trip_evaluation.py --limit 24
```

Run the separate phrase evaluation:

```bash
python -m src.lead_pipeline.phrase_evaluation
```

The phrase report is written to `reports/phrase_evaluation.json`. It distinguishes misrecognition, meaning ambiguity, and avatar mapping gaps using the checked-in skeleton clip index; unsupported glosses remain failures.

Capture sample-based demo evidence with:

```bash
python scripts/run_demo_evidence.py
```

This produces machine-readable evidence, **not** a recorded human demo video.

## Architecture and API

See [`doc/ARCHITECTURE.md`](doc/ARCHITECTURE.md) for the runtime boundaries and [`CONTRACT.md`](CONTRACT.md) for model and orchestrator data shapes.

**Project framing:** built a modular ISL recognition prototype with confidence-aware routing, deterministic semantic frames, explicit failure handling, and a measured end-to-end evaluation. It is intentionally described as a constrained prototype rather than a complete bidirectional translator.

- `GET /api/health` — readiness and label count.
- `GET /api/samples` — available processed test samples.
- `GET /api/failures` — latest logged clarification/failure events.
- `GET /api/infer?sample_id=...` — run inference on a processed sample.
- `POST /api/infer` — infer from JSON `features`, `mask`, and optional `timestamp`.
- `POST /api/infer_landmarks` — classify browser-extracted landmark frames and return top-two candidates.
- `POST /api/clarification` — record a user's candidate selection and compose its response.
- `POST /api/compose_sentence` — rebuild the grounded sentence after undo/clear actions.
- `POST /api/text_to_sign` — map text to supported glosses and unmatched words.

Camera frames are processed into landmarks in the browser and are not uploaded, stored, or logged. Landmark request bodies are size-capped and inference/text-to-sign endpoints are rate-limited. The legacy image-based `POST /api/frame` endpoint returns HTTP 410; camera processing no longer runs on the server. See [`doc/LIVE_SCHEMA.md`](doc/LIVE_SCHEMA.md) for the exact training feature layout and live JSON schema.

## Known limitations and honest next steps

- **Avatar:** skeleton clips are derived from INCLUDE dataset landmarks; the dataset license and redistribution rights require verification. A rigged 3D avatar remains future work. See [`avatar/README.md`](avatar/README.md) and [`avatar/ATTRIBUTION.md`](avatar/ATTRIBUTION.md).
- **Language coverage:** text-to-sign only maps individual supported words and the label synonyms in this repository. It does not provide a validated continuous-sign grammar or arbitrary natural-language ISL translation; there is no microphone input.
- **Optional LLM:** no key is required. If configured, the responder sends only the semantic frame to the configured OpenAI-compatible endpoint; do not send sensitive content.
- **Landmarks:** current model input uses hands and pose, not face/non-manual markers. Landmark extraction and the trained model are an experimental baseline.
- **Dataset split:** signer IDs are unavailable in the supplied metadata, so signer-independent separation cannot be verified. The `dry` class has zero test support.
- **Demo:** record a human-operated backup demo separately; generated JSON evidence is not video.
- **Hosted deployment:** this repository does not claim a live hosted service until the placeholder above is replaced with the deployed Space URL.

Dataset/video and third-party model asset terms remain subject to their respective providers' licenses. Review those terms before redistribution or commercial use.
