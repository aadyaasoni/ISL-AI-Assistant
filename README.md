# ISL Communication Assistant

A deployable prototype for a constrained Indian Sign Language (ISL) communication flow:

```text
camera frame -> MediaPipe hand/pose landmarks -> BiLSTM gloss prediction
             -> confidence-aware routing -> semantic frame -> response text
```

The application provides a browser camera preview, a real-sample inference probe, explicit clarification for uncertain or unmapped glosses, and a recent-failure dashboard. Avatar clip resolution is wired, but playback is **not available**: there are no licensed avatar or sign-animation assets in this repository.

> **Scope:** this is an 11-class isolated-gloss research/demo prototype, not a general-purpose ISL translator or a production accessibility service. It does not currently translate free-form text or audio into sign language.

## Current capabilities

- MediaPipe extraction of hand and pose landmarks into the 258-feature / 75-mask model input format.
- A trained BiLSTM checkpoint and recognition adapter returning gloss, confidence, and timestamp.
- Confidence thresholding, clarification responses, deterministic meaning frames for the 11 dataset glosses, and bounded conversation state.
- Append-only JSONL logging of low-confidence and unknown-gloss events, with a browser view of recent events.
- A separate 25-case phrase meaning/response/avatar-mapping evaluation and the existing 24-sample recognition evaluation.
- Docker image and Compose configuration for self-hosting; GitHub Actions runs tests and builds the image.

## Model evaluation

The checked-in test split contains 44 samples. Its reported macro-F1 is **0.685** and weighted F1 is **0.717**. The `97. dry` class has no samples in that original test split and is excluded from the score. See [`training_outputs/bilstm_baseline/test_evaluation.json`](training_outputs/bilstm_baseline/test_evaluation.json) and [`reports/real_sample_validation.md`](reports/real_sample_validation.md) for per-class results and caveats. These figures describe this fixed split, not expected performance on new signers or environments.

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

## Deploy with Docker

Requirements: Docker Engine and the Docker Compose plugin. The first build downloads the pinned Python dependencies and CPU-only PyTorch; expect a large image and allow several minutes for the initial build.

```bash
docker compose up --build
```

Open <http://localhost:8000>. Compose persists the append-only failure log in the `isl-reports` named volume. To use a different log location, set `ISL_FAILURE_LOG_PATH` to a **relative** path in the container. For a cloud container host, build from this repository, expose port `8000`, and configure the host's HTTPS endpoint so browser camera access works.

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

The phrase report is written to `reports/phrase_evaluation.json`. It distinguishes misrecognition, meaning ambiguity, and avatar mapping gaps. Because the avatar manifest is intentionally pending, missing clips remain failures; the evaluation does not mark hypothetical assets as available.

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
- `POST /api/frame` — extract one frame's landmarks from a JSON base64 image.
- `POST /api/infer` — infer from JSON `features`, `mask`, and optional `timestamp`.

Camera frames are processed by the local service. Do not expose this prototype publicly without adding the authentication, request limits, privacy controls, and operational monitoring appropriate for your deployment.

## Known limitations and honest next steps

- **Avatar playback:** no model or sign clips are included. A source and redistribution license must be confirmed before adding assets or enabling playback. See [`avatar/README.md`](avatar/README.md) and [`avatar/asset_manifest.json`](avatar/asset_manifest.json).
- **Conversation model:** responses are deterministic templates. No external LLM is configured; enabling one requires a provider decision, credentials, and explicit handling of conversation data.
- **Language coverage:** only 11 isolated adjective glosses have meaning frames. There is no validated continuous-sign grammar, natural-language ISL sentence model, text-to-sign system, or microphone input.
- **Landmarks:** current model input uses hands and pose, not face/non-manual markers. Landmark extraction and the trained model are an experimental baseline.
- **Dataset split:** signer IDs are unavailable in the supplied metadata, so signer-independent separation cannot be verified. The `dry` class has zero test support.
- **Demo:** record a human-operated backup demo separately; generated JSON evidence is not video.
- **Hosted deployment:** Docker files provide a portable deployment target. A public instance still needs a hosting account, HTTPS domain, and deployment-specific secrets/configuration; this repository does not claim a live hosted service.

Dataset/video and third-party model asset terms remain subject to their respective providers' licenses. Review those terms before redistribution or commercial use.
