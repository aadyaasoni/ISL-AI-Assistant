# INCLUDE-50 Landmark Preprocessing

This directory contains the Adjectives-category preprocessing baseline for the INCLUDE-50 Indian Sign Language dataset.

## Verified dataset

- Samples: 233 videos across 11 target labels
- Source metadata: `data/include_metadata/include50_metadata.csv`
- Original split: preserved from the supplied metadata
- Video quality: 233 readable, 0 failed
- Sequence length: 36 to 83 frames, average 55.91 frames
- Video rate: 25 FPS

## Landmark schema

Each `.npz` file contains:

- `features`: float32 array with shape `(T, 258)`
- `mask`: float32 array with shape `(T, 75)`

The 258 values per frame are ordered as:

1. Left hand: 21 landmarks x `(x, y, z)` = 63 values
2. Right hand: 21 landmarks x `(x, y, z)` = 63 values
3. Pose: 33 landmarks x `(x, y, z, visibility)` = 132 values

Within each group, MediaPipe landmark indices are kept in their native order: hand indices 0 through 20, followed by pose indices 0 through 32. The mask uses the same order: 21 left-hand flags, 21 right-hand flags, and 33 pose flags. A flag is 1 when that landmark is detected and 0 otherwise. Pose visibility is retained as the fourth pose feature and is not used as a replacement for the mask.

For each frame, let `p_i` be a landmark coordinate and let `c` be the reference center. When pose landmarks 11 and 12 are both available:

```text
c = (p_11 + p_12) / 2
s = ||p_11_xy - p_12_xy||_2
```

Otherwise, use all detected hand landmarks `H`:

```text
c = mean(H)
s = max(max(H_xy) - min(H_xy))
```

The scale is clamped to `1e-6`. Every coordinate is then normalized as:

```text
p_normalized = (p - c) / max(s, 1e-6)
```

If no pose or hand landmark is available, `c = (0, 0, 0)` and `s = 1`. Missing landmarks are zero-filled; videos are not silently deleted because of missed detections.

MediaPipe is used for a reproducible baseline because it provides hand and upper-body pose landmarks without retaining raw appearance information. It can fail under occlusion, poor lighting, unusual viewpoints, or motion blur. Landmark preprocessing also loses appearance, facial expression, and fine image detail. Face landmarks are intentionally deferred and are not part of this frozen baseline.

## Files

- `data/landmarks/include50/`: one compressed `.npz` sequence per video
- `data/include_metadata/metadata.csv`: final sample metadata
- `data/include_metadata/failed_videos.csv`: failed samples, currently empty apart from the CSV header
- `data/include_metadata/video_quality.csv`: raw video inspection results
- `reports/preprocessing_report.txt`: generated summary
- `scripts/check_video_quality.py`: video readability and duration scan
- `scripts/extract_landmarks.py`: MediaPipe Tasks extraction
- `scripts/validate_landmarks.py`: shape validation and report generation
- `models/mediapipe/`: required hand and pose `.task` assets

## Reproduce

Activate the environment and run:

```bash
source venv/bin/activate
python scripts/check_video_quality.py
python scripts/extract_landmarks.py \
  --hand-model models/mediapipe/hand_landmarker.task \
  --pose-model models/mediapipe/pose_landmarker_lite.task
python scripts/validate_landmarks.py
```

The original videos are read from `data/raw/include50` and are never modified.

## Colab loading example

Upload or mount the processed archive, then load samples with:

```python
import numpy as np
import pandas as pd

metadata = pd.read_csv("data/include_metadata/metadata.csv")
train = metadata[(metadata["split"] == "train") & (metadata["valid"])]

row = train.iloc[0]
sample = np.load(row["processed_file_path"])
features = sample["features"]
mask = sample["mask"]
label = row["label"]

print(features.shape, mask.shape, label)
```

For a BiLSTM or Transformer, keep sequences padded within each batch and use the mask or sequence lengths so padded frames do not contribute to attention or loss. Do not randomly redistribute samples: train/validation/test assignments are retained to avoid leakage. The supplied metadata does not expose signer IDs, so signer-independent separation cannot be verified from this preprocessing run. The `97. dry` class has no test samples in the original metadata and is left unchanged.

## Split counts and training contract

The final split counts are:

| Label | Train | Validation | Test |
|---|---:|---:|---:|
| `1. loud` | 13 | 2 | 6 |
| `2. quiet` | 16 | 2 | 3 |
| `3. happy` | 16 | 2 | 3 |
| `78. long` | 14 | 1 | 6 |
| `79. short` | 14 | 2 | 6 |
| `83. big large` | 15 | 2 | 4 |
| `84. small little` | 16 | 2 | 4 |
| `87. hot` | 14 | 1 | 6 |
| `91. new` | 16 | 2 | 3 |
| `94. good` | 16 | 2 | 3 |
| `97. dry` | 19 | 2 | 0 |
| **Total** | **169** | **20** | **44** |

Run the baseline with:

```bash
python -m pip install -r requirements-training.txt
python scripts/train_bilstm.py
```

The generated `training_outputs/bilstm_baseline/label_mapping.json` maps sorted original labels to integer class IDs. The model returns logits with shape `(batch_size, 11)`, in that mapping's order. Softmax probabilities and the predicted class are derived by the consumer; the model does not emit gloss text directly. The trainer writes `best_model.pt`, `metrics.json`, and `metrics.txt`. Macro-F1 is the primary summary metric, with precision, recall, F1, and support reported per class. `97. dry` is included in the label mapping but is explicitly marked not evaluated on test because its original test support is zero.

Run independent evaluation commands after training:

```bash
python scripts/evaluate_bilstm.py --split val
python scripts/evaluate_bilstm.py --split test
python scripts/audit_splits.py
python scripts/verify_normalization.py
```

The inference wrapper emits the integration contract expected by the communication pipeline:

```json
{
  "gloss": "1. loud",
  "confidence": 0.57,
  "timestamp": 1758210000.0,
  "class_id": 0,
  "logits": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  "feature_dimension": 258,
  "sequence_length": 56
}
```

`logits` contains one value for each class in `label_mapping.json`; `confidence` is the softmax probability of the selected gloss. The timestamp is supplied by the caller or generated at inference time.

## RecognitionAdapter integration

Use the adapter without changing the BiLSTM model:

```python
from scripts.recognition_adapter import RecognitionAdapter

adapter = RecognitionAdapter(
  "training_outputs/bilstm_baseline/best_model.pt",
  "training_outputs/bilstm_baseline/label_mapping.json",
)
result = adapter.predict(features, mask, timestamp)
```

`features` must have shape `(T, 258)` and `mask` must have shape `(T, 75)`. `result` contains exactly `gloss`, `confidence`, and `timestamp`, matching `CONTRACT.md`. The adapter also validates the two input shapes before invoking the model.

## Browser presentation layer

Run the local dashboard with the preprocessing environment, which provides MediaPipe and OpenCV:

```bash
venv/bin/python web/server.py 8000
```

Open `http://127.0.0.1:8000/` to preview the camera locally, extract MediaPipe hand/pose landmarks, and run short camera sequences through the checkpoint and lead runtime. The dashboard exposes low-confidence clarification and pending avatar fallback states instead of claiming unsupported clips are playable.

The browser sends JPEG frames to `POST /api/frame`; the server returns one normalized feature vector of length `258`, mask vector of length `75`, and a `has_landmarks` flag. Landmark-free frames are not buffered. The browser accumulates 24 valid frames and submits model-ready landmarks to `POST /api/infer` as JSON with `features` shaped `(T, 258)`, `mask` shaped `(T, 75)`, and an optional numeric `timestamp`. Invalid shapes are rejected with HTTP 400.

Run the 24-case runtime evaluation with:

```bash
python3 scripts/run_round_trip_evaluation.py --limit 24
```

The JSON report records exact gloss matches, route outcomes, low-confidence cases, unknown-gloss cases, and errors separately.

Capture a small reproducible demo evidence run with:

```bash
python3 scripts/run_demo_evidence.py
```

This writes `reports/demo_evidence.json` with raw predictions, route reasons, failure events, and avatar fallback status. It is an evidence artifact, not a substitute for a recorded video demo.

The orchestrator also records structured `failure_events` for low-confidence and unknown-gloss routes. Pass `failure_logger=callback` to `Orchestrator` to stream each event to application logging; each event contains `reason`, `gloss`, `confidence`, and `timestamp`.

## Validation and avatar preparation

The real-sample checks and aggregate evaluation results are recorded in [`reports/real_sample_validation.md`](reports/real_sample_validation.md). The adapter test suite currently passes all six tests.

Avatar integration requirements and the current asset status are documented in [`avatar/README.md`](avatar/README.md), [`avatar/asset_manifest.json`](avatar/asset_manifest.json), and [`avatar/ATTRIBUTION.md`](avatar/ATTRIBUTION.md). No avatar binaries are included until their source and redistribution license are confirmed.

## Scope and limitations

This is a documented landmark baseline, not a novel preprocessing method. It is suitable for the first recognition experiment. Later experiments can compare face/non-manual features, alternate normalization, temporal resampling, augmentation, and signer-aware splits when signer information becomes available.