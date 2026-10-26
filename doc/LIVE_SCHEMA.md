# Live Landmark Inference Schema

Live camera processing runs in the browser with MediaPipe Tasks Vision
`0.10.22-rc.20250304`. The Python server receives landmark coordinates only;
it does not import or run MediaPipe or OpenCV. The `.task` files are served
from the repository at `/models/mediapipe/`.

## Request

`POST /api/infer_landmarks` accepts JSON of this form:

```json
{
  "frames": [
    {
      "hands": [
        {
          "handedness": "Left",
          "landmarks": [
            {"x": 0.42, "y": 0.38, "z": -0.01}
          ]
        }
      ],
      "pose_landmarks": [
        {"x": 0.50, "y": 0.25, "z": 0.00, "visibility": 0.99}
      ]
    }
  ],
  "context_glosses": []
}
```

Each hand contains exactly 21 MediaPipe normalized image landmarks, in
MediaPipe hand-landmark index order, and its `handedness` is the
`categoryName` returned for that same detection. `pose_landmarks` is either
empty or contains 33 pose landmarks in MediaPipe pose index order. Pose values
include visibility. The browser sends the model task outputs directly; CSS
mirroring is for display only and does not reorder coordinates or handedness.
The shortened JSON sample above illustrates the object shape; production
requests include every landmark for each detected hand and pose.

The live UI resamples a detected segment to 56 frames, a common training length
and the sequence length in the checked-in inference example. The trained model
is variable-length: its 169 training examples range from 38 to 83 frames
(median 54; 56 is the most frequent length). Training pads each batch to that
batch's longest sequence with zero feature rows and zero frame-mask rows, while
passing each sample's true length to `pack_padded_sequence`. There is no
checkpoint-wide fixed sequence size. The endpoint therefore accepts 1–120
frames and sends the provided length directly to the adapter without batch
padding. `context_glosses` is optional, contains at most 15 prior supported
gloss labels, and is used only to compose the running sentence.

## Per-frame training feature layout

The server's JSON preprocessor is the same pure-NumPy implementation used by
`scripts/extract_landmarks.py`. Coordinates are converted to `float32`.

| Feature offsets | Contents |
|---|---|
| `0:63` | Left hand: 21 points × `(x, y, z)` |
| `63:126` | Right hand: 21 points × `(x, y, z)` |
| `126:258` | Pose: 33 points × `(x, y, z, visibility)` |

The 75-element mask is ordered as 21 left-hand flags, 21 right-hand flags,
then 33 pose-visibility flags. A detected hand marks its 21 flags as 1. A pose
flag is 1 exactly when visibility is greater than zero.

Normalization matches dataset extraction on each frame. When both shoulders
(pose indices 11 and 12) have positive visibility, their 3D midpoint is the
origin and the Euclidean distance between their 2D `(x, y)` positions is the
scale. Otherwise, the origin is the mean of nonzero hand points and scale is
the maximum 2D hand-coordinate range (or 1 when there are no hand points).
Scale is clamped to at least `1e-6`. Visible hand coordinates and visible pose
`(x, y, z)` are translated and divided by that scale; visibility is unchanged.
Missing points remain exact zeroes. Hands are assigned by MediaPipe
`categoryName`, not by detection order.

The classifier derives its frame mask from `mask.sum(axis=1) > 0`, applies the
trained 258-feature BiLSTM, and returns the two most probable dataset labels.
The routing threshold remains the orchestrator's configured confidence
threshold. On low confidence, the API returns `decision: "clarify"` and a
clarification route; otherwise it returns `decision: "confident"` with the
grounded sentence and recognized gloss sequence. The browser starts a segment
on visible hand motion, ends after 600 ms without motion or hand visibility,
ignores sequences shorter than eight detected frames or 350 ms, and sends a
56-frame nearest-neighbor resample.

## Clarification and privacy

Only landmark JSON is posted for live classification. The server does not
persist request bodies or include landmark coordinates in failure events.
`POST /api/clarification` records a selected candidate using the existing
append-only failure log fields (`reason`, `gloss`, and `confidence`).
The image-based `POST /api/frame` endpoint is deprecated and returns HTTP 410;
the browser no longer uploads camera images to the server.
