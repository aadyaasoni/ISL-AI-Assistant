# ISL Recognition Contract

## Model input

The recognition model accepts one processed landmark sequence:

- `features`: float32 array with shape `(T, 258)`
- `mask`: float32 array with shape `(T, 75)`
- `T`: variable sequence length

The 258 feature values are ordered as left hand `(21 x 3)`, right hand `(21 x 3)`, and pose `(33 x 4)`. The 75 mask values use the corresponding 21 left-hand, 21 right-hand, and 33 pose landmark order.

## Model output

The inference integration output is JSON:

```json
{
  "gloss": "label",
  "confidence": 0.0,
  "timestamp": 0.0
}
```

The current wrapper also includes review/debug fields:

- `class_id`: integer index from `label_mapping.json`
- `logits`: one value per class in label-map order
- `feature_dimension`: `258`
- `sequence_length`: input `T`

`confidence` is the softmax probability of `gloss`. `timestamp` is supplied by the caller or generated at inference time.

## Python integration

```python
from scripts.recognition_adapter import RecognitionAdapter

adapter = RecognitionAdapter(checkpoint_path, label_map_path)
result = adapter.predict(features, mask, timestamp)
```

`RecognitionAdapter.predict` returns the three contract fields without requiring changes to the model implementation.

## Orchestrator output

The lead-owned orchestrator accepts the model output above. Predictions below
the default confidence threshold of `0.60` must request clarification instead
of reaching the meaning layer.

Accepted predictions produce:

```json
{
  "status": "accepted",
  "meaning": {
    "intent": "greeting",
    "entities": {},
    "slots": {},
    "source_gloss": "hello"
  },
  "response_text": "Hello. How can I help you?",
  "confidence": 0.92,
  "timestamp": 0.0
}
```

Low-confidence or unknown glosses produce `status: "clarification_required"`
and must not be silently converted into a meaning frame.
