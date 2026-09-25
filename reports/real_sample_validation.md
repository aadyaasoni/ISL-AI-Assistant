# Real-Sample Validation

Validation was run on the merged `main` branch on 2026-09-25.

## Automated checks

- Split audit: passed; train/validation/test counts are `169 / 20 / 44`.
- Cross-split duplicate paths, IDs, processed files, and filenames: none.
- Normalization audit: `233` files checked, `0` zero-mask violations.
- Adapter tests: `6/6` passed with `python3 -m unittest discover -s tests -v`.

## Aggregate evaluation

- Validation: `20` samples, macro-F1 `0.8121`.
- Test: `44` samples, macro-F1 `0.6852`.
- `97. dry` was excluded from test scoring because it has zero test support.
- The independent evaluation completed with `python3 scripts/evaluate_bilstm.py --split val` and `python3 scripts/evaluate_bilstm.py --split test`.

## Inference samples

Command used:

```bash
python3 scripts/infer_bilstm.py data/landmarks/include50/<sample>.npz --timestamp <timestamp>
```

| Sample | Expected label from filename | Predicted gloss | Confidence | Sequence length |
|---|---|---|---:|---:|
| `1._loud_MVI_5177_0222.npz` | `1. loud` | `1. loud` | `0.571` | `56` |
| `2._quiet_MVI_5180_0210.npz` | `2. quiet` | `2. quiet` | `0.677` | `68` |
| `3._happy_MVI_5183_0224.npz` | `3. happy` | `79. short` | `0.282` | `71` |

All three samples returned the required contract fields, `feature_dimension: 258`, an 11-value logits vector, and the supplied timestamp.

## Follow-up

The `3. happy` sample is a model-quality error, not an adapter or contract failure. Investigate additional happy samples, class confusion, and training data quality before changing the integration wrapper.

Additional happy samples were checked:

| Sample | Predicted gloss | Confidence |
|---|---|---:|
| `3._happy_MVI_5184_0100.npz` | `3. happy` | `0.343` |
| `3._happy_MVI_5185_0034.npz` | `79. short` | `0.307` |
| `3._happy_MVI_5263_0219.npz` | `3. happy` | `0.355` |
| `3._happy_MVI_5264_0047.npz` | `3. happy` | `0.525` |

Across five happy samples, three were classified as `3. happy` and two as `79. short`. The confusion is worth tracking as a model-quality follow-up, but it does not indicate an adapter contract failure.

## Model-quality follow-up

The test-set confusion analysis found:

- `3. happy` -> `79. short`: 1 sample.
- `79. short` -> `84. small little`: 1 sample.
- `84. small little` -> `79. short`: 3 samples.
- Other nonzero errors: `83. big large` -> `1. loud` (1), `87. hot` -> `78. long` (1), `87. hot` -> `79. short` (1), `91. new` -> `2. quiet` (1), `91. new` -> `94. good` (1), and `94. good` -> `91. new` (2).

There were `22` low-confidence predictions below `0.5` on the 44-sample test set. The 13 files from `happy`, `short`, and `small little` were checked for finite values and expected feature/mask shapes; all passed. The current evidence supports collecting more samples and reviewing class similarity before augmentation, balancing, or retraining.

## Integration coverage and gaps

- Real `.npz` inference and the `gloss`, `confidence`, and `timestamp` contract passed.
- The label mapping contains 11 glosses and matches the model output order.
- The avatar manifest currently maps no glosses: all 11 labels are explicitly listed as unsupported until licensed animations are added.
- Low-confidence outputs are observable through `confidence`; no fallback behavior for unmapped glosses exists in `RecognitionAdapter` yet.
- A browser/avatar round-trip and backup demo video remain blocked until licensed avatar assets and the presentation layer are available.
