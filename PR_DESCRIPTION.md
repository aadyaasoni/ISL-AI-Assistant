## Summary

- Add the INCLUDE-50 Adjectives landmark preprocessing and BiLSTM baseline pipeline.
- Add split/normalization audits, separate validation/test evaluation, and the inference contract wrapper.
- Include reproducible metrics, label mapping, checkpoint, README, and review archive.

## Dataset and features

- Dataset: INCLUDE-50 Adjectives
- Classes/splits: 11 classes, `169 / 20 / 44` train/validation/test
- Features: `(T, 258)`
- Mask: `(T, 75)`
- Normalization: when pose landmarks 11 and 12 are available, `c = (p_11 + p_12) / 2`, `s = ||p_11_xy - p_12_xy||_2`, and `p' = (p - c) / max(s, 1e-6)`. If both shoulders are unavailable, the center is the mean of detected hand landmarks and the scale is the maximum XY range of those hand landmarks. Missing landmarks remain zero-filled and are represented by the mask.

## Results

- Validation macro-F1: `0.812121`
- Test macro-F1: `0.685211`
- Test data was not used for training or checkpoint selection.
- The best checkpoint was selected using validation macro-F1 only.
- `97. dry` has no test samples and is excluded from test macro-F1 rather than counted as zero.
- Split audit: no cross-split duplicate paths, IDs, processed files, or filenames.
- Normalization audit: 233 files checked, 0 zero-mask violations.

## Model output contract

```json
{
  "gloss": "label",
  "confidence": 0.0,
  "timestamp": 0.0
}
```

The wrapper also returns `class_id`, the 11-class `logits` vector, `feature_dimension`, and `sequence_length` for integration/debugging.

## Validation commands

```bash
python scripts/audit_splits.py
python scripts/verify_normalization.py
python scripts/evaluate_bilstm.py --split val
python scripts/evaluate_bilstm.py --split test
python scripts/infer_bilstm.py data/landmarks/include50/<sample>.npz --timestamp 123.0
```

## Review checklist

- [ ] Confirm clean checkout setup and dependencies.
- [ ] Review the feature/mask schema and normalization fallback.
- [ ] Review separate validation/test metrics.
- [ ] Review the inference contract and label mapping.
