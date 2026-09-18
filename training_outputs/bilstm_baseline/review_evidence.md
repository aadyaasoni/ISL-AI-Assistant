# Baseline Review Evidence

## Split separation

The authoritative metadata contains 169 train, 20 validation, and 44 test samples. The split audit found zero cross-split duplicates for original video paths, sample IDs, processed paths, and filenames. See `split_audit.json` and `split_audit.log`.

The training script constructs separate datasets and loaders from the `split` field. The test loader is not iterated during the training loop. Checkpoint selection compares only `val_macro_f1`; the test loader is used only after the best validation checkpoint is reloaded.

## Separate metric commands

```bash
python scripts/evaluate_bilstm.py --split val
python scripts/evaluate_bilstm.py --split test
```

Corrected final results:

```text
validation samples=20 macro_f1=0.8121212121212121
test samples=44 macro_f1=0.6852106227106227
```

The earlier displayed values `0.8061` and `0.8061` were rounded values from the superseded pre-correction run. They were not exactly equal: `0.8060606061` versus `0.8060739261`. After correcting missing-landmark normalization to preserve masked coordinates as exact zeros, the baseline was retrained and the independent metrics above replaced them.

## Dry class

`97. dry` has 19 train samples, 2 validation samples, and 0 test samples. It is present in the label mapping and evaluated on validation, but excluded from test macro-F1 because its test support is zero. The test evaluation artifact records it in `excluded_zero_support_classes`.

## Normalization

```bash
python scripts/verify_normalization.py
```

Result:

```text
files_checked: 233
zero_mask_violations: 0
```

## Inference contract

```bash
python scripts/infer_bilstm.py data/landmarks/include50/<sample>.npz --timestamp 123.0
```

The wrapper returns `gloss`, softmax `confidence`, `timestamp`, integer `class_id`, the complete 11-class `logits` vector, `feature_dimension`, and `sequence_length`.