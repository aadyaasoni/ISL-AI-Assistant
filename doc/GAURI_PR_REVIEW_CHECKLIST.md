# Gauri PR Review Checklist

## Repository and reproducibility

- [ ] PR targets `main` and does not merge directly.
- [ ] No raw videos, virtual environments, or machine-specific paths are committed.
- [ ] Setup and reproduction commands are documented.

## Dataset and splits

- [ ] Dataset labels and original paths are preserved.
- [ ] Train/validation/test counts are documented.
- [ ] No cross-split duplicate video paths, IDs, processed files, or filenames.
- [ ] Test data is not used during training or checkpoint selection.

## Landmark contract

- [ ] Features have shape `(T, 258)`.
- [ ] Masks have shape `(T, 75)`.
- [ ] Landmark order and normalization formula match `CONTRACT.md`.
- [ ] Missing landmark behavior is documented and tested.

## Training and evaluation

- [ ] Label mapping is included.
- [ ] Best checkpoint is selected using validation macro-F1 only.
- [ ] Validation and test evaluation are separate commands/artifacts.
- [ ] Per-class metrics are included.
- [ ] `97. dry` is excluded from test macro-F1 because test support is zero.

## Inference contract

- [ ] Wrapper emits `gloss`, `confidence`, and `timestamp`.
- [ ] Logits follow `label_mapping.json` order.
- [ ] Wrapper runs from a clean-style checkout using relative paths.