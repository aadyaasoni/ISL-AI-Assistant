# Full INCLUDE Dataset Workflow

This workflow reads the full INCLUDE archive from Google Drive or a local path, streams one video at a time through the repository's existing MediaPipe extractor, gates classes by valid sample count, creates deterministic train/validation/test splits, and trains the existing BiLSTM. The ZIP and source videos must stay outside Git.

## Requirements

- Access to the shared Drive folder: <https://drive.google.com/drive/folders/1t4i02oj5QIEyCqH1ilePXRkf8ACMxjIm?usp=share_link>
- Google Colab with enough persistent Drive space, or a local machine with the ZIP on disk.
- Python environment installed from `requirements-preprocessing.txt` and `requirements-training.txt`.
- MediaPipe task models from `models/mediapipe/`.

If the shared folder is listed under “Shared with me,” add a shortcut to My Drive in Drive first. The notebook authenticates as the signed-in Google account; the local process cannot access a private Drive folder unless the ZIP has been downloaded or mounted by the user.

## Colab: Gauri's steps

1. Open `notebooks/colab_extract_and_train.ipynb` in Colab or upload it to Colab.
2. Run all cells in order and authenticate to Drive when prompted.
3. In the file listing, identify the full INCLUDE ZIP. Set its file ID or exact name in the download cell. The download uses chunks and writes to My Drive; allow enough free space for the archive and generated landmarks.
4. Confirm the archive path printed by the notebook. The inventory cell lists video counts from the ZIP central directory without extracting members.
5. Keep `MIN_VALID_SAMPLES = 15`, or set the threshold required for the experiment. `FLUSH_EVERY = 25` writes metadata checkpoints and the progress log after each batch; each completed video is also logged.
6. Start extraction. For every member, the script writes only that member to a temporary file, imports and calls `scripts/extract_landmarks.py`, saves `features` `(T, 258)` and `mask` `(T, 75)` to `.npz`, then deletes the temporary video file.
7. If Colab disconnects, rerun the extraction cell with the same ZIP and output path. Existing valid `.npz` files are validated and skipped; incomplete outputs are removed and retried. `progress.jsonl` records each attempted video and failure.
8. Review `class_quality_report.json`. Only classes with at least the configured number of valid samples enter `training_metadata.csv`; the report lists every included and excluded class and exclusion reason.
9. Inspect split counts. Every included class has at least one train, validation, and test sample. The split is deterministic for a fixed seed. Source metadata has no signer IDs, so **signer-independent separation cannot be verified**; the workflow does not claim otherwise.
10. Train with the existing `scripts/train_bilstm.py`. The notebook writes the checkpoint and label map to a new `training_outputs/full_dataset` folder on Drive, reports validation/test per-class F1 and macro-F1, and includes top-label ECE plus multiclass Brier score. Test metrics are report-only and are not used for checkpoint selection.
11. The final cells regenerate the repository's landmark skeleton clips for every label in the new mapping and write a conservative `synonyms.json`. These clips are derived 2D landmark sequences, not a rigged 3D avatar or human-verified sign animation. Label-derived aliases are mechanical spelling variants only; aliases shared by multiple labels remain ambiguous candidates and are not auto-selected.

## Local execution

Activate a Python environment, install requirements, then inventory without extraction:

```bash
python -m pip install -r requirements-preprocessing.txt -r requirements-training.txt
python scripts/extract_from_zip.py /path/to/full_include.zip --list-only
```

Run a small extraction smoke test before the full job:

```bash
python scripts/extract_from_zip.py /path/to/full_include.zip \
  --output-root /path/to/isl_full_dataset \
  --hand-model models/mediapipe/hand_landmarker.task \
  --pose-model models/mediapipe/pose_landmarker_lite.task \
  --min-samples 15 --flush-every 25 --seed 42 --limit 10
```

After confirming the schema and progress behavior, rerun without `--limit 10` using the same output root. Then train:

```bash
python scripts/train_bilstm.py \
  --metadata /path/to/isl_full_dataset/training_metadata.csv \
  --output-dir /path/to/training_outputs/full_dataset
```

Generate full-vocabulary skeleton clips and aliases:

```bash
python scripts/build_avatar_clips.py \
  --metadata /path/to/isl_full_dataset/training_metadata.csv \
  --label-map /path/to/training_outputs/full_dataset/label_mapping.json \
  --clips-dir /path/to/avatar/clips/full_dataset
python scripts/build_label_aliases.py \
  --label-map /path/to/training_outputs/full_dataset/label_mapping.json \
  --output /path/to/training_outputs/full_dataset/synonyms.json
```

## Runtime estimate

The archive size, video count, video duration, storage speed, and available CPU/GPU are not known in advance, so a fixed wall-clock estimate would be misleading. Extraction is sequential and MediaPipe uses the existing CPU delegate. First run a representative 25-video pilot and record elapsed extraction time. Estimate the full extraction as:

`estimated extraction time = pilot elapsed time * total video count / pilot video count`

Add time for training separately; GPU availability, included class count, frame count, and epoch/patience determine training duration. Keep the Colab output on Drive so a disconnection does not require repeating completed extraction work.

## Bring back the small outputs

Do not copy the source ZIP or original videos into the repository. Keep all `.npz` landmarks on Drive unless local retraining is specifically needed. Copy only these generated artifacts back when ready:

- `training_outputs/full_dataset/best_model.pt`
- `training_outputs/full_dataset/label_mapping.json`
- `training_outputs/full_dataset/metrics.json` and `metrics.txt`
- `training_outputs/full_dataset/synonyms.json`
- `class_quality_report.json` and the metadata CSVs if the local machine needs to reproduce evaluation
- `avatar/clips/full_dataset/index.json` and its generated clip JSONs

The tracked `.gitignore` excludes full-dataset workspace outputs, ZIP files, and common video formats. Do not force-add those files. Before enabling a full-vocabulary web runtime, set matching paths together:

```bash
ISL_CHECKPOINT_DIR=training_outputs/full_dataset \
ISL_DATASET_METADATA=/path/to/isl_full_dataset/training_metadata.csv \
ISL_AVATAR_CLIPS_DIR=avatar/clips/full_dataset \
ISL_SYNONYMS_PATH=training_outputs/full_dataset/synonyms.json \
python web/server.py 8000
```

To return to the original 11-class model, set `ISL_CHECKPOINT_DIR=training_outputs/bilstm_baseline` and point the clip and metadata variables at the matching legacy resources. The default remains the legacy checkpoint when `ISL_CHECKPOINT_DIR` is unset.

## What is not verified here

The shared Drive folder requires Google sign-in and was not inspectable from the local browser session. No full archive was available locally, so full-dataset extraction, training, class counts, runtime, F1, calibration values, and generated full-vocabulary clips have not been run or measured. The synthetic ZIP tests validate streaming/resume behavior without real video. Review dataset license and redistribution terms before sharing generated clips or model artifacts.
