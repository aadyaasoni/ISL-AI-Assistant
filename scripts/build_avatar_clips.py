import csv
import argparse
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_DIR = ROOT / "training_outputs/bilstm_baseline"
CHECKPOINT_DIR = Path(os.environ.get("ISL_CHECKPOINT_DIR", DEFAULT_CHECKPOINT_DIR))
METADATA_PATH = Path(os.environ.get("ISL_DATASET_METADATA", ROOT / "data/include_metadata/metadata.csv"))
LABEL_MAP_PATH = CHECKPOINT_DIR / "label_mapping.json"
CLIPS_DIR = Path(os.environ.get("ISL_AVATAR_CLIPS_DIR", ROOT / "avatar/clips"))
FRAME_COUNT = 30


def clip_filename(gloss: str) -> str:
    return f"{gloss.strip().lower().replace(' ', '_')}.json"


def representative_samples(metadata_path=METADATA_PATH, label_map_path=LABEL_MAP_PATH) -> dict[str, dict[str, str]]:
    with Path(label_map_path).open(encoding="utf-8") as file:
        labels = json.load(file)
    with Path(metadata_path).open(newline="", encoding="utf-8") as file:
        rows = [
            row
            for row in csv.DictReader(file)
            if row["valid"] == "True" and row["label"] in labels
        ]

    representatives = {}
    for split_order in ("test", "train", "val"):
        for row in sorted(rows, key=lambda item: item["sample_id"]):
            if row["split"] == split_order:
                representatives.setdefault(row["label"], row)

    missing = set(labels).difference(representatives)
    if missing:
        raise ValueError(f"No valid landmark samples for glosses: {sorted(missing)}")
    return representatives


def rounded(values: np.ndarray) -> list:
    return np.round(values, 3).tolist()


def build_clips(metadata_path=METADATA_PATH, label_map_path=LABEL_MAP_PATH, clips_dir=CLIPS_DIR) -> dict[str, str]:
    clips_dir = Path(clips_dir)
    clips_dir.mkdir(parents=True, exist_ok=True)
    index = {}

    for gloss, row in representative_samples(metadata_path, label_map_path).items():
        sample_path = ROOT / row["processed_file_path"]
        with np.load(sample_path) as sample:
            features = sample["features"]
            mask = sample["mask"]

        if features.ndim != 2 or features.shape[1] != 258:
            raise ValueError(f"{row['sample_id']}: features must have shape (T, 258)")
        if mask.shape != (features.shape[0], 75):
            raise ValueError(f"{row['sample_id']}: mask must have shape (T, 75)")

        frame_indices = np.linspace(
            0,
            features.shape[0] - 1,
            num=min(FRAME_COUNT, features.shape[0]),
        ).round().astype(int)
        frames = []
        for frame_index in frame_indices:
            values = features[frame_index]
            frames.append(
                {
                    "left_hand": rounded(values[:63].reshape(21, 3)),
                    "right_hand": rounded(values[63:126].reshape(21, 3)),
                    "pose": rounded(values[126:].reshape(33, 4)),
                    "mask": mask[frame_index].astype(int).tolist(),
                }
            )

        filename = clip_filename(gloss)
        payload = {
            "gloss": gloss,
            "sample_id": row["sample_id"],
            "coordinate_space": "normalized-landmarks",
            "frames": frames,
        }
        (clips_dir / filename).write_text(
            json.dumps(payload, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        index[gloss] = filename

    (clips_dir / "index.json").write_text(
        json.dumps({"gloss_to_clip": index}, indent=2) + "\n",
        encoding="utf-8",
    )
    return index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate landmark skeleton clips for labels in a checkpoint mapping.")
    parser.add_argument("--metadata", type=Path, default=METADATA_PATH)
    parser.add_argument("--label-map", type=Path, default=LABEL_MAP_PATH)
    parser.add_argument("--clips-dir", type=Path, default=CLIPS_DIR)
    args = parser.parse_args()
    clips = build_clips(args.metadata, args.label_map, args.clips_dir)
    total_bytes = sum(path.stat().st_size for path in args.clips_dir.glob("*.json"))
    print(f"Built {len(clips)} skeleton clips ({total_bytes} bytes).")
