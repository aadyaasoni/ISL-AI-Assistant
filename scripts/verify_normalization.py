import argparse
import csv
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Verify mask semantics and normalized landmark invariants.")
    parser.add_argument("--metadata", type=Path, default=Path("data/include_metadata/metadata.csv"))
    args = parser.parse_args()
    with args.metadata.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    zero_mask_violations = 0
    for row in rows:
        with np.load(row["processed_file_path"]) as sample:
            features = sample["features"]
            mask = sample["mask"]
        left = features[:, :63].reshape(-1, 21, 3)
        right = features[:, 63:126].reshape(-1, 21, 3)
        pose = features[:, 126:].reshape(-1, 33, 4)
        zero_mask_violations += int(np.any(left[mask[:, :21] == 0] != 0))
        zero_mask_violations += int(np.any(right[mask[:, 21:42] == 0] != 0))
        zero_mask_violations += int(np.any(pose[:, :, :3][mask[:, 42:] == 0] != 0))
    print(f"files_checked: {len(rows)}")
    print(f"zero_mask_violations: {zero_mask_violations}")
    if zero_mask_violations:
        raise SystemExit(1)


if __name__ == "__main__":
    main()