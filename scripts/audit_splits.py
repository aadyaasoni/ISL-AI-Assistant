import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Audit split counts and cross-split identity collisions.")
    parser.add_argument("--metadata", type=Path, default=Path("data/include_metadata/metadata.csv"))
    parser.add_argument("--output", type=Path, default=Path("training_outputs/bilstm_baseline/split_audit.json"))
    args = parser.parse_args()
    with args.metadata.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    split_rows = {split: [row for row in rows if row["split"] == split] for split in ("train", "val", "test")}

    def cross_split_duplicates(field):
        locations = defaultdict(set)
        for row in rows:
            locations[row[field]].add(row["split"])
        return sorted(value for value, splits in locations.items() if len(splits) > 1)

    audit = {
        "sample_counts": {split: len(split_rows[split]) for split in split_rows},
        "class_counts": {split: dict(sorted(Counter(row["label"] for row in split_rows[split]).items())) for split in split_rows},
        "cross_split_duplicates": {
            "original_video_path": cross_split_duplicates("original_video_path"),
            "sample_id": cross_split_duplicates("sample_id"),
            "processed_file_path": cross_split_duplicates("processed_file_path"),
        },
        "cross_split_filename_collisions": [],
    }
    filenames = defaultdict(set)
    for row in rows:
        filenames[Path(row["original_video_path"]).name].add(row["split"])
    audit["cross_split_filename_collisions"] = sorted(name for name, splits in filenames.items() if len(splits) > 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()