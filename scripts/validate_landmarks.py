import argparse
import csv
from collections import Counter
from pathlib import Path
from statistics import mean

import numpy as np


EXPECTED_FEATURE_DIMENSION = 258
EXPECTED_MASK_DIMENSION = 75


def parse_args():
    parser = argparse.ArgumentParser(description="Validate extracted landmark files and generate reports.")
    parser.add_argument("--landmark-metadata", type=Path, default=Path("data/include_metadata/include_metadata.csv"))
    parser.add_argument("--quality-report", type=Path, default=Path("data/include_metadata/video_quality.csv"))
    parser.add_argument("--metadata-output", type=Path, default=Path("data/include_metadata/metadata.csv"))
    parser.add_argument("--failed-output", type=Path, default=Path("data/include_metadata/failed_videos.csv"))
    parser.add_argument("--report-output", type=Path, default=Path("reports/preprocessing_report.txt"))
    return parser.parse_args()


def main():
    args = parse_args()
    with args.landmark_metadata.open(newline="", encoding="utf-8") as file:
        landmark_rows = list(csv.DictReader(file))
    with args.quality_report.open(newline="", encoding="utf-8") as file:
        quality_rows = {
            row["video_path"]: row for row in csv.DictReader(file)
        }

    final_rows = []
    failures = []
    sequence_lengths = []
    invalid_shapes = []

    for row in landmark_rows:
        quality = quality_rows.get(row["original_video_path"], {})
        processed_path = Path(row["processed_file_path"])
        row["fps"] = quality.get("fps", "")
        row["duration_seconds"] = quality.get("duration_seconds", "")

        if row["valid"] == "True" and processed_path.exists():
            try:
                with np.load(processed_path) as data:
                    features = data["features"]
                    mask = data["mask"]
                if features.ndim != 2 or features.shape[1] != EXPECTED_FEATURE_DIMENSION:
                    raise ValueError(f"features_shape={features.shape}")
                if mask.shape != (features.shape[0], EXPECTED_MASK_DIMENSION):
                    raise ValueError(f"mask_shape={mask.shape}")
                row["total_frames"] = str(features.shape[0])
                row["valid_frames"] = str(int(np.any(mask, axis=1).sum()))
                sequence_lengths.append(features.shape[0])
            except Exception as error:
                row["valid"] = "False"
                row["failure_reason"] = f"invalid_landmark_file:{error}"
                invalid_shapes.append(row["sample_id"])
        elif row["valid"] == "True":
            row["valid"] = "False"
            row["failure_reason"] = "missing_landmark_file"

        if row["valid"] != "True":
            failures.append(row.copy())
        final_rows.append(row)

    fieldnames = [
        "sample_id",
        "split",
        "category",
        "label",
        "original_video_path",
        "processed_file_path",
        "total_frames",
        "fps",
        "duration_seconds",
        "valid_frames",
        "feature_dimension",
        "valid",
        "failure_reason",
    ]
    args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
    args.failed_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    for output_path, rows in ((args.metadata_output, final_rows), (args.failed_output, failures)):
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    labels = sorted({row["label"] for row in final_rows})
    failure_categories = Counter(row["failure_reason"] or "none" for row in failures)
    successful = sum(row["valid"] == "True" for row in final_rows)
    report_lines = [
        "INCLUDE-50 Adjectives preprocessing report",
        "",
        f"Total videos found: {len(final_rows)}",
        f"Total videos processed: {len(final_rows)}",
        f"Successful extractions: {successful}",
        f"Failed extractions: {len(failures)}",
        f"Classes processed: {', '.join(labels)}",
        f"Feature dimension: {EXPECTED_FEATURE_DIMENSION}",
        f"Minimum sequence length: {min(sequence_lengths) if sequence_lengths else 0}",
        f"Maximum sequence length: {max(sequence_lengths) if sequence_lengths else 0}",
        f"Average sequence length: {mean(sequence_lengths) if sequence_lengths else 0:.2f}",
        f"Invalid landmark files: {len(invalid_shapes)}",
        "Failure categories:",
    ]
    report_lines.extend(f"  {category}: {count}" for category, count in failure_categories.items())
    report_lines.extend(
        [
            "",
            "Feature layout per frame:",
            "  left hand: 21 landmarks x (x, y, z)",
            "  right hand: 21 landmarks x (x, y, z)",
            "  pose: 33 landmarks x (x, y, z, visibility)",
            "Missing landmarks are zero-filled and represented in the 75-element mask.",
            "Coordinates are normalized around the shoulder midpoint when pose is available.",
        ]
    )
    args.report_output.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Validated: {len(final_rows)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(failures)}")
    print(f"Report: {args.report_output}")


if __name__ == "__main__":
    main()