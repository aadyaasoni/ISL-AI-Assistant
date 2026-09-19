import argparse
import csv
from pathlib import Path

import cv2


VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}


def inspect_video(video_path):
    result = {
        "video_exists": video_path.is_file(),
        "readable": False,
        "total_frames": 0,
        "fps": 0.0,
        "duration_seconds": 0.0,
        "failure_reason": "",
    }

    if not result["video_exists"]:
        result["failure_reason"] = "missing_file"
        return result

    if video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        result["failure_reason"] = "unsupported_extension"
        return result

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        result["failure_reason"] = "cannot_open"
        capture.release()
        return result

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    decoded_frames = 0

    while True:
        success, _ = capture.read()
        if not success:
            break
        decoded_frames += 1

    capture.release()

    result["total_frames"] = decoded_frames
    result["fps"] = fps
    result["duration_seconds"] = decoded_frames / fps if fps > 0 else 0.0

    if decoded_frames == 0:
        result["failure_reason"] = "zero_decoded_frames"
    elif fps <= 0:
        result["failure_reason"] = "invalid_fps"
    elif declared_frames > 0 and abs(declared_frames - decoded_frames) > 1:
        result["failure_reason"] = "frame_count_mismatch"
    else:
        result["readable"] = True

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Check videos referenced by INCLUDE metadata.")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("data/include_metadata/include50_metadata.csv"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/include50"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/include_metadata/video_quality.csv"),
    )
    parser.add_argument(
        "--category",
        default="Adjectives",
        help="Only inspect metadata rows in this category.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rows = []

    with args.metadata.open(newline="", encoding="utf-8-sig") as metadata_file:
        for metadata_row in csv.DictReader(metadata_file):
            if metadata_row["parent_label"] != args.category:
                continue

            video_path = args.dataset_root / metadata_row["video_path"]
            quality = inspect_video(video_path)
            rows.append(
                {
                    "split": metadata_row["split"],
                    "category": metadata_row["parent_label"],
                    "label": metadata_row["label"],
                    "video_path": metadata_row["video_path"],
                    **quality,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    successful = sum(row["readable"] for row in rows)
    failures = len(rows) - successful
    print(f"Videos inspected: {len(rows)}")
    print(f"Readable videos: {successful}")
    print(f"Failed videos: {failures}")
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()