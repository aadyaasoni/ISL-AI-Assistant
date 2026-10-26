import argparse
import csv
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    from .landmark_preprocessing import (
        FEATURE_DIMENSION,
        TARGET_LABELS,
        empty_frame,
        frame_features,
        landmark_xyz,
        normalize_landmarks,
    )
except ImportError:
    from landmark_preprocessing import (
        FEATURE_DIMENSION,
        TARGET_LABELS,
        empty_frame,
        frame_features,
        landmark_xyz,
        normalize_landmarks,
    )

def build_landmarkers(hand_model, pose_model):
    hand_options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(hand_model),
            delegate=python.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose_options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(pose_model),
            delegate=python.BaseOptions.Delegate.CPU,
        ),
        running_mode=vision.RunningMode.VIDEO,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.HandLandmarker.create_from_options(hand_options), vision.PoseLandmarker.create_from_options(pose_options)


def extract_video(video_path, hand_landmarker, pose_landmarker, timestamp_offset):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("cannot_open")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    features = []
    masks = []
    frame_index = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        timestamp_ms = timestamp_offset + int(frame_index * 1000 / fps)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        hand_result = hand_landmarker.detect_for_video(image, timestamp_ms)
        pose_result = pose_landmarker.detect_for_video(image, timestamp_ms)
        feature_row, mask_row = frame_features(hand_result, pose_result)
        features.append(feature_row)
        masks.append(mask_row)
        frame_index += 1

    capture.release()
    if not features:
        raise RuntimeError("zero_decoded_frames")
    return np.stack(features), np.stack(masks)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract normalized MediaPipe hand and pose sequences.")
    parser.add_argument("--metadata", type=Path, default=Path("data/include_metadata/include50_metadata.csv"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/include50"))
    parser.add_argument("--output-root", type=Path, default=Path("data/landmarks/include50"))
    parser.add_argument("--hand-model", type=Path, required=True)
    parser.add_argument("--pose-model", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Process only the first N rows for a smoke test.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    with args.metadata.open(newline="", encoding="utf-8-sig") as metadata_file:
        for metadata_row in csv.DictReader(metadata_file):
            if metadata_row["parent_label"] == "Adjectives" and metadata_row["label"] in TARGET_LABELS:
                rows.append(metadata_row)
    if args.limit > 0:
        rows = rows[: args.limit]

    hand_landmarker, pose_landmarker = build_landmarkers(args.hand_model, args.pose_model)
    metadata_rows = []
    failed_rows = []
    try:
        for index, metadata_row in enumerate(rows):
            source_path = args.dataset_root / metadata_row["video_path"]
            sample_id = f"{metadata_row['label'].replace(' ', '_')}_{Path(metadata_row['video_path']).stem}_{index:04d}"
            output_path = args.output_root / f"{sample_id}.npz"
            result = {
                "sample_id": sample_id,
                "split": metadata_row["split"],
                "category": metadata_row["parent_label"],
                "label": metadata_row["label"],
                "original_video_path": metadata_row["video_path"],
                "processed_file_path": str(output_path),
                "total_frames": 0,
                "valid_frames": 0,
                "feature_dimension": FEATURE_DIMENSION,
                "valid": False,
                "failure_reason": "",
            }
            try:
                features, masks = extract_video(
                    source_path,
                    hand_landmarker,
                    pose_landmarker,
                    timestamp_offset=index * 1000000,
                )
                np.savez_compressed(output_path, features=features, mask=masks)
                result["total_frames"] = int(features.shape[0])
                result["valid_frames"] = int(np.any(masks, axis=1).sum())
                result["valid"] = True
            except Exception as error:
                result["failure_reason"] = str(error)
                failed_rows.append(result.copy())
            metadata_rows.append(result)
            print(f"[{index + 1}/{len(rows)}] {metadata_row['video_path']} -> {result['failure_reason'] or 'ok'}")
    finally:
        hand_landmarker.close()
        pose_landmarker.close()

    metadata_dir = args.output_root.parent.parent / "include_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    metadata_output = metadata_dir / "include_metadata.csv"
    failed_output = metadata_dir / "failed_videos.csv"
    fieldnames = list(metadata_rows[0].keys()) if metadata_rows else []
    with metadata_output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata_rows)
    with failed_output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(failed_rows)
    print(json.dumps({"processed": len(metadata_rows), "failed": len(failed_rows), "feature_dimension": FEATURE_DIMENSION}))


if __name__ == "__main__":
    main()