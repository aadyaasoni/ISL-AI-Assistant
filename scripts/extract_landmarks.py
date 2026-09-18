import argparse
import csv
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


HAND_LANDMARKS = 21
POSE_LANDMARKS = 33
FEATURE_DIMENSION = (2 * HAND_LANDMARKS * 3) + (POSE_LANDMARKS * 4)
TARGET_LABELS = {
    "1. loud",
    "2. quiet",
    "3. happy",
    "78. long",
    "79. short",
    "83. big large",
    "84. small little",
    "87. hot",
    "91. new",
    "94. good",
    "97. dry",
}


def empty_frame():
    return np.zeros(FEATURE_DIMENSION, dtype=np.float32), np.zeros(75, dtype=np.float32)


def landmark_xyz(landmarks, count):
    values = np.zeros((count, 3), dtype=np.float32)
    for index, landmark in enumerate(landmarks[:count]):
        values[index] = [landmark.x, landmark.y, landmark.z]
    return values


def normalize_landmarks(left_hand, right_hand, pose):
    visible_pose = pose[:, 3] > 0
    if visible_pose[11] and visible_pose[12]:
        center = (pose[11, :3] + pose[12, :3]) / 2.0
        scale = np.linalg.norm(pose[11, :2] - pose[12, :2])
    else:
        visible_hands = np.concatenate([left_hand, right_hand])
        visible_hands = visible_hands[np.any(visible_hands != 0, axis=1)]
        center = visible_hands[:, :3].mean(axis=0) if len(visible_hands) else np.zeros(3)
        scale = np.ptp(visible_hands[:, :2], axis=0).max() if len(visible_hands) else 1.0

    scale = max(float(scale), 1e-6)
    visible_left = np.any(left_hand != 0, axis=1)
    visible_right = np.any(right_hand != 0, axis=1)
    left_hand[visible_left] = (left_hand[visible_left] - center) / scale
    right_hand[visible_right] = (right_hand[visible_right] - center) / scale
    pose[visible_pose, :3] = (pose[visible_pose, :3] - center) / scale
    return left_hand, right_hand, pose


def frame_features(hand_result, pose_result):
    left_hand = np.zeros((HAND_LANDMARKS, 3), dtype=np.float32)
    right_hand = np.zeros((HAND_LANDMARKS, 3), dtype=np.float32)
    pose = np.zeros((POSE_LANDMARKS, 4), dtype=np.float32)
    mask = np.zeros(75, dtype=np.float32)

    for index, landmarks in enumerate(hand_result.hand_landmarks):
        hand = landmark_xyz(landmarks, HAND_LANDMARKS)
        handedness = hand_result.handedness[index][0].category_name.lower()
        if handedness == "left":
            left_hand = hand
            mask[:HAND_LANDMARKS] = 1.0
        elif handedness == "right":
            right_hand = hand
            mask[HAND_LANDMARKS : 2 * HAND_LANDMARKS] = 1.0

    if pose_result.pose_landmarks:
        for index, landmark in enumerate(pose_result.pose_landmarks[0][:POSE_LANDMARKS]):
            pose[index] = [landmark.x, landmark.y, landmark.z, landmark.visibility]
        mask[2 * HAND_LANDMARKS :] = pose[:, 3] > 0

    left_hand, right_hand, pose = normalize_landmarks(left_hand, right_hand, pose)
    features = np.concatenate([left_hand.reshape(-1), right_hand.reshape(-1), pose.reshape(-1)])
    return features.astype(np.float32), mask


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