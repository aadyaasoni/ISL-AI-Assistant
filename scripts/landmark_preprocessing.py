from typing import Any

import numpy as np


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


def _coordinate(item: Any, *, with_visibility: bool) -> list[float]:
    if not isinstance(item, dict):
        raise ValueError("landmarks must be objects with x, y, and z coordinates")
    keys = ("x", "y", "z", "visibility") if with_visibility else ("x", "y", "z")
    values = []
    for key in keys:
        value = item.get(key, 0.0 if key == "visibility" else None)
        if value is None:
            raise ValueError(f"landmark is missing {key}")
        try:
            parsed = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"landmark {key} must be numeric") from error
        if not np.isfinite(parsed):
            raise ValueError(f"landmark {key} must be finite")
        values.append(parsed)
    return values


def frame_from_json(frame: Any) -> tuple[np.ndarray, np.ndarray]:
    """Convert MediaPipe Tasks Vision JSON to the training feature and mask row."""
    if not isinstance(frame, dict):
        raise ValueError("each frame must be an object")
    hands = frame.get("hands", [])
    pose_landmarks = frame.get("pose_landmarks", [])
    if not isinstance(hands, list) or not isinstance(pose_landmarks, list):
        raise ValueError("hands and pose_landmarks must be arrays")
    if len(hands) > 2:
        raise ValueError("at most two hands are supported")
    if pose_landmarks and len(pose_landmarks) != POSE_LANDMARKS:
        raise ValueError(f"pose_landmarks must contain {POSE_LANDMARKS} landmarks")

    hand_landmarks = []
    handedness = []
    for hand in hands:
        if not isinstance(hand, dict):
            raise ValueError("each hand must be an object")
        category = hand.get("handedness")
        landmarks = hand.get("landmarks")
        if not isinstance(category, str) or category.lower() not in {"left", "right"}:
            raise ValueError("handedness must be Left or Right")
        if not isinstance(landmarks, list) or len(landmarks) != HAND_LANDMARKS:
            raise ValueError(f"each hand must contain {HAND_LANDMARKS} landmarks")
        hand_landmarks.append(
            [_SimpleLandmark(*_coordinate(item, with_visibility=False), visibility=1.0) for item in landmarks]
        )
        handedness.append([_SimpleCategory(category)])

    pose = []
    for item in pose_landmarks:
        values = _coordinate(item, with_visibility=True)
        pose.append(_SimpleLandmark(*values[:3], visibility=values[3]))
    hand_result = _SimpleHandResult(hand_landmarks, handedness)
    pose_result = _SimplePoseResult([pose] if pose else [])
    return frame_features(hand_result, pose_result)


def sequence_from_json(frames: Any) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(frames, list) or not frames:
        raise ValueError("frames must be a non-empty array")
    if len(frames) > 120:
        raise ValueError("a sign sequence cannot exceed 120 frames")
    converted = [frame_from_json(frame) for frame in frames]
    features = np.stack([frame[0] for frame in converted]).astype(np.float32)
    masks = np.stack([frame[1] for frame in converted]).astype(np.float32)
    if not np.any(masks[:, : 2 * HAND_LANDMARKS]):
        raise ValueError("a sign sequence must contain at least one detected hand")
    return features, masks


class _SimpleLandmark:
    def __init__(self, x, y, z, visibility=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility


class _SimpleCategory:
    def __init__(self, category_name):
        self.category_name = category_name


class _SimpleHandResult:
    def __init__(self, hand_landmarks, handedness):
        self.hand_landmarks = hand_landmarks
        self.handedness = handedness


class _SimplePoseResult:
    def __init__(self, pose_landmarks):
        self.pose_landmarks = pose_landmarks
