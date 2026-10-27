import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np

CSV_FIELDS = [
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
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm"}


def scan_zip(zip_path):
    """List class folders and video counts from the ZIP central directory only."""
    counts = Counter()
    videos = []
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if info.is_dir():
                continue
            path = PurePosixPath(info.filename)
            if path.suffix.lower() not in VIDEO_EXTENSIONS or any(part.startswith(".") for part in path.parts):
                continue
            if "__MACOSX" in path.parts or len(path.parts) < 2:
                continue
            label = path.parent.name.strip()
            if not label:
                continue
            videos.append((info.filename, label, path.parent.parent.name if len(path.parts) > 2 else "INCLUDE"))
            counts[label] += 1
    return sorted(videos), dict(sorted(counts.items()))


def stable_sample_id(label, member_name):
    stem = PurePosixPath(member_name).stem
    safe_label = "".join(char if char.isalnum() or char in "._-" else "_" for char in label)
    safe_stem = "".join(char if char.isalnum() or char in "._-" else "_" for char in stem)
    digest = hashlib.sha1(member_name.encode("utf-8")).hexdigest()[:10]
    return f"{safe_label}_{safe_stem}_{digest}"


def split_class_rows(rows, seed):
    """Assign deterministic stratified splits; every included class gets all three splits."""
    grouped = {}
    for row in rows:
        grouped.setdefault(row["label"], []).append(row)
    for label, class_rows in grouped.items():
        ordered = sorted(class_rows, key=lambda row: hashlib.sha256(f"{seed}:{row['sample_id']}".encode()).hexdigest())
        count = len(ordered)
        val_count = max(1, round(count * 0.1))
        test_count = max(1, round(count * 0.1))
        while val_count + test_count >= count:
            if val_count >= test_count and val_count > 1:
                val_count -= 1
            elif test_count > 1:
                test_count -= 1
            else:
                break
        train_count = count - val_count - test_count
        if min(train_count, val_count, test_count) < 1:
            raise ValueError(f"class {label!r} needs at least 3 valid samples for train/val/test")
        for index, row in enumerate(ordered):
            if index < train_count:
                row["split"] = "train"
            elif index < train_count + val_count:
                row["split"] = "val"
            else:
                row["split"] = "test"


def write_csv_atomic(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
        file.flush()
        os.fsync(file.fileno())
    temp_path.replace(path)


def load_existing_rows(path):
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as file:
        return {row["original_video_path"]: row for row in csv.DictReader(file)}


def append_progress(path, event):
    with Path(path).open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(event) + "\n")
        log_file.flush()
        os.fsync(log_file.fileno())


def inspect_sequence(npz_path):
    with np.load(npz_path) as data:
        features = data["features"]
        mask = data["mask"]
    if features.ndim != 2 or features.shape[1] != 258:
        raise ValueError(f"features_shape={features.shape}")
    if mask.shape != (features.shape[0], 75):
        raise ValueError(f"mask_shape={mask.shape}")
    if not np.isfinite(features).all() or not np.isfinite(mask).all():
        raise ValueError("non_finite_landmarks")
    valid_frames = int(np.any(mask, axis=1).sum())
    if valid_frames == 0:
        raise ValueError("no_detected_landmarks")
    return int(features.shape[0]), valid_frames


def build_row(member_name, label, category, output_path, processed_path, *, split="", valid=False, failure_reason=""):
    return {
        "sample_id": stable_sample_id(label, member_name),
        "split": split,
        "category": category,
        "label": label,
        "original_video_path": member_name,
        "processed_file_path": str(processed_path),
        "total_frames": 0,
        "fps": "",
        "duration_seconds": "",
        "valid_frames": 0,
        "feature_dimension": 258,
        "valid": str(valid),
        "failure_reason": failure_reason,
    }


def write_quality_report(path, counts, min_samples, included, excluded, split_counts):
    report = {
        "minimum_valid_samples_per_class": min_samples,
        "class_counts": counts,
        "included_classes": sorted(included),
        "excluded_classes": excluded,
        "split_counts": split_counts,
        "signer_independent_split_verified": False,
        "signer_split_note": "Signer IDs are not present in the source metadata, so signer-independent separation cannot be verified.",
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def process_zip(
    zip_path,
    output_root,
    hand_model,
    pose_model,
    *,
    min_samples=15,
    flush_every=25,
    seed=42,
    limit=0,
    list_only=False,
    extractor=None,
):
    zip_path = Path(zip_path)
    output_root = Path(output_root)
    landmark_root = output_root / "landmarks"
    landmark_root.mkdir(parents=True, exist_ok=True)
    metadata_path = output_root / "metadata.csv"
    training_metadata_path = output_root / "training_metadata.csv"
    log_path = output_root / "progress.jsonl"
    quality_path = output_root / "class_quality_report.json"

    videos, archive_counts = scan_zip(zip_path)
    archive_members = {member_name for member_name, _, _ in videos}
    print("Video inventory (ZIP is not extracted):")
    for label, count in archive_counts.items():
        print(f"  {label}: {count}")
    if list_only:
        return {"video_inventory": archive_counts, "videos": len(videos)}
    if limit > 0:
        videos = videos[:limit]

    if extractor is None:
        scripts_dir = str(Path(__file__).resolve().parent)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        import extract_landmarks as extractor

    hand_landmarker, pose_landmarker = extractor.build_landmarkers(hand_model, pose_model)
    existing = load_existing_rows(metadata_path)
    rows_by_member = {member: row for member, row in existing.items() if member in archive_members}
    last_flush = 0
    try:
        with zipfile.ZipFile(zip_path) as archive:
            for index, (member_name, label, category) in enumerate(videos, start=1):
                prior = rows_by_member.get(member_name)
                output_path = landmark_root / f"{stable_sample_id(label, member_name)}.npz"
                if output_path.is_file():
                    try:
                        frame_count, valid_frames = inspect_sequence(output_path)
                        row = prior or build_row(member_name, label, category, output_path, output_path, valid=True)
                        row.update({
                            "processed_file_path": str(output_path),
                            "total_frames": frame_count,
                            "valid_frames": valid_frames,
                            "feature_dimension": 258,
                            "valid": "True",
                            "failure_reason": "",
                        })
                        rows_by_member[member_name] = row
                        append_progress(log_path, {
                            "index": index,
                            "total": len(videos),
                            "video": member_name,
                            "status": "skipped_existing_output",
                            "valid": "True",
                            "failure_reason": "",
                        })
                        continue
                    except Exception:
                        output_path.unlink(missing_ok=True)

                row = build_row(member_name, label, category, output_path, output_path)
                start = time.time()
                temp_path = None
                temp_output = output_path.with_suffix(".npz.tmp")
                try:
                    suffix = PurePosixPath(member_name).suffix or ".video"
                    with tempfile.NamedTemporaryFile(prefix="isl_video_", suffix=suffix, delete=False) as temp_file:
                        temp_path = Path(temp_file.name)
                        with archive.open(member_name) as source:
                            shutil.copyfileobj(source, temp_file, length=1024 * 1024)
                    capture = extractor.cv2.VideoCapture(str(temp_path))
                    fps = float(capture.get(extractor.cv2.CAP_PROP_FPS) or 25.0)
                    capture.release()
                    features, masks = extractor.extract_video(
                        temp_path,
                        hand_landmarker,
                        pose_landmarker,
                        timestamp_offset=index * 1000000,
                    )
                    with temp_output.open("wb") as file:
                        np.savez_compressed(file, features=features, mask=masks)
                    frame_count, valid_frames = inspect_sequence(temp_output)
                    temp_output.replace(output_path)
                    row.update({
                        "total_frames": frame_count,
                        "fps": fps,
                        "duration_seconds": frame_count / fps if fps > 0 else "",
                        "valid_frames": valid_frames,
                        "valid": "True",
                    })
                except Exception as error:
                    row["failure_reason"] = str(error)
                    row["valid"] = "False"
                finally:
                    if temp_path is not None:
                        temp_path.unlink(missing_ok=True)
                    temp_output.unlink(missing_ok=True)
                rows_by_member[member_name] = row
                append_progress(log_path, {
                    "index": index,
                    "total": len(videos),
                    "video": member_name,
                    "status": "processed",
                    "valid": row["valid"],
                    "failure_reason": row["failure_reason"],
                    "elapsed_seconds": round(time.time() - start, 3),
                })
                print(f"[{index}/{len(videos)}] {member_name}: {row['failure_reason'] or 'ok'}")
                if index - last_flush >= flush_every:
                    write_csv_atomic(metadata_path, list(rows_by_member.values()))
                    last_flush = index
    finally:
        hand_landmarker.close()
        pose_landmarker.close()

    all_rows = list(rows_by_member.values())
    valid_rows = [row.copy() for row in all_rows if row["valid"] == "True"]
    valid_counts = Counter(row["label"] for row in valid_rows)
    class_counts = {label: valid_counts.get(label, 0) for label in archive_counts}
    class_counts = dict(sorted(class_counts.items()))
    included = {label for label, count in class_counts.items() if count >= min_samples}
    excluded = {
        label: {"valid_samples": count, "reason": f"below minimum threshold of {min_samples} valid samples"}
        for label, count in class_counts.items()
        if count < min_samples
    }
    train_rows = [row for row in valid_rows if row["label"] in included]
    split_class_rows(train_rows, seed) if train_rows else None
    split_by_class = {}
    for row in train_rows:
        split_by_class.setdefault(row["label"], Counter())[row["split"]] += 1
    split_counts = {label: dict(sorted(counts.items())) for label, counts in sorted(split_by_class.items())}
    train_by_member = {row["original_video_path"]: row for row in train_rows}
    final_rows = []
    for row in all_rows:
        updated = train_by_member.get(row["original_video_path"], row.copy())
        if row["valid"] == "True" and row["label"] not in included:
            updated["split"] = ""
        final_rows.append(updated)
    write_csv_atomic(metadata_path, final_rows)
    write_csv_atomic(training_metadata_path, train_rows)
    write_quality_report(quality_path, class_counts, min_samples, included, excluded, split_counts)
    print(f"Valid samples: {len(valid_rows)}; included classes: {len(included)}; excluded classes: {len(excluded)}")
    return {"metadata": metadata_path, "training_metadata": training_metadata_path, "quality_report": quality_path}


def parse_args():
    parser = argparse.ArgumentParser(description="Stream videos from an INCLUDE ZIP and save MediaPipe landmarks.")
    parser.add_argument("zip_path", type=Path, help="Path to the shared full INCLUDE ZIP; the archive is never extracted.")
    parser.add_argument("--output-root", type=Path, default=Path("data/full_dataset"))
    parser.add_argument("--hand-model", type=Path, default=Path("models/mediapipe/hand_landmarker.task"))
    parser.add_argument("--pose-model", type=Path, default=Path("models/mediapipe/pose_landmarker_lite.task"))
    parser.add_argument("--min-samples", type=int, default=15)
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="Optional extraction smoke-test limit.")
    parser.add_argument("--list-only", action="store_true", help="List video counts from the ZIP central directory without extracting or processing.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.min_samples < 3:
        raise SystemExit("--min-samples must be at least 3 to support train/val/test")
    if args.flush_every < 1:
        raise SystemExit("--flush-every must be positive")
    process_zip(args.zip_path, args.output_root, args.hand_model, args.pose_model, min_samples=args.min_samples, flush_every=args.flush_every, seed=args.seed, limit=args.limit, list_only=args.list_only)


if __name__ == "__main__":
    main()
