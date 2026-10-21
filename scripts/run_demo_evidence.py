import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.recognition_adapter import RecognitionAdapter
from src.lead_pipeline import GlossClipResolver, Orchestrator

DEFAULT_SAMPLES = [
    "1._loud_MVI_5177_0222",
    "2._quiet_MVI_5180_0210",
    "3._happy_MVI_5183_0224",
]


def load_rows():
    with (ROOT / "data/include_metadata/metadata.csv").open(newline="", encoding="utf-8") as file:
        return {row["sample_id"]: row for row in csv.DictReader(file)}


def main():
    parser = argparse.ArgumentParser(description="Capture a reproducible recognition-to-avatar demo evidence run.")
    parser.add_argument("--output", type=Path, default=Path("reports/demo_evidence.json"))
    parser.add_argument("--sample", action="append", dest="samples")
    args = parser.parse_args()

    rows = load_rows()
    sample_ids = args.samples or DEFAULT_SAMPLES
    failures = []
    orchestrator = Orchestrator(failure_logger=failures.append)
    adapter = RecognitionAdapter(
        ROOT / "training_outputs/bilstm_baseline/best_model.pt",
        ROOT / "training_outputs/bilstm_baseline/label_mapping.json",
        device="cpu",
    )
    avatar_resolver = GlossClipResolver({}, lambda path: path, fallback_clip="pending://avatar-assets")
    cases = []

    for index, sample_id in enumerate(sample_ids):
        row = rows.get(sample_id)
        if row is None:
            raise SystemExit(f"Unknown sample_id: {sample_id}")
        with np.load(ROOT / row["processed_file_path"]) as sample:
            prediction = adapter.predict(sample["features"], sample["mask"], float(index))
        result = orchestrator.route(prediction)
        avatar_clip = avatar_resolver.resolve([prediction["gloss"]])[0]
        cases.append({
            "sample_id": sample_id,
            "expected_gloss": row["label"],
            "route_status": result["status"],
            "route_reason": result.get("reason"),
            "prediction": prediction,
            "avatar_clip": avatar_clip,
            "avatar_status": "pending-assets",
        })

    report = {
        "samples": len(cases),
        "cases": cases,
        "failure_events": failures,
        "notes": [
            "This is a reproducible evidence run, not a recorded video.",
            "Avatar playback remains pending until licensed model and animation assets are available.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"samples": len(cases), "failure_events": len(failures), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
