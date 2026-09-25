import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.recognition_adapter import RecognitionAdapter
from src.lead_pipeline import RecognitionRuntime


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate real samples through recognition and orchestration.")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--output", type=Path, default=Path("reports/round_trip_evaluation.json"))
    return parser.parse_args()


def load_rows():
    with (ROOT / "data/include_metadata/metadata.csv").open(newline="", encoding="utf-8") as file:
        return [
            row for row in csv.DictReader(file) if row["split"] == "test" and row["valid"] == "True"
        ]


def main():
    args = parse_args()
    rows = load_rows()[: args.limit]
    adapter = RecognitionAdapter(
        ROOT / "training_outputs/bilstm_baseline/best_model.pt",
        ROOT / "training_outputs/bilstm_baseline/label_mapping.json",
        device="cpu",
    )
    runtime = RecognitionRuntime(adapter)
    cases = []
    counts = Counter()
    exact_matches = 0

    for index, row in enumerate(rows):
        try:
            with np.load(ROOT / row["processed_file_path"]) as sample:
                prediction = adapter.predict(sample["features"], sample["mask"], timestamp=float(index))
            routed = runtime.orchestrator.route(prediction)
            if prediction["gloss"] == row["label"]:
                exact_matches += 1
            outcome = routed.get("reason", routed.get("status", "unknown"))
            counts[outcome] += 1
            cases.append({
                "sample_id": row["sample_id"],
                "expected_gloss": row["label"],
                "prediction": prediction,
                "route": routed,
                "outcome": outcome,
            })
        except Exception as error:
            counts["error"] += 1
            cases.append({"sample_id": row["sample_id"], "expected_gloss": row["label"], "outcome": "error", "error": str(error)})

    report = {
        "samples": len(rows),
        "exact_gloss_matches": exact_matches,
        "exact_match_rate": exact_matches / len(rows) if rows else 0.0,
        "outcome_counts": dict(sorted(counts.items())),
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))
    print(f"Report: {args.output}")


if __name__ == "__main__":
    main()
