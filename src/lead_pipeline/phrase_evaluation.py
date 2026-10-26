import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .avatar import GlossClipResolver
from .orchestrator import Orchestrator


_ROOT = Path(__file__).resolve().parents[2]
_PROPERTY_MEANINGS = {
    "1. loud": "loud",
    "2. quiet": "quiet",
    "3. happy": "happy",
    "78. long": "long",
    "79. short": "short",
    "83. big large": "large",
    "84. small little": "small",
    "87. hot": "hot",
    "91. new": "new",
    "94. good": "good",
    "97. dry": "dry",
}


@dataclass(frozen=True)
class PhraseEvaluationCase:
    case_id: str
    gloss_sequence: tuple[str, ...]
    expected_meaning: tuple[str, ...]
    expected_response: str
    confidence_sequence: tuple[float, ...] = ()


def _phrase_case(
    case_id: str,
    gloss_sequence: tuple[str, ...],
    confidence_sequence: tuple[float, ...] = (),
) -> PhraseEvaluationCase:
    expected_meaning = tuple(
        _PROPERTY_MEANINGS.get(gloss, "unmapped") for gloss in gloss_sequence
    )
    if "unmapped" in expected_meaning:
        expected_response = "I do not recognize that sign yet. Please repeat it."
    else:
        expected_response = " ".join(
            f"I understood the property: {meaning}."
            for meaning in expected_meaning
        )
    return PhraseEvaluationCase(
        case_id=case_id,
        gloss_sequence=gloss_sequence,
        expected_meaning=expected_meaning,
        expected_response=expected_response,
        confidence_sequence=confidence_sequence,
    )


PHRASE_EVALUATION_CASES = (
    _phrase_case("single-loud", ("1. loud",)),
    _phrase_case("single-quiet", ("2. quiet",)),
    _phrase_case("single-happy", ("3. happy",)),
    _phrase_case("single-long", ("78. long",)),
    _phrase_case("single-short", ("79. short",)),
    _phrase_case("single-large", ("83. big large",)),
    _phrase_case("single-small", ("84. small little",)),
    _phrase_case("single-hot", ("87. hot",)),
    _phrase_case("single-new", ("91. new",)),
    _phrase_case("single-good", ("94. good",)),
    _phrase_case("single-dry", ("97. dry",)),
    _phrase_case("pair-loud-quiet", ("1. loud", "2. quiet")),
    _phrase_case("pair-happy-good", ("3. happy", "94. good")),
    _phrase_case("pair-long-short", ("78. long", "79. short")),
    _phrase_case("pair-large-small", ("83. big large", "84. small little")),
    _phrase_case("pair-hot-dry", ("87. hot", "97. dry")),
    _phrase_case("pair-new-good", ("91. new", "94. good")),
    _phrase_case("pair-quiet-happy", ("2. quiet", "3. happy")),
    _phrase_case("pair-short-large", ("79. short", "83. big large")),
    _phrase_case("pair-small-new", ("84. small little", "91. new")),
    _phrase_case("pair-dry-hot", ("97. dry", "87. hot")),
    _phrase_case("pair-good-loud", ("94. good", "1. loud")),
    _phrase_case("low-confidence-loud", ("1. loud",), (0.2,)),
    _phrase_case("unmapped-gloss", ("unseen gloss",)),
    _phrase_case("pair-long-new", ("78. long", "91. new")),
)


def evaluate_phrase_cases(
    cases: Iterable[PhraseEvaluationCase],
    orchestrator: Orchestrator,
    avatar_resolver: GlossClipResolver,
) -> dict[str, Any]:
    results = []
    failures = []

    for case in cases:
        confidences = case.confidence_sequence or (0.95,) * len(case.gloss_sequence)
        if len(confidences) != len(case.gloss_sequence):
            raise ValueError(
                f"{case.case_id}: confidence count must match gloss count"
            )

        routed = [
            orchestrator.route(
                {
                    "gloss": gloss,
                    "confidence": confidence,
                    "timestamp": float(index),
                }
            )
            for index, (gloss, confidence) in enumerate(
                zip(case.gloss_sequence, confidences)
            )
        ]

        actual_meaning = tuple(
            result.get("meaning", {}).get("slots", {}).get("property", "unmapped")
            for result in routed
        )
        actual_response = " ".join(
            result.get("response_text", result.get("message", ""))
            for result in routed
        )

        failure_category = None
        if any(result.get("reason") == "low_confidence" for result in routed):
            failure_category = "misrecognition"
        elif any(result.get("status") != "accepted" for result in routed):
            failure_category = "meaning_ambiguity"
        else:
            if (
                actual_meaning != case.expected_meaning
                or actual_response != case.expected_response
            ):
                failure_category = "meaning_ambiguity"
            else:
                try:
                    avatar_resolver.resolve(list(case.gloss_sequence))
                except KeyError:
                    failure_category = "avatar_mapping_gap"

        result = {
            "case_id": case.case_id,
            "gloss_sequence": list(case.gloss_sequence),
            "expected_meaning": list(case.expected_meaning),
            "actual_meaning": list(actual_meaning),
            "expected_response": case.expected_response,
            "actual_response": actual_response,
            "passed": failure_category is None,
        }
        if failure_category is not None:
            result["failure_category"] = failure_category
            result["routes"] = routed
            failures.append(result)
        results.append(result)

    categories: dict[str, int] = {}
    for failure in failures:
        category = failure["failure_category"]
        categories[category] = categories.get(category, 0) + 1

    total = len(results)
    passed = sum(result["passed"] for result in results)
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total else 0.0,
        "failure_categories": categories,
        "cases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate fixed gloss phrases through meaning, response, and avatar routing."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/phrase_evaluation.json"),
    )
    args = parser.parse_args()
    output_path = args.output
    if not output_path.is_absolute():
        output_path = _ROOT / output_path

    with (_ROOT / "avatar/clips/index.json").open(encoding="utf-8") as file:
        clip_index = json.load(file)["gloss_to_clip"]
    avatar_resolver = GlossClipResolver(
        mapping={
            gloss.strip().lower(): str(_ROOT / "avatar/clips" / filename)
            for gloss, filename in clip_index.items()
        },
        clip_loader=lambda clip_path: clip_path,
    )
    report = evaluate_phrase_cases(
        PHRASE_EVALUATION_CASES,
        Orchestrator(),
        avatar_resolver,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "cases"},
            indent=2,
        )
    )
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
