from dataclasses import dataclass
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    prediction: dict[str, Any]
    expected_intent: str


def evaluate_cases(
    cases: Iterable[EvaluationCase],
    route: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate meaning preservation and classify failures for review."""
    results = []
    failures = []

    for case in cases:
        output = route(case.prediction)
        passed = (
            output.get("status") == "accepted"
            and output.get("meaning", {}).get("intent") == case.expected_intent
        )
        result = {"case_id": case.case_id, "passed": passed}
        if passed:
            results.append(result)
            continue

        reason = _failure_reason(output, case.expected_intent)
        result["failure_category"] = reason
        result["output"] = output
        results.append(result)
        failures.append(result)

    total = len(results)
    passed_count = sum(result["passed"] for result in results)
    categories: dict[str, int] = {}
    for failure in failures:
        category = failure["failure_category"]
        categories[category] = categories.get(category, 0) + 1

    return {
        "total": total,
        "passed": passed_count,
        "failed": total - passed_count,
        "pass_rate": passed_count / total if total else 0.0,
        "failure_categories": categories,
        "cases": results,
    }


def _failure_reason(output: dict[str, Any], expected_intent: str) -> str:
    if output.get("reason") == "low_confidence":
        return "misrecognition"
    if output.get("status") == "clarification_required":
        return "meaning_layer_ambiguity"
    if output.get("meaning", {}).get("intent") == expected_intent:
        return "avatar_mapping_gap"
    return "meaning_layer_ambiguity"