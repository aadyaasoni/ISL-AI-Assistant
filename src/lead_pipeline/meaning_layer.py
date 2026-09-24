from typing import Any


_GLOSS_MEANINGS: dict[str, dict[str, Any]] = {
    "hello": {"intent": "greeting", "entities": {}, "slots": {}},
    "dry": {
        "intent": "describe_property",
        "entities": {},
        "slots": {"property": "dry"},
    },
}


def meaning_for_gloss(gloss: str) -> dict[str, Any]:
    """Map a normalized gloss to a deterministic semantic frame."""
    normalized_gloss = gloss.strip().lower()
    meaning = _GLOSS_MEANINGS.get(normalized_gloss)

    if meaning is None:
        return {
            "intent": "unknown",
            "entities": {},
            "slots": {},
            "source_gloss": normalized_gloss,
        }

    return {**meaning, "source_gloss": normalized_gloss}