from typing import Any


_GLOSS_MEANINGS: dict[str, dict[str, Any]] = {
    "hello": {"intent": "greeting", "entities": {}, "slots": {}},
    "1. loud": {
        "intent": "describe_property",
        "entities": {},
        "slots": {"property": "loud"},
    },
    "2. quiet": {
        "intent": "describe_property",
        "entities": {},
        "slots": {"property": "quiet"},
    },
    "3. happy": {
        "intent": "describe_property",
        "entities": {},
        "slots": {"property": "happy"},
    },
    "78. long": {
        "intent": "describe_property",
        "entities": {},
        "slots": {"property": "long"},
    },
    "79. short": {
        "intent": "describe_property",
        "entities": {},
        "slots": {"property": "short"},
    },
    "83. big large": {
        "intent": "describe_property",
        "entities": {},
        "slots": {"property": "large"},
    },
    "84. small little": {
        "intent": "describe_property",
        "entities": {},
        "slots": {"property": "small"},
    },
    "87. hot": {
        "intent": "describe_property",
        "entities": {},
        "slots": {"property": "hot"},
    },
    "91. new": {
        "intent": "describe_property",
        "entities": {},
        "slots": {"property": "new"},
    },
    "94. good": {
        "intent": "describe_property",
        "entities": {},
        "slots": {"property": "good"},
    },
    "97. dry": {
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
            "mapping_status": "unmapped",
            "source_gloss": normalized_gloss,
        }

    return {
        **meaning,
        "mapping_status": "mapped",
        "source_gloss": normalized_gloss,
    }