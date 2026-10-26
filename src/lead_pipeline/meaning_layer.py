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


def sentence_for_glosses(glosses: list[str]) -> str:
    """Build a plain sentence that explicitly remains grounded in recognized words."""
    if not glosses:
        raise ValueError("at least one gloss is required")
    properties = []
    for gloss in glosses:
        meaning = meaning_for_gloss(gloss)
        if meaning["mapping_status"] != "mapped" or meaning["intent"] != "describe_property":
            raise ValueError(f"Gloss has no supported property meaning: {gloss}")
        properties.append(meaning["slots"]["property"])
    if len(properties) == 1:
        words = properties[0]
    else:
        words = f"{', '.join(properties[:-1])} and {properties[-1]}"
    noun = "word" if len(properties) == 1 else "words"
    verb = "is" if len(properties) == 1 else "are"
    return f"The recognized {noun} {verb} {words}."


def meaning_for_gloss_sequence(glosses: list[str]) -> dict[str, Any]:
    normalized = [gloss.strip().lower() for gloss in glosses]
    if not normalized:
        raise ValueError("at least one gloss is required")
    for gloss in normalized:
        meaning = meaning_for_gloss(gloss)
        if meaning["mapping_status"] != "mapped" or meaning["intent"] != "describe_property":
            raise ValueError(f"Gloss has no supported property meaning: {gloss}")
    return {
        "intent": "recognized_words",
        "entities": {},
        "slots": {"glosses": normalized},
        "mapping_status": "mapped",
        "source_glosses": normalized,
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