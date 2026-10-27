import re
import json
import os
from functools import lru_cache
from pathlib import Path


_ALIASES: dict[tuple[str, ...], str] = {
    ("loud",): "1. loud",
    ("quiet",): "2. quiet",
    ("happy",): "3. happy",
    ("long",): "78. long",
    ("short",): "79. short",
    ("big", "large"): "83. big large",
    ("big",): "83. big large",
    ("large",): "83. big large",
    ("small", "little"): "84. small little",
    ("small",): "84. small little",
    ("little",): "84. small little",
    ("hot",): "87. hot",
    ("new",): "91. new",
    ("good",): "94. good",
    ("dry",): "97. dry",
}

_ALIASES_BY_LENGTH = sorted(_ALIASES, key=len, reverse=True)


@lru_cache(maxsize=8)
def _aliases_for_path(path: str):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _ALIASES
    alias_mapping = payload.get("alias_to_canonical")
    if not isinstance(alias_mapping, dict):
        return _ALIASES
    dynamic = {}
    for alias, glosses in alias_mapping.items():
        if len(glosses) != 1:
            continue
        tokens = tuple(re.findall(r"[a-z0-9]+", alias.lower()))
        if tokens:
            dynamic[tokens] = glosses[0]
    return dynamic


def text_to_sign(text: str) -> dict[str, list[str]]:
    """Map supported words to dataset gloss labels without external inference."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    checkpoint_dir = Path(os.environ.get("ISL_CHECKPOINT_DIR", "training_outputs/bilstm_baseline"))
    aliases_path = os.environ.get("ISL_SYNONYMS_PATH", str(checkpoint_dir / "synonyms.json"))
    aliases = _aliases_for_path(aliases_path)
    aliases_by_length = sorted(aliases, key=len, reverse=True)
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    glosses = []
    unmatched = []
    position = 0
    while position < len(tokens):
        match = next(
            (
                alias
                for alias in aliases_by_length
                if tuple(tokens[position : position + len(alias)]) == alias
            ),
            None,
        )
        if match is None:
            unmatched.append(tokens[position])
            position += 1
            continue
        glosses.append(aliases[match])
        position += len(match)

    return {"glosses": glosses, "unmatched": unmatched}
