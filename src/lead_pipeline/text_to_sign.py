import re


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


def text_to_sign(text: str) -> dict[str, list[str]]:
    """Map supported words to dataset gloss labels without external inference."""
    if not isinstance(text, str):
        raise TypeError("text must be a string")

    tokens = re.findall(r"[a-z0-9]+", text.lower())
    glosses = []
    unmatched = []
    position = 0
    while position < len(tokens):
        match = next(
            (
                alias
                for alias in _ALIASES_BY_LENGTH
                if tuple(tokens[position : position + len(alias)]) == alias
            ),
            None,
        )
        if match is None:
            unmatched.append(tokens[position])
            position += 1
            continue
        glosses.append(_ALIASES[match])
        position += len(match)

    return {"glosses": glosses, "unmatched": unmatched}
