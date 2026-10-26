import json
from collections import deque
from pathlib import Path
from typing import Any


def read_failure_events(path: Path, limit: int = 100) -> list[dict[str, Any]]:
    if limit < 1:
        raise ValueError("limit must be positive")
    try:
        with path.open(encoding="utf-8") as log_file:
            events = deque(
                (json.loads(line) for line in log_file if line.strip()),
                maxlen=limit,
            )
    except FileNotFoundError:
        return []
    return list(events)
