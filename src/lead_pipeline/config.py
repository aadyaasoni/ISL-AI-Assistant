import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class OrchestratorConfig:
    failure_log_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("ISL_FAILURE_LOG_PATH", "reports/failure_events.jsonl")
        )
    )
