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

    def __post_init__(self) -> None:
        if self.failure_log_path.is_absolute():
            raise ValueError("failure_log_path must be relative")
