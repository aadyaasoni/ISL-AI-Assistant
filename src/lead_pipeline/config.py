from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OrchestratorConfig:
    failure_log_path: Path = Path("reports/failure_events.jsonl")

    def __post_init__(self) -> None:
        if self.failure_log_path.is_absolute():
            raise ValueError("failure_log_path must be relative")
