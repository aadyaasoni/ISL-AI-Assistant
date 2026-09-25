from typing import Any, Optional, Protocol

from .orchestrator import Orchestrator


class Recognizer(Protocol):
    def predict(
        self,
        features: Any,
        mask: Any,
        timestamp: float,
    ) -> dict[str, Any]:
        """Return the recognition contract from CONTRACT.md."""


class RecognitionRuntime:
    """Route real recognizer output into the lead-owned communication flow."""

    def __init__(
        self,
        recognizer: Recognizer,
        orchestrator: Optional[Orchestrator] = None,
    ) -> None:
        self.recognizer = recognizer
        self.orchestrator = orchestrator or Orchestrator()

    def process(
        self,
        features: Any,
        mask: Any,
        timestamp: float,
    ) -> dict[str, Any]:
        prediction = self.recognizer.predict(features, mask, timestamp)
        return self.orchestrator.route(prediction)