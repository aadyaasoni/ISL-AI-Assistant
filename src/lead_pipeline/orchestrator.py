from typing import Any

from .conversation_agent import ConversationState
from .meaning_layer import meaning_for_gloss


class Orchestrator:
    def __init__(self, confidence_threshold: float = 0.60) -> None:
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.confidence_threshold = confidence_threshold
        self.conversation = ConversationState()

    def route(self, prediction: dict[str, Any]) -> dict[str, Any]:
        self._validate_prediction(prediction)
        gloss = prediction["gloss"].strip().lower()
        confidence = float(prediction["confidence"])
        timestamp = float(prediction["timestamp"])

        if confidence < self.confidence_threshold:
            return {
                "status": "clarification_required",
                "reason": "low_confidence",
                "message": "I am not confident I understood that sign. Please repeat it.",
                "confidence": confidence,
                "timestamp": timestamp,
            }

        meaning = meaning_for_gloss(gloss)
        if meaning["intent"] == "unknown":
            return {
                "status": "clarification_required",
                "reason": "unknown_gloss",
                "message": "I do not recognize that sign yet. Please repeat it.",
                "confidence": confidence,
                "timestamp": timestamp,
            }

        response_text = self.conversation.response_for(meaning)
        self.conversation.add_turn(meaning, response_text)
        return {
            "status": "accepted",
            "meaning": meaning,
            "response_text": response_text,
            "confidence": confidence,
            "timestamp": timestamp,
        }

    @staticmethod
    def _validate_prediction(prediction: dict[str, Any]) -> None:
        required = {"gloss", "confidence", "timestamp"}
        missing = required.difference(prediction)
        if missing:
            raise ValueError(f"Missing prediction fields: {sorted(missing)}")
        if not isinstance(prediction["gloss"], str) or not prediction["gloss"].strip():
            raise ValueError("gloss must be a non-empty string")
        confidence = float(prediction["confidence"])
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")