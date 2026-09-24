from dataclasses import dataclass, field
from typing import Any


@dataclass
class ConversationState:
    """State boundary for the future LLM-backed conversation agent."""

    max_turns: int = 6
    turns: list[dict[str, Any]] = field(default_factory=list)

    def add_turn(self, user_input: dict[str, Any], response_text: str) -> None:
        self.turns.append({"input": user_input, "response": response_text})
        self.turns = self.turns[-self.max_turns :]

    def response_for(self, meaning: dict[str, Any]) -> str:
        """Return a deterministic response until LLM integration is added."""
        intent = meaning["intent"]
        if intent == "greeting":
            return "Hello. How can I help you?"
        if intent == "describe_property":
            return f"I understood the property: {meaning['slots']['property']}."
        return "I do not have a meaning for that sign yet."