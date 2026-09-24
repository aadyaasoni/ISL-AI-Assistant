from dataclasses import dataclass, field
from typing import Any, Callable


ResponseGenerator = Callable[[dict[str, Any], list[dict[str, Any]]], str]


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


class ConversationAgent:
    """Constrained response boundary with injectable LLM-compatible behavior."""

    SYSTEM_SCOPE = (
        "Respond only from the supplied semantic frame and recent conversation "
        "state. Do not invent signs, entities, or slots."
    )

    def __init__(
        self,
        state: ConversationState | None = None,
        response_generator: ResponseGenerator | None = None,
    ) -> None:
        self.state = state or ConversationState()
        self.response_generator = response_generator

    def respond(self, meaning: dict[str, Any]) -> str:
        if meaning.get("intent") == "unknown":
            response = "I do not have a meaning for that sign yet."
        elif self.response_generator is None:
            response = self.state.response_for(meaning)
        else:
            response = self.response_generator(meaning, list(self.state.turns))
            if not isinstance(response, str) or not response.strip():
                raise ValueError("response_generator must return non-empty text")

        self.state.add_turn(meaning, response)
        return response