from typing import Any, Protocol

from .runtime import RecognitionRuntime


class ClipResolver(Protocol):
    def resolve(self, gloss_sequence: list[str]) -> list[str]:
        """Resolve glosses to playable clip handles."""


class CommunicationPipeline:
    """Coordinate recognition, conversation routing, and avatar preparation."""

    def __init__(self, runtime: RecognitionRuntime, clip_resolver: ClipResolver) -> None:
        self.runtime = runtime
        self.clip_resolver = clip_resolver

    def process(self, features: Any, mask: Any, timestamp: float) -> dict[str, Any]:
        result = self.runtime.process(features, mask, timestamp)
        if result["status"] != "accepted":
            return {**result, "avatar_status": "not_requested", "avatar_clips": []}

        gloss = result["meaning"]["source_gloss"]
        try:
            clips = self.clip_resolver.resolve([gloss])
        except KeyError:
            return {
                **result,
                "avatar_status": "fallback_required",
                "avatar_clips": [],
            }

        return {**result, "avatar_status": "ready", "avatar_clips": clips}