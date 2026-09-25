from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Optional


ClipLoader = Callable[[str], str]


@dataclass
class GlossClipResolver:
    """Resolve fixed-vocabulary glosses to animation clips with caching."""

    mapping: Mapping[str, str]
    clip_loader: ClipLoader
    fallback_clip: Optional[str] = None
    _cache: dict[str, str] = field(default_factory=dict, init=False)

    def resolve(self, gloss_sequence: Sequence[str]) -> list[str]:
        clips = []
        for gloss in gloss_sequence:
            normalized_gloss = gloss.strip().lower()
            if not normalized_gloss:
                raise ValueError("gloss sequence cannot contain empty glosses")
            clip_path = self.mapping.get(normalized_gloss)
            if clip_path is None:
                if self.fallback_clip is None:
                    raise KeyError(f"No animation clip mapped for gloss: {normalized_gloss}")
                cache_key = "__fallback__"
                clip_path = self.fallback_clip
            else:
                cache_key = normalized_gloss
            if cache_key not in self._cache:
                self._cache[cache_key] = self.clip_loader(clip_path)
            clips.append(self._cache[cache_key])
        return clips

    @property
    def cache_size(self) -> int:
        return len(self._cache)