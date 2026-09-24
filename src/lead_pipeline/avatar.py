from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field


ClipLoader = Callable[[str], str]


@dataclass
class GlossClipResolver:
    """Resolve fixed-vocabulary glosses to animation clips with caching."""

    mapping: Mapping[str, str]
    clip_loader: ClipLoader
    _cache: dict[str, str] = field(default_factory=dict, init=False)

    def resolve(self, gloss_sequence: Sequence[str]) -> list[str]:
        clips = []
        for gloss in gloss_sequence:
            normalized_gloss = gloss.strip().lower()
            if not normalized_gloss:
                raise ValueError("gloss sequence cannot contain empty glosses")
            clip_path = self.mapping.get(normalized_gloss)
            if clip_path is None:
                raise KeyError(f"No animation clip mapped for gloss: {normalized_gloss}")
            if normalized_gloss not in self._cache:
                self._cache[normalized_gloss] = self.clip_loader(clip_path)
            clips.append(self._cache[normalized_gloss])
        return clips

    @property
    def cache_size(self) -> int:
        return len(self._cache)