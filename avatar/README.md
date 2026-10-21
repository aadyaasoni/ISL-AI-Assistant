# Avatar Assets

The avatar directory is reserved for the communication-layer presentation assets. No binary assets are committed yet.

## Required package

- `avatar.glb` or `avatar.gltf`: the rigged avatar model
- `idle` animation: default resting pose
- Sign animation clips keyed by the model gloss labels
- `ATTRIBUTION.md`: source, license, and redistribution terms

## Integration requirements

- Use a web-compatible glTF/GLB model.
- Keep animation names stable and document them in an asset manifest.
- Preserve a neutral idle pose and visible hands.
- Do not bundle raw dataset videos or landmark files as avatar assets.
- Map animation names to the model labels in a separate manifest rather than hardcoding paths in recognition code.

Once the model and clips are available, add an `asset_manifest.json` containing the model path, animation names, supported glosses, and license metadata.

Run `python3 scripts/validate_avatar_assets.py` before enabling playback. It rejects missing files, duplicate animation names, unsupported model formats, and incomplete license metadata. The current manifest intentionally validates as `pending-assets`.
