# Avatar Playback

The browser displays a lightweight 2D stick-figure animation made from the extracted landmark samples. The gloss-to-clip index is `clips/index.json`; source attribution and the unresolved dataset redistribution question are documented in `ATTRIBUTION.md`.

## Current skeleton mode

- `clips/<label>.json`: one downsampled normalized landmark sequence per gloss
- `clips/index.json`: dataset gloss to clip filename mapping
- `asset_manifest.json`: declares `skeleton-avatar`, not a rigged or licensed 3D model

Regenerate clips from the checked-in processed landmarks with:

```bash
python scripts/build_avatar_clips.py
```

No raw video is copied into the clips. Check the dataset attribution and redistribution terms before redistributing generated skeleton data.

A rigged 3D avatar, licensed sign animation assets, and a renderer are future work. Validate the manifest with `python scripts/validate_avatar_assets.py`.
