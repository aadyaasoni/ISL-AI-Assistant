import argparse
import json
from pathlib import Path


SUPPORTED_MODEL_SUFFIXES = {".glb", ".gltf"}


def validate_manifest(manifest_path: Path) -> list[str]:
    errors = []
    with manifest_path.open(encoding="utf-8") as file:
        manifest = json.load(file)
    asset_root = manifest_path.parent
    status = manifest.get("status")

    if status == "pending-assets":
        if manifest.get("model") is not None or manifest.get("animations"):
            errors.append("pending-assets manifest must not declare playable assets")
        if manifest.get("license") is not None:
            errors.append("pending-assets manifest must not claim a license")
        return errors

    if status == "skeleton-avatar":
        clip_index = manifest.get("clip_index")
        index_path = (asset_root / clip_index).resolve() if isinstance(clip_index, str) else None
        if (
            index_path is None
            or asset_root.resolve() not in index_path.parents
            or not index_path.is_file()
        ):
            errors.append("skeleton-avatar manifest requires an existing clip index")
        else:
            with index_path.open(encoding="utf-8") as file:
                index = json.load(file)
            clips = index.get("gloss_to_clip")
            if not isinstance(clips, dict):
                errors.append("skeleton-avatar clip index requires a gloss_to_clip mapping")
            else:
                supported = manifest.get("supported_glosses")
                if not isinstance(supported, list) or set(supported) != set(clips):
                    errors.append("skeleton-avatar supported glosses must match the clip index")
                for filename in clips.values():
                    clip_path = (asset_root / "clips" / filename).resolve()
                    if asset_root.resolve() not in clip_path.parents or not clip_path.is_file():
                        errors.append(f"missing or invalid skeleton clip: {filename}")
        if manifest.get("model") is not None or manifest.get("animations"):
            errors.append("skeleton-avatar manifest must not claim rigged 3D assets")
        if manifest.get("license") is not None:
            errors.append("skeleton-avatar manifest must not claim a dataset license")
        return errors

    if status != "ready":
        errors.append("status must be pending-assets, skeleton-avatar, or ready")
        return errors

    model = manifest.get("model")
    if not isinstance(model, str) or Path(model).suffix.lower() not in SUPPORTED_MODEL_SUFFIXES:
        errors.append("ready manifest requires a .glb or .gltf model path")
    elif not (asset_root / model).is_file():
        errors.append(f"missing avatar model: {model}")

    animations = manifest.get("animations")
    if not isinstance(animations, list) or not animations:
        errors.append("ready manifest requires at least one animation")
    else:
        names = set()
        for animation in animations:
            if not isinstance(animation, dict) or not animation.get("name") or not animation.get("path"):
                errors.append("each animation requires a name and path")
                continue
            if animation["name"] in names:
                errors.append(f"duplicate animation name: {animation['name']}")
            names.add(animation["name"])
            if not (asset_root / animation["path"]).is_file():
                errors.append(f"missing animation clip: {animation['path']}")

    license_metadata = manifest.get("license")
    if not isinstance(license_metadata, dict) or not all(
        isinstance(license_metadata.get(key), str) and license_metadata[key].strip()
        for key in ("source", "name", "redistribution")
    ):
        errors.append("ready manifest requires license source, name, and redistribution metadata")
    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate avatar asset manifest and licensed files.")
    parser.add_argument("--manifest", type=Path, default=Path("avatar/asset_manifest.json"))
    args = parser.parse_args()
    errors = validate_manifest(args.manifest)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print(f"Avatar manifest valid: {args.manifest}")


if __name__ == "__main__":
    main()
