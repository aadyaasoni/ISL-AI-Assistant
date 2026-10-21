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

    if status != "ready":
        errors.append("status must be either pending-assets or ready")
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
