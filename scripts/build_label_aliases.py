import argparse
import json
import re
from pathlib import Path


def aliases_for_label(label):
    canonical = label.strip().lower()
    without_number = re.sub(r"^\s*\d+\s*[.):-]?\s*", "", canonical)
    variants = {canonical, without_number}
    for value in tuple(variants):
        variants.add(re.sub(r"[_-]+", " ", value))
        variants.add(re.sub(r"[^\w\s]", " ", value))
    variants.update(token for token in re.findall(r"[\w]+", without_number) if token)
    return sorted(value.strip() for value in variants if value.strip())


def build_aliases(label_map):
    ordered_labels = [label for label, _ in sorted(label_map.items(), key=lambda item: item[1])]
    canonical_to_aliases = {label: aliases_for_label(label) for label in ordered_labels}
    alias_to_canonical = {}
    for label, aliases in canonical_to_aliases.items():
        for alias in aliases:
            alias_to_canonical.setdefault(alias, []).append(label)
    return {
        "source": "label_mapping.json",
        "alias_policy": "Mechanical spelling and label-format variants only; no semantic synonyms are inferred.",
        "canonical_to_aliases": canonical_to_aliases,
        "alias_to_canonical": dict(sorted(alias_to_canonical.items())),
        "ambiguous_aliases": {
            alias: glosses for alias, glosses in sorted(alias_to_canonical.items()) if len(glosses) > 1
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Generate conservative label aliases from a checkpoint label mapping.")
    parser.add_argument("--label-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    label_map = json.loads(args.label_map.read_text(encoding="utf-8"))
    output = args.output or args.label_map.with_name("synonyms.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_aliases(label_map), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote aliases for {len(label_map)} labels to {output}")


if __name__ == "__main__":
    main()
