import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import torch
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader

from train_bilstm import BiLSTMClassifier, LandmarkDataset, collate_batch, read_metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a saved BiLSTM checkpoint on exactly one split.")
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--metadata", type=Path, default=Path("data/include_metadata/metadata.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("training_outputs/bilstm_baseline/best_model.pt"))
    parser.add_argument("--label-map", type=Path, default=Path("training_outputs/bilstm_baseline/label_mapping.json"))
    return parser.parse_args()


def main():
    args = parse_args()
    with args.label_map.open(encoding="utf-8") as file:
        label_to_index = json.load(file)
    labels = [label for label, _ in sorted(label_to_index.items(), key=lambda item: item[1])]
    rows = [row for row in read_metadata(args.metadata) if row["split"] == args.split]
    dataset = LandmarkDataset(rows, label_to_index)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, collate_fn=collate_batch, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    config = checkpoint["config"]
    model = BiLSTMClassifier(
        checkpoint["feature_dimension"],
        int(config["hidden_size"]),
        int(config["num_layers"]),
        float(config["dropout"]),
        len(labels),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    targets = []
    predictions = []
    with torch.no_grad():
        for features, frame_mask, lengths, batch_labels, _ in loader:
            logits = model(features.to(device), frame_mask.to(device), lengths)
            targets.extend(batch_labels.tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    support = Counter(targets)
    evaluated_indices = [index for index in range(len(labels)) if support[index] > 0]
    evaluated_labels = [labels[index] for index in evaluated_indices]
    report = classification_report(
        targets,
        predictions,
        labels=evaluated_indices,
        target_names=evaluated_labels,
        output_dict=True,
        zero_division=0,
    )
    output = {
        "split": args.split,
        "samples": len(rows),
        "evaluated_classes": evaluated_labels,
        "macro_f1": f1_score(targets, predictions, labels=evaluated_indices, average="macro", zero_division=0),
        "per_class": {label: report[label] for label in evaluated_labels},
        "excluded_zero_support_classes": [labels[index] for index in range(len(labels)) if support[index] == 0],
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()