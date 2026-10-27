import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

try:
    from .bilstm_model import BiLSTMClassifier
except ImportError:
    from bilstm_model import BiLSTMClassifier

FEATURE_DIMENSION = 258
SEED = 42


class LandmarkDataset(Dataset):
    def __init__(self, rows, label_to_index):
        self.rows = rows
        self.label_to_index = label_to_index

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with np.load(row["processed_file_path"]) as sample:
            features = sample["features"].astype(np.float32)
            landmark_mask = sample["mask"].astype(np.float32)
        frame_mask = (landmark_mask.sum(axis=1) > 0).astype(np.float32)
        return {
            "sample_id": row["sample_id"],
            "features": torch.from_numpy(features),
            "frame_mask": torch.from_numpy(frame_mask),
            "label": self.label_to_index[row["label"]],
        }


def collate_batch(batch):
    lengths = torch.tensor([item["features"].shape[0] for item in batch], dtype=torch.long)
    max_length = int(lengths.max())
    features = torch.zeros(len(batch), max_length, FEATURE_DIMENSION, dtype=torch.float32)
    frame_mask = torch.zeros(len(batch), max_length, dtype=torch.float32)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    sample_ids = []
    for index, item in enumerate(batch):
        length = item["features"].shape[0]
        features[index, :length] = item["features"]
        frame_mask[index, :length] = item["frame_mask"]
        sample_ids.append(item["sample_id"])
    return features, frame_mask, lengths, labels, sample_ids


def read_metadata(metadata_path):
    with metadata_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    rows = [row for row in rows if row["valid"] == "True"]
    return rows


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, criterion, optimizer, device, training):
    model.train(training)
    total_loss = 0.0
    predictions = []
    targets = []
    for features, frame_mask, lengths, labels, _ in loader:
        features = features.to(device)
        frame_mask = frame_mask.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(training):
            logits = model(features, frame_mask, lengths)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())
    macro_f1 = f1_score(targets, predictions, average="macro", zero_division=0)
    return total_loss / len(loader.dataset), macro_f1, targets, predictions


def evaluate(model, loader, device, labels):
    model.eval()
    targets = []
    predictions = []
    probabilities = []
    with torch.no_grad():
        for features, frame_mask, lengths, batch_labels, _ in loader:
            logits = model(features.to(device), frame_mask.to(device), lengths)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(batch_labels.tolist())
            probabilities.extend(torch.softmax(logits, dim=1).cpu().tolist())
    report = classification_report(
        targets,
        predictions,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    return targets, predictions, report, np.asarray(probabilities, dtype=np.float64)


def calibration_metrics(targets, probabilities, bins=15):
    """Return top-label ECE and multiclass Brier score as evaluation diagnostics."""
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] != len(targets):
        raise ValueError("probabilities must have shape (sample_count, class_count)")
    if not len(targets):
        raise ValueError("calibration requires at least one sample")
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = (predictions == targets).astype(np.float64)
    ece = 0.0
    bin_rows = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        if index == bins - 1:
            selected = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            selected = (confidence >= edges[index]) & (confidence < edges[index + 1])
        count = int(selected.sum())
        if not count:
            continue
        accuracy = float(correctness[selected].mean())
        mean_confidence = float(confidence[selected].mean())
        ece += count / len(targets) * abs(accuracy - mean_confidence)
        bin_rows.append({"lower": float(edges[index]), "upper": float(edges[index + 1]), "count": count, "accuracy": accuracy, "mean_confidence": mean_confidence})
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[targets]
    brier = float(np.square(probabilities - one_hot).sum(axis=1).mean())
    return {"method": "top-label ECE with equal-width bins", "bins": bins, "ece": float(ece), "multiclass_brier_score": brier, "bin_details": bin_rows}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a padded variable-length BiLSTM on INCLUDE landmarks.")
    parser.add_argument("--metadata", type=Path, default=Path("data/include_metadata/metadata.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("training_outputs/bilstm_baseline"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--patience", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_metadata(args.metadata)
    labels = sorted({row["label"] for row in rows})
    label_to_index = {label: index for index, label in enumerate(labels)}
    with (args.output_dir / "label_mapping.json").open("w", encoding="utf-8") as file:
        json.dump(label_to_index, file, indent=2)

    split_rows = {split: [row for row in rows if row["split"] == split] for split in ("train", "val", "test")}
    datasets = {split: LandmarkDataset(split_rows[split], label_to_index) for split in split_rows}
    loaders = {
        split: DataLoader(
            datasets[split],
            batch_size=args.batch_size,
            shuffle=split == "train",
            collate_fn=collate_batch,
            num_workers=0,
        )
        for split in split_rows
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMClassifier(
        FEATURE_DIMENSION,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        len(labels),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    history = []
    best_val_f1 = -1.0
    epochs_without_improvement = 0
    checkpoint_path = args.output_dir / "best_model.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss, train_f1, _, _ = run_epoch(model, loaders["train"], criterion, optimizer, device, True)
        val_loss, val_f1, _, _ = run_epoch(model, loaders["val"], criterion, optimizer, device, False)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_macro_f1": train_f1, "val_loss": val_loss, "val_macro_f1": val_f1})
        print(f"epoch={epoch:03d} train_loss={train_loss:.4f} train_macro_f1={train_f1:.4f} val_loss={val_loss:.4f} val_macro_f1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0
            checkpoint_config = {key: str(value) for key, value in vars(args).items()}
            torch.save({"model_state_dict": model.state_dict(), "labels": labels, "feature_dimension": FEATURE_DIMENSION, "config": checkpoint_config}, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    val_targets, val_predictions, val_report, val_probabilities = evaluate(model, loaders["val"], device, labels)
    test_targets, test_predictions, test_report, test_probabilities = evaluate(model, loaders["test"], device, labels)
    metrics = {
        "device": str(device),
        "checkpoint_selection_metric": "validation_macro_f1",
        "test_used_during_training_or_checkpoint_selection": False,
        "best_validation_macro_f1": best_val_f1,
        "validation_macro_f1": val_report["macro avg"]["f1-score"],
        "test_macro_f1": test_report["macro avg"]["f1-score"],
        "test_weighted_f1": test_report["weighted avg"]["f1-score"],
        "validation_samples": len(val_targets),
        "test_samples": len(test_targets),
        "per_class_validation": {label: val_report[label] for label in labels},
        "per_class_test": {label: test_report[label] for label in labels},
        "per_class": {label: test_report[label] for label in labels},
        "confidence_calibration": {
            "validation": calibration_metrics(val_targets, val_probabilities),
            "test": calibration_metrics(test_targets, test_probabilities),
            "note": "Diagnostic only; no post-hoc calibration parameters are fit or applied.",
        },
        "split_counts": {split: dict(Counter(row["label"] for row in split_rows[split])) for split in split_rows},
        "history": history,
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    with (args.output_dir / "metrics.txt").open("w", encoding="utf-8") as file:
        file.write("Validation per-class report\n")
        file.write(classification_report(val_targets, val_predictions, labels=list(range(len(labels))), target_names=labels, zero_division=0))
        file.write("\nTest per-class report\n")
        file.write(classification_report(test_targets, test_predictions, labels=list(range(len(labels))), target_names=labels, zero_division=0))
        file.write("\nConfidence calibration (ECE / multiclass Brier)\n")
        file.write(json.dumps(metrics["confidence_calibration"], indent=2) + "\n")
    print(f"best_validation_macro_f1={best_val_f1:.4f}")
    print(f"test_macro_f1={metrics['test_macro_f1']:.4f}")
    print(f"checkpoint={checkpoint_path}")


if __name__ == "__main__":
    main()