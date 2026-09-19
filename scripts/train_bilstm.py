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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset


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


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_classes):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, features, frame_mask, lengths):
        features = features * frame_mask.unsqueeze(-1)
        packed = pack_padded_sequence(
            features,
            lengths.cpu().clamp_min(1),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.encoder(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        valid_mask = frame_mask[:, : output.shape[1]].bool().unsqueeze(-1)
        output = output.masked_fill(~valid_mask, torch.finfo(output.dtype).min)
        pooled = output.max(dim=1).values
        return self.classifier(self.dropout(pooled))


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
    with torch.no_grad():
        for features, frame_mask, lengths, batch_labels, _ in loader:
            logits = model(features.to(device), frame_mask.to(device), lengths)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(batch_labels.tolist())
    report = classification_report(
        targets,
        predictions,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    return targets, predictions, report


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
    test_targets, test_predictions, _ = evaluate(model, loaders["test"], device, labels)
    test_support = Counter(test_targets)
    evaluated_indices = [index for index in range(len(labels)) if test_support[index] > 0]
    evaluated_labels = [labels[index] for index in evaluated_indices]
    test_report = classification_report(
        test_targets,
        test_predictions,
        labels=evaluated_indices,
        target_names=evaluated_labels,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "device": str(device),
        "checkpoint_selection_metric": "validation_macro_f1",
        "test_used_during_training_or_checkpoint_selection": False,
        "best_validation_macro_f1": best_val_f1,
        "test_macro_f1": test_report["macro avg"]["f1-score"],
        "test_weighted_f1": test_report["weighted avg"]["f1-score"],
        "test_samples": len(test_targets),
        "per_class": {
            label: test_report[label] if index in evaluated_indices else {
                "evaluated": False,
                "support": 0,
                "note": "not evaluated on test: original test split contains 0 samples",
            }
            for index, label in enumerate(labels)
        },
        "test_evaluation_notes": {
            "97. dry": "not evaluated on test: original test split contains 0 samples",
        },
        "split_counts": {split: dict(Counter(row["label"] for row in split_rows[split])) for split in split_rows},
        "history": history,
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    with (args.output_dir / "metrics.txt").open("w", encoding="utf-8") as file:
        file.write(classification_report(test_targets, test_predictions, labels=evaluated_indices, target_names=evaluated_labels, zero_division=0))
        file.write("\n97. dry: not evaluated on test because the original test split has 0 samples.\n")
    print(f"best_validation_macro_f1={best_val_f1:.4f}")
    print(f"test_macro_f1={metrics['test_macro_f1']:.4f}")
    print(f"checkpoint={checkpoint_path}")


if __name__ == "__main__":
    main()