import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from train_bilstm import BiLSTMClassifier


def load_predictor(checkpoint_path, label_map_path, device=None):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    with label_map_path.open(encoding="utf-8") as file:
        label_to_index = json.load(file)
    labels = [label for label, _ in sorted(label_to_index.items(), key=lambda item: item[1])]
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
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
    return model, labels, device


def predict_npz(npz_path, model, labels, device, timestamp=None):
    with np.load(npz_path) as sample:
        features = sample["features"].astype(np.float32)
        landmark_mask = sample["mask"].astype(np.float32)
    frame_mask = (landmark_mask.sum(axis=1) > 0).astype(np.float32)
    feature_tensor = torch.from_numpy(features).unsqueeze(0).to(device)
    mask_tensor = torch.from_numpy(frame_mask).unsqueeze(0).to(device)
    lengths = torch.tensor([features.shape[0]], dtype=torch.long)
    with torch.no_grad():
        logits = model(feature_tensor, mask_tensor, lengths)
        probabilities = torch.softmax(logits, dim=-1)[0]
    class_id = int(probabilities.argmax().item())
    return {
        "gloss": labels[class_id],
        "confidence": float(probabilities[class_id].item()),
        "timestamp": float(time.time() if timestamp is None else timestamp),
        "class_id": class_id,
        "logits": logits[0].cpu().tolist(),
        "feature_dimension": int(features.shape[1]),
        "sequence_length": int(features.shape[0]),
    }


def main():
    parser = argparse.ArgumentParser(description="Run the BiLSTM and emit the recognition contract.")
    parser.add_argument("npz_path", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("training_outputs/bilstm_baseline/best_model.pt"))
    parser.add_argument("--label-map", type=Path, default=Path("training_outputs/bilstm_baseline/label_mapping.json"))
    parser.add_argument("--timestamp", type=float, default=None)
    args = parser.parse_args()
    model, labels, device = load_predictor(args.checkpoint, args.label_map)
    print(json.dumps(predict_npz(args.npz_path, model, labels, device, args.timestamp), indent=2))


if __name__ == "__main__":
    main()