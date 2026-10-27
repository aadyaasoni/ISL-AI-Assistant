import json
import os
import time
from pathlib import Path

import numpy as np
import torch

try:
    from .bilstm_model import BiLSTMClassifier
except ImportError:
    from bilstm_model import BiLSTMClassifier


def resolve_checkpoint_paths(checkpoint_path=None, label_map_path=None):
    default_dir = Path(__file__).resolve().parents[1] / "training_outputs/bilstm_baseline"
    checkpoint_dir = Path(os.environ.get("ISL_CHECKPOINT_DIR", default_dir))
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else checkpoint_dir / "best_model.pt"
    label_map_path = Path(label_map_path) if label_map_path else checkpoint_path.parent / "label_mapping.json"
    return checkpoint_path, label_map_path


class RecognitionAdapter:
    """Load the baseline recognizer and expose the integration contract."""

    def __init__(self, checkpoint_path=None, label_map_path=None, device=None):
        self.checkpoint_path, self.label_map_path = resolve_checkpoint_paths(checkpoint_path, label_map_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        with self.label_map_path.open(encoding="utf-8") as file:
            label_to_index = json.load(file)
        self.labels = [label for label, _ in sorted(label_to_index.items(), key=lambda item: item[1])]
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        config = checkpoint["config"]
        self.model = BiLSTMClassifier(
            checkpoint["feature_dimension"],
            int(config["hidden_size"]),
            int(config["num_layers"]),
            float(config["dropout"]),
            len(self.labels),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, features, mask, timestamp=None):
        """Return gloss, confidence, and timestamp from one landmark sequence."""
        probabilities = self._predict_probabilities(features, mask)
        class_id = int(probabilities.argmax().item())
        return {
            "gloss": self.labels[class_id],
            "confidence": float(probabilities[class_id].item()),
            "timestamp": float(time.time() if timestamp is None else timestamp),
        }

    def predict_with_candidates(self, features, mask, timestamp=None):
        """Return the existing prediction contract plus the best two labels."""
        probabilities = self._predict_probabilities(features, mask)
        class_ids = probabilities.topk(min(2, len(self.labels))).indices.tolist()
        prediction = {
            "gloss": self.labels[class_ids[0]],
            "confidence": float(probabilities[class_ids[0]].item()),
            "timestamp": float(time.time() if timestamp is None else timestamp),
        }
        candidates = [
            {
                "gloss": self.labels[class_id],
                "confidence": float(probabilities[class_id].item()),
            }
            for class_id in class_ids
        ]
        return prediction, candidates

    def _predict_probabilities(self, features, mask):
        features = np.asarray(features, dtype=np.float32)
        mask = np.asarray(mask, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != 258:
            raise ValueError("features must have shape (T, 258)")
        if mask.shape != (features.shape[0], 75):
            raise ValueError("mask must have shape (T, 75)")
        frame_mask = (mask.sum(axis=1) > 0).astype(np.float32)
        feature_tensor = torch.from_numpy(features).unsqueeze(0).to(self.device)
        mask_tensor = torch.from_numpy(frame_mask).unsqueeze(0).to(self.device)
        lengths = torch.tensor([features.shape[0]], dtype=torch.long)
        with torch.no_grad():
            logits = self.model(feature_tensor, mask_tensor, lengths)
            return torch.softmax(logits, dim=-1)[0]