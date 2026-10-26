import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


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
