"""
From-scratch deep learning model: an Embedding + BiLSTM classifier
implemented directly in PyTorch (no pretrained weights). This is the
"core deep learning" building block of the capstone, contrasted with the
TF-IDF baseline and the fine-tuned transformer.
"""

import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids, lengths=None):
        # input_ids: (batch, seq_len)
        embedded = self.embedding(input_ids)  # (batch, seq_len, embed_dim)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (h_n, _) = self.lstm(packed)
        else:
            _, (h_n, _) = self.lstm(embedded)

        # h_n: (num_layers * num_directions, batch, hidden_dim)
        # concat last layer's forward and backward hidden states
        h_forward = h_n[-2]
        h_backward = h_n[-1]
        h_cat = torch.cat([h_forward, h_backward], dim=1)  # (batch, hidden_dim*2)

        out = self.dropout(h_cat)
        logits = self.fc(out)  # (batch, num_classes)
        return logits
