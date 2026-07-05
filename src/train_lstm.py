"""
Trains the from-scratch BiLSTM emotion classifier.

Run on Google Colab (GPU strongly recommended):
    python src/train_lstm.py --csv_path data/emotion-dataset.csv --epochs 6
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from data import LABEL_NAMES, load_emotion_data, split_data
from vocab import Vocab
from models import BiLSTMClassifier


class EmotionDataset(Dataset):
    def __init__(self, texts, labels, vocab: Vocab, max_len: int = 40):
        self.texts = list(texts)
        self.labels = list(labels)
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        length = min(len(text.split()), self.max_len) or 1
        ids = self.vocab.encode(text, self.max_len)
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(self.labels[idx], dtype=torch.long),
        )


def evaluate(model, loader, device, criterion):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for ids, lengths, labels in loader:
            ids, lengths, labels = ids.to(device), lengths.to(device), labels.to(device)
            logits = model(ids, lengths)
            loss = criterion(logits, labels)
            total_loss += loss.item() * ids.size(0)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    avg_loss = total_loss / len(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, macro_f1, all_preds, all_labels


def run(csv_path, sample_frac, epochs, batch_size, max_len, lr, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = load_emotion_data(csv_path, sample_frac=sample_frac)
    train, val, test = split_data(df)

    vocab = Vocab(train["text_clean"], min_freq=2, max_size=30000)
    print(f"Vocab size: {len(vocab)}")

    train_ds = EmotionDataset(train["text_clean"], train["label"], vocab, max_len)
    val_ds = EmotionDataset(val["text_clean"], val["label"], vocab, max_len)
    test_ds = EmotionDataset(test["text_clean"], test["label"], vocab, max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = BiLSTMClassifier(
        vocab_size=len(vocab), num_classes=len(LABEL_NAMES), pad_idx=vocab.pad_id
    ).to(device)

    # class weights to counter imbalance (surprise/love are rare)
    class_counts = train["label"].value_counts().sort_index().values
    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_macro_f1": []}
    best_val_f1 = -1
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for ids, lengths, labels in train_loader:
            ids, lengths, labels = ids.to(device), lengths.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(ids, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * ids.size(0)

        train_loss = total_loss / len(train_ds)
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device, criterion)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)

        print(
            f"Epoch {epoch}/{epochs} | train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_loss, test_acc, test_f1, preds, labels = evaluate(model, test_loader, device, criterion)
    print(f"\nTest accuracy: {test_acc:.4f} | Macro-F1: {test_f1:.4f}")
    print(classification_report(labels, preds, target_names=LABEL_NAMES))

    report = classification_report(labels, preds, target_names=LABEL_NAMES, output_dict=True)
    cm = confusion_matrix(labels, preds).tolist()

    torch.save(model.state_dict(), os.path.join(out_dir, "bilstm_model.pt"))
    with open(os.path.join(out_dir, "vocab.json"), "w") as f:
        json.dump(vocab.itos, f)
    metrics = {
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "report": report,
        "confusion_matrix": cm,
        "history": history,
    }
    with open(os.path.join(out_dir, "lstm_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="data/emotion-dataset.csv")
    parser.add_argument("--sample_frac", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_len", type=int, default=40)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out_dir", default="artifacts/lstm")
    args = parser.parse_args()
    run(
        args.csv_path,
        args.sample_frac,
        args.epochs,
        args.batch_size,
        args.max_len,
        args.lr,
        args.out_dir,
    )
