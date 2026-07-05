"""
Transfer learning model: fine-tunes a pretrained DistilBERT
(distilbert-base-uncased) for 6-way emotion classification using the
Hugging Face Trainer API.

Run on Google Colab with a GPU runtime:
    python src/train_transformer.py --csv_path data/emotion-dataset.csv --epochs 2
"""

import argparse
import json
import os

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from data import LABEL_NAMES, ID2LABEL, LABEL2ID, load_emotion_data, split_data

MODEL_NAME = "distilbert-base-uncased"


def to_hf_dataset(df):
    return Dataset.from_dict({"text": df["text_clean"].tolist(), "label": df["label"].tolist()})


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def run(csv_path, sample_frac, epochs, batch_size, max_len, lr, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    df = load_emotion_data(csv_path, sample_frac=sample_frac)
    train, val, test = split_data(df)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len)

    train_ds = to_hf_dataset(train).map(tokenize, batched=True)
    val_ds = to_hf_dataset(val).map(tokenize, batched=True)
    test_ds = to_hf_dataset(test).map(tokenize, batched=True)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_NAMES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=100,
        report_to=[],
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    preds_output = trainer.predict(test_ds)
    preds = np.argmax(preds_output.predictions, axis=1)
    labels = preds_output.label_ids

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro")
    print(f"\nTest accuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f}")
    print(classification_report(labels, preds, target_names=LABEL_NAMES))

    report = classification_report(labels, preds, target_names=LABEL_NAMES, output_dict=True)
    cm = confusion_matrix(labels, preds).tolist()

    trainer.save_model(os.path.join(out_dir, "final_model"))
    tokenizer.save_pretrained(os.path.join(out_dir, "final_model"))

    metrics = {"test_accuracy": acc, "test_macro_f1": macro_f1, "report": report, "confusion_matrix": cm}
    with open(os.path.join(out_dir, "transformer_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="data/emotion-dataset.csv")
    parser.add_argument("--sample_frac", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--out_dir", default="artifacts/transformer")
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
