"""
Baseline model: TF-IDF + Logistic Regression.

This is the "classical ML" reference point the deep learning models
(BiLSTM, fine-tuned DistilBERT) are compared against.
"""

import argparse
import json
import os

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from data import LABEL_NAMES, load_emotion_data, split_data


def run(csv_path: str, sample_frac: float, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    df = load_emotion_data(csv_path, sample_frac=sample_frac)
    train, val, test = split_data(df)

    vectorizer = TfidfVectorizer(
        max_features=30000, ngram_range=(1, 2), sublinear_tf=True, min_df=2
    )
    X_train = vectorizer.fit_transform(train["text_clean"])
    X_test = vectorizer.transform(test["text_clean"])

    clf = LogisticRegression(
        max_iter=1000, C=5.0, class_weight="balanced", n_jobs=-1
    )
    clf.fit(X_train, train["label"])

    preds = clf.predict(X_test)
    acc = accuracy_score(test["label"], preds)
    macro_f1 = f1_score(test["label"], preds, average="macro")
    report = classification_report(
        test["label"], preds, target_names=LABEL_NAMES, output_dict=True
    )
    cm = confusion_matrix(test["label"], preds).tolist()

    print(f"Baseline (TF-IDF + LogisticRegression)")
    print(f"Test accuracy: {acc:.4f} | Macro-F1: {macro_f1:.4f}")
    print(classification_report(test["label"], preds, target_names=LABEL_NAMES))

    joblib.dump(vectorizer, os.path.join(out_dir, "tfidf_vectorizer.joblib"))
    joblib.dump(clf, os.path.join(out_dir, "logreg_model.joblib"))

    metrics = {"accuracy": acc, "macro_f1": macro_f1, "report": report, "confusion_matrix": cm}
    with open(os.path.join(out_dir, "baseline_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", default="data/emotion-dataset.csv")
    parser.add_argument("--sample_frac", type=float, default=1.0)
    parser.add_argument("--out_dir", default="artifacts/baseline")
    args = parser.parse_args()
    run(args.csv_path, args.sample_frac, args.out_dir)
