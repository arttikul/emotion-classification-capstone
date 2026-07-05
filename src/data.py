"""
Data loading & preprocessing for the Emotion Classification capstone.

Dataset: dair-ai/emotion (unsplit, ~417k short English texts labeled with
one of 6 emotions). CSV columns: text, label (int 0-5).

Label mapping (standard dair-ai/emotion convention):
    0 -> sadness
    1 -> joy
    2 -> love
    3 -> anger
    4 -> fear
    5 -> surprise
"""

import re
import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_NAMES = ["sadness", "joy", "love", "anger", "fear", "surprise"]
LABEL2ID = {name: i for i, name in enumerate(LABEL_NAMES)}
ID2LABEL = {i: name for i, name in enumerate(LABEL_NAMES)}


def clean_text(text: str) -> str:
    """Light cleaning: lowercase, strip extra whitespace, remove stray non-alpha noise.
    The dataset is already lowercase / de-punctuated, so this mostly guards against
    unexpected input (e.g. when using the model on new sentences)."""
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_emotion_data(
    csv_path: str,
    sample_frac: float = 1.0,
    random_state: int = 42,
):
    """Load the CSV, optionally subsample (stratified) for quick experiments,
    clean text, and drop empty/duplicate rows.

    Args:
        csv_path: path to emotion-dataset.csv
        sample_frac: fraction of the full dataset to keep (1.0 = use all ~417k rows).
                     Useful for fast local smoke-tests (e.g. 0.01).
        random_state: seed for reproducibility.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["text", "label"]).drop_duplicates(subset=["text"])
    df["label"] = df["label"].astype(int)

    if sample_frac < 1.0:
        df, _ = train_test_split(
            df, train_size=sample_frac, stratify=df["label"], random_state=random_state
        )

    df["text_clean"] = df["text"].apply(clean_text)
    df = df[df["text_clean"].str.len() > 0].reset_index(drop=True)
    return df


def split_data(df: pd.DataFrame, test_size=0.15, val_size=0.15, random_state=42):
    """Stratified split into train/val/test. val_size/test_size are fractions of
    the *original* dataframe (not of the remainder)."""
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=random_state
    )
    val_relative = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_relative, stratify=train_val["label"], random_state=random_state
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/emotion-dataset.csv"
    frac = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
    df = load_emotion_data(path, sample_frac=frac)
    train, val, test = split_data(df)
    print(f"Total: {len(df)} | Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(df["label"].map(ID2LABEL).value_counts())
