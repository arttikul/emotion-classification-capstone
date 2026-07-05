"""Assembles the deliverable Colab notebook (notebook/emotion_classification.ipynb)
by writing raw nbformat JSON. Run once locally to (re)generate the notebook."""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(text):
    cells.append(nbf.v4.new_code_cell(text))


md("""\
# Emotion Classification: From TF-IDF to Fine-Tuned Transformers

**Deep Learning Capstone Project**

This notebook walks through a full development cycle for a 6-class text emotion
classifier on the **dair-ai/emotion** dataset (~417k short English sentences,
labeled `sadness`, `joy`, `love`, `anger`, `fear`, `surprise`).

Three models are trained and compared, in increasing order of sophistication:

1. **Baseline** — TF-IDF + Logistic Regression (classical ML reference point)
2. **From-scratch deep learning** — Embedding + BiLSTM trained from random init in PyTorch
3. **Transfer learning** — fine-tuned `distilbert-base-uncased` (Hugging Face)

**Runtime:** Runtime → Change runtime type → GPU (T4 is enough).

**Author:** Artem | **Deadline:** 06.07.2026
""")

code("""\
# 1. Setup — clone/upload the repo, install deps
# If running from the GitHub repo in Colab:
# !git clone <your-repo-url>
# %cd <your-repo-folder>

!pip install -q -r requirements.txt
""")

code("""\
import sys
sys.path.append("src")

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data import LABEL_NAMES, ID2LABEL, load_emotion_data, split_data
from viz import plot_confusion_matrix, plot_training_curves, plot_model_comparison

CSV_PATH = "data/emotion-dataset.csv"   # place emotion-dataset.csv in data/
SAMPLE_FRAC = 1.0   # set < 1.0 (e.g. 0.2) for a faster run while iterating
""")

md("""\
## 2. Exploratory Data Analysis

The dataset has 6 emotion classes with a natural imbalance (joy/sadness are
common, surprise/love are rarer) — this matters for how we weight the loss
and which metric we trust (macro-F1, not just accuracy).
""")

code("""\
df = load_emotion_data(CSV_PATH, sample_frac=SAMPLE_FRAC)
print(f"Total examples: {len(df):,}")

label_counts = df["label"].map(ID2LABEL).value_counts()
print(label_counts)

fig, ax = plt.subplots(figsize=(6, 4))
label_counts.sort_index().plot(kind="bar", ax=ax, color="#4C72B0")
ax.set_title("Class distribution")
ax.set_ylabel("# examples")
plt.tight_layout()
plt.show()
""")

code("""\
# A few example rows per class
for label_id, name in ID2LABEL.items():
    sample = df[df["label"] == label_id]["text"].sample(2, random_state=1).tolist()
    print(f"\\n[{name}]")
    for s in sample:
        print(" -", s)
""")

code("""\
train_df, val_df, test_df = split_data(df)
print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)
test_df.to_csv("data/test.csv", index=False)
""")

md("""\
## 3. Baseline — TF-IDF + Logistic Regression

A strong, fast classical baseline. Any deep learning model we build should
clearly beat this (or we should be able to explain why it doesn't).
""")

code("""\
from train_baseline import run as run_baseline

baseline_metrics = run_baseline(CSV_PATH, sample_frac=SAMPLE_FRAC, out_dir="artifacts/baseline")
""")

code("""\
plot_confusion_matrix(
    baseline_metrics["confusion_matrix"], LABEL_NAMES, "Baseline (TF-IDF + LogReg) — confusion matrix"
)
plt.show()
""")

md("""\
## 4. From-scratch deep learning — BiLSTM

An embedding layer trained from random initialization, fed into a 2-layer
bidirectional LSTM, with the final forward/backward hidden states
concatenated and passed through a linear classification head. This is the
"pure deep learning, no pretraining" data point.
""")

code("""\
from train_lstm import run as run_lstm

lstm_metrics = run_lstm(
    csv_path=CSV_PATH,
    sample_frac=SAMPLE_FRAC,
    epochs=6,
    batch_size=128,
    max_len=40,
    lr=1e-3,
    out_dir="artifacts/lstm",
)
""")

code("""\
plot_training_curves(lstm_metrics["history"], title="BiLSTM training curves")
plt.show()

plot_confusion_matrix(lstm_metrics["confusion_matrix"], LABEL_NAMES, "BiLSTM — confusion matrix")
plt.show()
""")

md("""\
## 5. Transfer learning — Fine-tuned DistilBERT

Fine-tuning a pretrained transformer (`distilbert-base-uncased`) via the
Hugging Face `Trainer` API. This should outperform the from-scratch BiLSTM
because it starts from language representations already learned on a huge
text corpus, rather than learning everything from this one dataset.

**Note:** this step needs a GPU runtime — training will be slow on CPU.
""")

code("""\
from train_transformer import run as run_transformer

transformer_metrics = run_transformer(
    csv_path=CSV_PATH,
    sample_frac=SAMPLE_FRAC,
    epochs=2,
    batch_size=32,
    max_len=64,
    lr=2e-5,
    out_dir="artifacts/transformer",
)
""")

code("""\
plot_confusion_matrix(
    transformer_metrics["confusion_matrix"], LABEL_NAMES, "Fine-tuned DistilBERT — confusion matrix"
)
plt.show()
""")

md("""\
## 6. Model comparison

Side-by-side comparison of all three models on the same held-out test set.
""")

code("""\
results = {
    "TF-IDF + LogReg": {
        "accuracy": baseline_metrics["accuracy"],
        "macro_f1": baseline_metrics["macro_f1"],
    },
    "BiLSTM (scratch)": {
        "accuracy": lstm_metrics["test_accuracy"],
        "macro_f1": lstm_metrics["test_macro_f1"],
    },
    "DistilBERT (fine-tuned)": {
        "accuracy": transformer_metrics["test_accuracy"],
        "macro_f1": transformer_metrics["test_macro_f1"],
    },
}

comparison_df = pd.DataFrame(results).T
comparison_df.columns = ["Accuracy", "Macro-F1"]
display(comparison_df.round(4))

plot_model_comparison(results)
plt.show()
""")

md("""\
## 7. Inference demo

Try the fine-tuned transformer on a few new, unseen sentences.
""")

code("""\
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = "artifacts/transformer/final_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

samples = [
    "I can't believe I finally got the job, I'm over the moon!",
    "I'm terrified of what might happen tomorrow.",
    "This is so unfair, I'm absolutely furious about it.",
    "I miss my grandmother so much, I keep crying at night.",
    "Wait, you're kidding me?! I did not see that coming at all.",
]

inputs = tokenizer(samples, truncation=True, padding=True, max_length=64, return_tensors="pt").to(device)
with torch.no_grad():
    logits = model(**inputs).logits
preds = logits.argmax(dim=1).cpu().numpy()

for text, pred in zip(samples, preds):
    print(f"[{ID2LABEL[int(pred)]:>9}]  {text}")
""")

md("""\
## 8. Conclusions

- Fill in after running: which model won on macro-F1, and on which specific
  classes (e.g. rare classes like `surprise`/`love`) did the deep learning
  models help most relative to the TF-IDF baseline?
- Discuss the accuracy/compute trade-off: BiLSTM vs. DistilBERT training
  time and parameter count vs. the macro-F1 gain.
- Note any remaining failure modes (e.g. confusable pairs from the
  confusion matrices, like `fear` vs `sadness` or `joy` vs `love`).

## 9. Possible extensions
- Data augmentation / back-translation for the rare `surprise` and `love` classes.
- Try a larger transformer backbone (`bert-base`, `roberta-base`) or a lighter one (`albert`, `tinybert`) for a speed/accuracy trade-off study.
- Error analysis notebook: pull the highest-loss misclassified examples and inspect them qualitatively.
- Wrap the fine-tuned model in a small Gradio demo for interactive testing.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
}

with open("notebook/emotion_classification.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook written to notebook/emotion_classification.ipynb")
