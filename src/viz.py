"""Shared plotting helpers for the notebook: confusion matrices, training
curves, and a final model-comparison bar chart."""

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix(cm, label_names, title, ax=None):
    cm = np.array(cm)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(label_names)))
    ax.set_yticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticklabels(label_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(len(label_names)):
        for j in range(len(label_names)):
            ax.text(
                j, i, f"{cm_norm[i, j]:.2f}",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
                fontsize=8,
            )
    return ax


def plot_training_curves(history, title="BiLSTM training curves"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="train_loss", marker="o")
    axes[0].plot(epochs, history["val_loss"], label="val_loss", marker="o")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["val_acc"], label="val_accuracy", marker="o")
    axes[1].plot(epochs, history["val_macro_f1"], label="val_macro_f1", marker="o")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation metrics")
    axes[1].legend()

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_model_comparison(results: dict):
    """results: {model_name: {"accuracy": ..., "macro_f1": ...}}"""
    names = list(results.keys())
    acc = [results[n]["accuracy"] for n in names]
    f1 = [results[n]["macro_f1"] for n in names]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, acc, width, label="Accuracy")
    ax.bar(x + width / 2, f1, width, label="Macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison on held-out test set")
    ax.legend()
    for i, (a, f) in enumerate(zip(acc, f1)):
        ax.text(i - width / 2, a + 0.01, f"{a:.3f}", ha="center", fontsize=9)
        ax.text(i + width / 2, f + 0.01, f"{f:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    return fig
