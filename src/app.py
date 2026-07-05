"""
Gradio demo for interactively testing the trained emotion classifiers.

Auto-detects which of the three models have trained artifacts under
artifacts/ (baseline / lstm / transformer) and lets you pick one to
classify an arbitrary English sentence.

Run:
    python src/app.py
"""

import json
import os

import gradio as gr
import torch

from data import ID2LABEL, LABEL_NAMES, clean_text
from models import BiLSTMClassifier, get_device
from vocab import Vocab

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")
LSTM_MAX_LEN = 40
TRANSFORMER_MAX_LEN = 64

EXAMPLES = [
    "I can't believe I finally got the job, I'm over the moon!",
    "I'm terrified of what might happen tomorrow.",
    "This is so unfair, I'm absolutely furious about it.",
    "I miss my grandmother so much, I keep crying at night.",
    "Wait, you're kidding me?! I did not see that coming at all.",
]

_predictors = {}


def _load_baseline():
    import joblib

    base_dir = os.path.join(ARTIFACTS_DIR, "baseline")
    vectorizer = joblib.load(os.path.join(base_dir, "tfidf_vectorizer.joblib"))
    clf = joblib.load(os.path.join(base_dir, "logreg_model.joblib"))

    def predict(text):
        x = vectorizer.transform([clean_text(text)])
        probs = clf.predict_proba(x)[0]
        return {ID2LABEL[i]: float(p) for i, p in enumerate(probs)}

    return predict


def _load_lstm():
    lstm_dir = os.path.join(ARTIFACTS_DIR, "lstm")
    with open(os.path.join(lstm_dir, "vocab.json")) as f:
        itos = json.load(f)
    vocab = Vocab.from_itos(itos)

    device = get_device()
    model = BiLSTMClassifier(
        vocab_size=len(vocab), num_classes=len(LABEL_NAMES), pad_idx=vocab.pad_id
    )
    state_dict = torch.load(
        os.path.join(lstm_dir, "bilstm_model.pt"), map_location="cpu"
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    def predict(text):
        cleaned = clean_text(text)
        ids = vocab.encode(cleaned, LSTM_MAX_LEN)
        length = min(len(cleaned.split()), LSTM_MAX_LEN) or 1
        ids_t = torch.tensor([ids], dtype=torch.long, device=device)
        length_t = torch.tensor([length], dtype=torch.long, device=device)
        with torch.no_grad():
            probs = torch.softmax(model(ids_t, length_t), dim=1)[0].cpu().numpy()
        return {ID2LABEL[i]: float(p) for i, p in enumerate(probs)}

    return predict


def _load_transformer():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = get_device()
    model_dir = os.path.join(ARTIFACTS_DIR, "transformer", "final_model")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    def predict(text):
        inputs = tokenizer(
            clean_text(text),
            truncation=True,
            max_length=TRANSFORMER_MAX_LEN,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits, dim=1)[0].cpu().numpy()
        return {ID2LABEL[i]: float(p) for i, p in enumerate(probs)}

    return predict


# name -> (loader, path that must exist for the model to be usable)
_MODELS = {
    "TF-IDF + Logistic Regression": (
        _load_baseline,
        os.path.join(ARTIFACTS_DIR, "baseline", "logreg_model.joblib"),
    ),
    "BiLSTM (з нуля)": (
        _load_lstm,
        os.path.join(ARTIFACTS_DIR, "lstm", "bilstm_model.pt"),
    ),
    "DistilBERT (fine-tuned)": (
        _load_transformer,
        os.path.join(ARTIFACTS_DIR, "transformer", "final_model"),
    ),
}


def available_models():
    return [name for name, (_, marker) in _MODELS.items() if os.path.exists(marker)]


def get_predictor(name):
    if name not in _predictors:
        loader, _ = _MODELS[name]
        _predictors[name] = loader()
    return _predictors[name]


def predict(text, model_name):
    if not text or not text.strip():
        return {}
    return get_predictor(model_name)(text)


def build_demo():
    models = available_models()
    if not models:
        raise RuntimeError(
            "No trained model artifacts found under artifacts/. Train at least "
            "one model first (run the notebook or src/train_*.py), or copy the "
            "artifacts/ folder produced on Colab into this repo."
        )

    with gr.Blocks(title="Emotion Classification Demo") as demo:
        gr.Markdown(
            "# Класифікація емоцій — демо\n"
            "Введи речення англійською і подивись, яку емоцію передбачає кожна модель."
        )
        text_input = gr.Textbox(
            label="Текст (англійською)",
            placeholder="I can't believe this is happening...",
            lines=3,
        )
        model_choice = gr.Radio(models, value=models[-1], label="Модель")
        output = gr.Label(label="Передбачені емоції", num_top_classes=6)
        submit = gr.Button("Класифікувати", variant="primary")

        submit.click(predict, inputs=[text_input, model_choice], outputs=output)
        text_input.submit(predict, inputs=[text_input, model_choice], outputs=output)
        gr.Examples(examples=EXAMPLES, inputs=text_input)

    return demo


if __name__ == "__main__":
    build_demo().launch()
