"""
Simple whitespace vocabulary + numericalization utilities used by the
from-scratch BiLSTM model (train_lstm.py). Kept dependency-free (no torch)
so it can be unit-tested anywhere.
"""

from collections import Counter

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


class Vocab:
    def __init__(self, texts, min_freq: int = 2, max_size: int = 30000):
        counter = Counter()
        for t in texts:
            counter.update(t.split())

        self.itos = [PAD_TOKEN, UNK_TOKEN]
        for word, freq in counter.most_common():
            if freq < min_freq:
                continue
            if len(self.itos) >= max_size:
                break
            self.itos.append(word)

        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD_TOKEN]
        self.unk_id = self.stoi[UNK_TOKEN]

    def __len__(self):
        return len(self.itos)

    def encode(self, text: str, max_len: int = 40):
        ids = [self.stoi.get(w, self.unk_id) for w in text.split()][:max_len]
        if len(ids) < max_len:
            ids = ids + [self.pad_id] * (max_len - len(ids))
        return ids

    def encode_batch(self, texts, max_len: int = 40):
        return [self.encode(t, max_len) for t in texts]


if __name__ == "__main__":
    # tiny self-test, no external deps
    sample = ["i feel happy today", "i feel sad and alone", "this is great news"]
    v = Vocab(sample, min_freq=1)
    print(f"Vocab size: {len(v)}")
    encoded = v.encode_batch(sample, max_len=6)
    for t, e in zip(sample, encoded):
        assert len(e) == 6
        print(t, "->", e)
    # unseen word should map to <unk>
    unk_encoded = v.encode("zzz_never_seen_word", max_len=6)
    assert unk_encoded[0] == v.unk_id
    print("OK: unseen word mapped to <unk> id", v.unk_id)
