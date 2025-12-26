import os
import random

import pandas as pd
from pythainlp import word_tokenize

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
SOURCE_FILE = os.getenv(
    "SOURCE_FILE",
    os.path.join(DATA_DIR, "thai_sentiment_dataset_relabeled.csv"),
)
OUTPUT_FILE = os.path.join(DATA_DIR, "thai_sentiment_dataset.csv")

SEED = 42
AUGMENTATIONS_PER_TEXT = int(os.getenv("AUG_PER_TEXT", "3"))
USE_HF = os.getenv("USE_HF", "1") == "1"
HF_DATASET = os.getenv("HF_DATASET", "pythainlp/wisesight_sentiment")
HF_SPLITS = [split.strip() for split in os.getenv("HF_SPLITS", "train,validation,test").split(",") if split.strip()]
HF_Q_LABEL = os.getenv("HF_Q_LABEL", "neutral").strip().lower()

LABEL_MAP = {
    "very negative": 0,
    "negative": 1,
    "neutral": 2,
    "positive": 3,
    "very positive": 4,
}

HF_LABEL_MAP_STR = {
    "pos": 3,
    "neu": 2,
    "neg": 1,
    "q": 2,
}

HF_LABEL_MAP_INT = {
    0: 3,
    1: 2,
    2: 1,
    3: 2,
}


def normalize_label(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except ValueError:
            return None
    text = str(value).strip().lower()
    if text == "":
        return None
    if text.isdigit():
        return int(text)
    return LABEL_MAP.get(text)


def map_hf_label(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        int_value = int(value)
        if int_value == 3 and HF_Q_LABEL == "drop":
            return None
        return HF_LABEL_MAP_INT.get(int_value)
    text = str(value).strip().lower()
    if text.isdigit():
        int_value = int(text)
        if int_value == 3 and HF_Q_LABEL == "drop":
            return None
        return HF_LABEL_MAP_INT.get(int_value)
    if text == "q" and HF_Q_LABEL == "drop":
        return None
    return HF_LABEL_MAP_STR.get(text)


def random_delete(tokens, p, rng):
    if len(tokens) <= 1:
        return tokens[:]
    kept = [tok for tok in tokens if rng.random() > p]
    return kept if kept else [tokens[rng.randrange(len(tokens))]]


def random_swap(tokens, n, rng):
    if len(tokens) < 2:
        return tokens[:]
    out = tokens[:]
    for _ in range(n):
        i, j = rng.sample(range(len(out)), 2)
        out[i], out[j] = out[j], out[i]
    return out


def augment_tokens(tokens, rng):
    augmented = tokens[:]
    if rng.random() < 0.7:
        augmented = random_delete(augmented, p=0.1, rng=rng)
    if rng.random() < 0.7:
        swaps = max(1, len(augmented) // 5)
        augmented = random_swap(augmented, swaps, rng)
    return augmented


def tokens_to_text(tokens):
    return "".join(tokens)


def load_base_rows():
    if not os.path.exists(SOURCE_FILE):
        raise FileNotFoundError(f"Source dataset not found: {SOURCE_FILE}")

    df = pd.read_csv(SOURCE_FILE)
    if "text" not in df.columns:
        raise ValueError("Expected a 'text' column in the source dataset.")

    label_col = None
    if "new_label" in df.columns:
        label_col = "new_label"
    elif "label" in df.columns:
        label_col = "label"
    elif "original_label" in df.columns:
        label_col = "original_label"
    else:
        raise ValueError("No label column found in the source dataset.")

    rows = []
    for _, row in df.iterrows():
        text = str(row["text"])
        raw_label = row.get(label_col)
        label = normalize_label(raw_label)
        if label is None and label_col != "original_label" and "original_label" in df.columns:
            label = normalize_label(row.get("original_label"))
        if label is None:
            continue
        rows.append((text, label))
    return rows


def load_hf_rows():
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets not installed; skipping Hugging Face dataset.")
        return []

    rows = []
    for split in HF_SPLITS:
        dataset = load_dataset(HF_DATASET, split=split)
        for item in dataset:
            text = item.get("texts") or item.get("text")
            label = item.get("category") if "category" in item else item.get("label")
            mapped = map_hf_label(label)
            if text is None or mapped is None:
                continue
            rows.append((str(text), mapped))
    return rows


def build_dataset(rows, rng):
    data = []
    for text, label in rows:
        tokens = word_tokenize(text)
        data.append({"text": text, "tokens": tokens, "label": label})
        for _ in range(AUGMENTATIONS_PER_TEXT):
            aug_tokens = augment_tokens(tokens, rng)
            aug_text = tokens_to_text(aug_tokens)
            data.append({"text": aug_text, "tokens": aug_tokens, "label": label})
    rng.shuffle(data)
    return pd.DataFrame(data)


def main():
    rng = random.Random(SEED)
    rows = load_base_rows()
    if USE_HF:
        rows.extend(load_hf_rows())
    df = build_dataset(rows, rng)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset created and saved to {OUTPUT_FILE}")
    print(f"Total samples: {len(df)}")
    print(df.head())


if __name__ == "__main__":
    main()
