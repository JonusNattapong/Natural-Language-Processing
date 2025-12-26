import os
import random

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_NAME = os.getenv("MODEL_NAME", "tabularisai/multilingual-sentiment-analysis")
HF_DATASET = os.getenv("HF_DATASET", "pythainlp/wisesight_sentiment")
HF_SPLITS = [s.strip() for s in os.getenv("HF_SPLITS", "train,validation,test").split(",") if s.strip()]
OUTPUT_FILE = os.getenv(
    "OUTPUT_FILE",
    os.path.join(DATA_DIR, "thai_sentiment_dataset_relabeled.csv"),
)

MAX_SAMPLES = int(os.getenv("MAX_SAMPLES", "20000"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
SEED = int(os.getenv("SEED", "42"))

POS_VERY_THRESHOLD = float(os.getenv("POS_VERY_THRESHOLD", "0.75"))
NEG_VERY_THRESHOLD = float(os.getenv("NEG_VERY_THRESHOLD", "0.75"))


def label_name_to_class(label_name, confidence):
    name = label_name.lower()
    if "pos" in name:
        return 4 if confidence >= POS_VERY_THRESHOLD else 3
    if "neg" in name:
        return 0 if confidence >= NEG_VERY_THRESHOLD else 1
    return 2


def resolve_label_name(id2label, label_id):
    label = id2label.get(label_id, str(label_id))
    return str(label)


def predict_labels(texts, tokenizer, model, device):
    enc = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        outputs = model(**enc)
        probs = torch.softmax(outputs.logits, dim=-1)
    confidences, label_ids = torch.max(probs, dim=-1)
    return label_ids.cpu().tolist(), confidences.cpu().tolist()


def balance_and_cap(rows, rng):
    by_label = {i: [] for i in range(5)}
    for row in rows:
        by_label[row["label"]].append(row)

    per_label = MAX_SAMPLES // 5
    selected = []
    leftovers = []
    for label, items in by_label.items():
        rng.shuffle(items)
        take = min(len(items), per_label)
        selected.extend(items[:take])
        leftovers.extend(items[take:])

    remaining = MAX_SAMPLES - len(selected)
    if remaining > 0 and leftovers:
        rng.shuffle(leftovers)
        selected.extend(leftovers[:remaining])

    rng.shuffle(selected)
    return selected


def main():
    rng = random.Random(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    rows = []
    for split in HF_SPLITS:
        dataset = load_dataset(HF_DATASET, split=split)
        texts = dataset["texts"] if "texts" in dataset.column_names else dataset["text"]
        total = len(texts)
        batch_range = range(0, total, BATCH_SIZE)
        if tqdm:
            batch_range = tqdm(batch_range, desc=f"Relabeling {split}", unit="batch")
        for i in batch_range:
            batch_texts = [str(t) for t in texts[i : i + BATCH_SIZE]]
            label_ids, confidences = predict_labels(batch_texts, tokenizer, model, device)
            for text, label_id, conf in zip(batch_texts, label_ids, confidences):
                label_name = resolve_label_name(model.config.id2label, label_id)
                label = label_name_to_class(label_name, conf)
                rows.append(
                    {
                        "text": text,
                        "label": label,
                        "model_label": label_name,
                        "confidence": round(float(conf), 4),
                    }
                )

    rows = balance_and_cap(rows, rng)
    os.makedirs(DATA_DIR, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_FILE, index=False)
    print(f"Relabeled dataset saved to {OUTPUT_FILE}")
    print(f"Total samples: {len(rows)}")


if __name__ == "__main__":
    main()
