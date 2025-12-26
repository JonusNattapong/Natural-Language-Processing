import json
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import numpy as np

from pathlib import Path

# Load dataset (resolve path relative to this script)
data = []
data_file = Path(__file__).resolve().parents[1] / 'data' / 'ThaiNER.jsonl'
if not data_file.exists():
    raise SystemExit(f"Data file not found: {data_file}. Ensure you're running the script from the repository root or provide the correct path.")

with data_file.open('r', encoding='utf-8') as f:
    for line in f:
        item = json.loads(line)
        if 'tags' in item:  # Only use entries with tags format
            data.append(item)

print(f'Loaded {len(data)} examples from {data_file}')

# Extract unique labels
all_tags = set()
for item in data:
    all_tags.update(item['tags'])
label_list = sorted(list(all_tags))
label_to_id = {label: i for i, label in enumerate(label_list)}
id_to_label = {i: label for label, i in label_to_id.items()}

print(f"Labels: {label_list}")
print(f"Number of labels: {len(label_list)}")

# Split data
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Load model and tokenizer
model_name = "Pavarissy/phayathaibert-thainer"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(
        model_name, 
        num_labels=len(label_list), 
        id2label=id_to_label, 
        label2id=label_to_id,
        ignore_mismatched_sizes=True
    )

# Custom Dataset
class NERDataset(Dataset):
    def __init__(self, data, tokenizer, label_to_id, max_len=512):
        self.data = data
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item['tokens']
        tags = item['tags']

        # Tokenize
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        # Align labels
        word_ids = encoding.word_ids()
        labels = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                labels.append(-100)
            elif word_idx != previous_word_idx:
                labels.append(self.label_to_id[tags[word_idx]])
            else:
                labels.append(-100)
            previous_word_idx = word_idx

        encoding['labels'] = torch.tensor(labels)
        return {k: v.squeeze() for k, v in encoding.items()}

# Create datasets
train_dataset = NERDataset(train_data, tokenizer, label_to_id)
test_dataset = NERDataset(test_data, tokenizer, label_to_id)

# Data collator and metrics
from transformers import DataCollatorForTokenClassification
from seqeval.metrics import precision_score, recall_score, f1_score

data_collator = DataCollatorForTokenClassification(tokenizer)

def compute_metrics(p):
    predictions, labels = p
    preds = np.argmax(predictions, axis=2)
    true_labels, pred_labels = [], []
    for i in range(len(labels)):
        t, pr = [], []
        for j, lab in enumerate(labels[i]):
            if lab != -100:
                t.append(id_to_label[lab])
                pr.append(id_to_label[preds[i][j]])
        true_labels.append(t)
        pred_labels.append(pr)
    return {
        'precision': precision_score(true_labels, pred_labels),
        'recall': recall_score(true_labels, pred_labels),
        'f1': f1_score(true_labels, pred_labels),
    }

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model='f1',
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train
trainer.train()

# Save model
model.save_pretrained('./model')
tokenizer.save_pretrained('./model')