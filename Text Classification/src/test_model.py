import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the trained model and tokenizer
model_path = 'Text Classification/src/model'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Sentiment labels
sentiment_labels = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]

# Function to predict sentiment
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
    return sentiment_labels[predictions.item()]

# Test with some examples
test_texts = [
    "ร้านนี้ดีมาก ชอบอาหาร",
    "บริการแย่ ไม่กลับมาอีก",
    "สินค้าดี ราคาถูก"
]

for text in test_texts:
    sentiment = predict_sentiment(text)
    print(f"Text: {text}")
    print(f"Sentiment: {sentiment}")
    print("-" * 30)