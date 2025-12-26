from transformers import AutoTokenizer, AutoModelForTokenClassification
import json

# Model name
model_name = "Pavarissy/phayathaibert-thainer"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Save to local directory
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

print("Model and tokenizer saved to ./model")