from datasets import load_dataset
import json

# Load the external dataset
dataset = load_dataset("bltlab/open-ner-standardized", "ThaiNNER/tha")

print("Dataset info:")
print(dataset)
print("\nColumn names:")
print(dataset['train'].column_names)
print("\nFirst example:")
print(dataset['train'][0])

# Check if there are other splits
for split in dataset.keys():
    print(f"\n{split} split has {len(dataset[split])} examples")