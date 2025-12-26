# Thai Text Classification

This project implements a text classification model for Thai language sentiment analysis with 5 sentiment classes.

## Files

- `requirements.txt`: Python dependencies
- `create_dataset.py`: Script to create a synthetic Thai sentiment dataset
- `train_model.py`: Script to train the classification model using WangchanBERTa
- `test_model.py`: Script to test the trained model

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Create the dataset:
   ```
   python create_dataset.py
   ```

3. Train the model:
   ```
   python train_model.py
   ```

4. Test the model:
   ```
   python test_model.py
   ```

## Model

The model uses `airesearch/wangchanberta-base-att-spm-uncased` as the base model, fine-tuned for 5-class sentiment classification (Very Negative, Negative, Neutral, Positive, Very Positive).

## Dataset

The dataset is a synthetic collection of Thai texts with 5 sentiment labels, tokenized using pythainlp.