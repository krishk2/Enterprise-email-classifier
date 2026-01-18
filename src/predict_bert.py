import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import argparse
import os
import pandas as pd

def get_label_map(data_path='cleaned_data_sets/labeled_email_dataset.csv'):
    try:
        df = pd.read_csv(data_path)
        # Reconstruct the map used in training
        # Note: This relies on the dataset being the same as used in training
        df = df.dropna(subset=['body', 'category'])
        label_map = {label: i for i, label in enumerate(df['category'].unique())}
        id2label = {i: label for label, i in label_map.items()}
        return id2label
    except Exception as e:
        print(f"Warning: Could not load dataset to reconstruct labels: {e}")
        return None

def predict_email(text, model_path='src/email_classifier_model.pkl'):
    if not os.path.exists(model_path):
        print(f"Error: Model directory not found at {model_path}.")
        print("Please ensure you have trained the model using 'python src/train_bert.py'")
        return

    print(f"Loading model from {model_path}...")
    try:
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = DistilBertForSequenceClassification.from_pretrained(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    
    with torch.no_grad():
        logits = model(**inputs).logits
    
    predicted_class_id = logits.argmax().item()
    
    id2label = get_label_map()
    
    print(f"Predicted Class ID: {predicted_class_id}")
    if id2label:
        predicted_label = id2label.get(predicted_class_id, "Unknown")
        print(f"Predicted Label: {predicted_label}")
    else:
        print("Raw Class ID returned (Label mapping could not be reconstructed)")

    return predicted_class_id

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict email category using BERT')
    parser.add_argument('--text', type=str, help='Email text to classify')
    parser.add_argument('--model_path', type=str, default='src/email_classifier_model.pkl', help='Path to trained model')
    
    args = parser.parse_args()
    
    if args.text:
       predict_email(args.text, args.model_path)
    else:
        # Interactive mode if no args
        print("Enter email text to classify (or 'q' to quit):")
        while True:
            text = input("> ")
            if text.lower() == 'q':
                break
            predict_email(text, args.model_path)
