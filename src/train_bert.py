import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_bert():
    input_path = 'cleaned_data_sets/labeled_email_dataset.csv'
    output_dir = 'src/bert_model'
    
    print(f"Loading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Could not find {input_path}")
        return

    # Drop missing
    df = df.dropna(subset=['body', 'category'])

    # Label Encoding
    label_map = {label: i for i, label in enumerate(df['category'].unique())}
    print(f"Label Mapping: {label_map}")
    df['label'] = df['category'].map(label_map)

    X_train, X_test, y_train, y_test = train_test_split(df['body'], df['label'], test_size=0.2, random_state=42, stratify=df['label'])

    print("Initializing Tokenizer and Model...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(label_map))

    train_dataset = EmailDataset(X_train.to_numpy(), y_train.to_numpy(), tokenizer)
    test_dataset = EmailDataset(X_test.to_numpy(), y_test.to_numpy(), tokenizer)

    # Training Arguments
    # Using small batch size and 1 epoch for demonstration/speed on CPU
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=1,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        no_cuda=not torch.cuda.is_available() # Detect GPU automatically
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    print("Starting Training...")
    trainer.train()

    print("Evaluating...")
    eval_result = trainer.evaluate()
    print(f"Evaluation Results: {eval_result}")

    print(f"Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done!")

if __name__ == "__main__":
    train_bert()
