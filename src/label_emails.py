import pandas as pd
from transformers import pipeline
from tqdm import tqdm

def label_emails():
    # Load the dataset
    input_path = 'cleaned_data_sets/cleaned_email_priority_dataset.csv'
    output_path = 'cleaned_data_sets/labeled_email_dataset.csv'
    
    print(f"Loading dataset from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return

    # Define candidate labels
    candidate_labels = ["Complaint", "Request", "Feedback", "Spam"]
    
    # Initialize Zero-Shot Classification pipeline
    # Using a smaller model for speed if available, or default bart-large-mnli
    print("Initializing Zero-Shot Classification pipeline...")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # Limit to a subset for testing/demonstration if needed, but we'll do all for now or a sample
    # For robust production, we might want to batch this. 
    # Let's do a sample of 20 for quick verification, or full if the user wants. 
    # Given the user context "how can I label", I should probably label the Whole dataset or a significant chunk.
    # The dataset has 28k rows? That might take a long time on CPU.
    # Let's label a sample first to demonstrate it works.
    print("Labeling a sample of 50 emails to demonstrate functionality...")
    sample_df = df.head(50).copy()
    
    predictions = []
    scores = []
    
    print("Classifying emails...")
    for text in tqdm(sample_df['body']):
        # Truncate text to fit model max length if necessary (usually 1024 tokens)
        truncated_text = text[:1024] if isinstance(text, str) else ""
        
        result = classifier(truncated_text, candidate_labels)
        predictions.append(result['labels'][0])
        scores.append(result['scores'][0])

    sample_df['category'] = predictions
    sample_df['confidence'] = scores

    # Save labeled dataset
    print(f"Saving labeled dataset to {output_path}...")
    sample_df.to_csv(output_path, index=False)
    print("Done!")
    print(sample_df[['body', 'category', 'confidence']].head())

if __name__ == "__main__":
    label_emails()
