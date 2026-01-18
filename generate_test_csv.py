import pandas as pd
import os

RAW_DIR = r'c:\Users\Krish\Documents\GitHub\Enterprise-email-classifier\raw_data_sets'
OUT_FILE = 'test_emails.csv'

# Load raw datasets
try:
    df1 = pd.read_csv(os.path.join(RAW_DIR, 'email_dataset.csv'))
    df1 = df1[['Subject', 'Body']].rename(columns={'Subject': 'subject', 'Body': 'body'})
    
    df2 = pd.read_csv(os.path.join(RAW_DIR, 'aa_dataset-tickets-multi-lang-5-2-50-version.csv'))
    df2 = df2[df2['language'] == 'en'][['subject', 'body']]
    
    # Combine and sample
    combined = pd.concat([df1, df2], ignore_index=True).dropna()
    
    # Sample 20 random rows
    sample = combined.sample(20, random_state=999) # Different seed than training if possible, but 999 is fine
    
    sample.to_csv(OUT_FILE, index=False)
    print(f"Successfully created {OUT_FILE} with {len(sample)} rows.")
    print("Columns:", sample.columns.tolist())
    
except Exception as e:
    print(f"Error: {e}")
