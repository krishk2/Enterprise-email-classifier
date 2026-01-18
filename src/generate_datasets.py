import pandas as pd
import numpy as np
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Set paths relative to this script (assuming it's in src/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, '..', 'raw_data_sets')
CLEAN_DIR = os.path.join(BASE_DIR, '..', 'cleaned_data_sets')
os.makedirs(CLEAN_DIR, exist_ok=True)

# NLTK Setup
print("Setting up NLTK...")
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4') # Often needed for lemmatizer

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
PRIORITY_TERMS = r'\b(dear|support|team|ni|hello|hope|well|customer|data|please|message|team|nan|could|would|assistance|problem|issue|failure|system|update)\b'

def clean_text(text):
    # Match logic from notebook
    text = str(text)
    text = text.lower()
    # 1. Remove URLs (http/www)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # 2. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # 3. Remove specific HTML artifacts/attributes found in spam dataset
    html_artifacts = r'\b(href|src|width|height|font|size|face|arial|sans|serif|color|border|style|nbsp|img|align|center|br|div|table|tr|td|span|strong|em)\b'
    text = re.sub(html_artifacts, '', text)
    
    text = re.sub(r'[\r\n\t\a\b]+', ' ', text)
    text = re.sub(PRIORITY_TERMS, '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = text.split()
    tokens = [
        lemmatizer.lemmatize(word) 
        for word in tokens 
        if word not in stop_words and len(word) > 1
    ]
    return ' '.join(tokens)

# 1. Load Data
print("Loading data...")
try:
    a = pd.read_csv(os.path.join(RAW_DIR, 'aa_dataset-tickets-multi-lang-5-2-50-version.csv'))
    b = pd.read_csv(os.path.join(RAW_DIR, 'dataset-tickets-multi-lang-4-20k.csv'))
    c = pd.read_csv(os.path.join(RAW_DIR, 'email_dataset.csv'))
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit(1)

# 2. Filter English (Tickets only)
a = a[a['language'] == 'en']
b = b[b['language'] == 'en']

# 3. Combine Tickets
cols = ['type', 'priority', 'body', 'subject']
tag_cols = [f'tag_{i}' for i in range(1, 9)]
ticket_cols = cols + tag_cols

existing_cols_a = [col for col in ticket_cols if col in a.columns]
existing_cols_b = [col for col in ticket_cols if col in b.columns]

tickets = pd.concat([a[existing_cols_a], b[existing_cols_b]], ignore_index=True)

# 4. Feedback Extraction Logic
def process_row(row):
    current_type = row.get('type', '')
    if current_type in ['Incident', 'Change']:
        tags = [str(row.get(f'tag_{i}', '')) for i in range(1, 9)]
        if 'Feedback' in tags:
            return 'Feedback'
    if current_type == 'Problem':
        return 'Complaint'
    return current_type

print("Processing rows...")
tickets['type'] = tickets.apply(process_row, axis=1)

# 5. Filter for Valid Types Only
valid_types = ['Request', 'Complaint', 'Feedback']
tickets = tickets[tickets['type'].isin(valid_types)].copy()

# 6. Prepare Spam
c.rename(columns={'Label': 'type', 'Body': 'body', 'Subject': 'subject'}, inplace=True)
c = c[c['type'] == 'spam'].copy()
c['priority'] = 'low'
for t in tag_cols:
    c[t] = np.nan
    
# 7. Create Master DataFrame
print("Creating Master DataFrame...")
final_cols = ['type', 'priority', 'body', 'subject']
master_df = pd.concat([tickets[final_cols], c[final_cols]], ignore_index=True)
master_df.dropna(subset=['body', 'type', 'priority'], inplace=True)

# 8. Balancing
print("Balancing...")
min_count = master_df['type'].value_counts().min()
balanced_df = master_df.groupby('type').apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# 9. Save Raw Datasets
print("Saving Raw Datasets...")
balanced_df[['body', 'priority']].to_csv(os.path.join(CLEAN_DIR, 'urgency_dataset.csv'), index=False)
balanced_df[['body', 'type']].to_csv(os.path.join(CLEAN_DIR, 'classification_dataset.csv'), index=False)

# 10. Clean and Save
print("Cleaning text (this may take a moment)...")
balanced_df['body'] = balanced_df['body'].apply(clean_text)
# Optional: Clean subject if requested, but generally 'body' is the main feature. 
# User asked for "cleaned data set... free from all the space, links..." 
# Usually implies the input text features. I'll clean subject too just in case they use it.
balanced_df['subject'] = balanced_df['subject'].apply(clean_text)

print("Saving Cleaned Datasets...")
balanced_df[['body', 'priority']].to_csv(os.path.join(CLEAN_DIR, 'cleaned_urgency_dataset.csv'), index=False)
balanced_df[['body', 'type']].to_csv(os.path.join(CLEAN_DIR, 'cleaned_classification_dataset.csv'), index=False)

print("Done!")
print("Final Stats:")
print(balanced_df['type'].value_counts())
print(balanced_df['priority'].value_counts())
