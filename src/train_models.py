import pandas as pd
import numpy as np
import re
import nltk
import joblib
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# Ensure dependencies
print("Checking NLTK...")
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) - {'system', 'problem', 'issue', 'failure', 'bug', 'urgent', 'critical', 'thank', 'you', 'please'}
PRIORITY_TERMS = r'\b(dear|support|team|ni|hello|hope|well|customer|data|please|message|team|nan|could|would|assistance|update)\b'
HTML_ARTIFACTS = r'\b(href|src|width|height|font|size|face|arial|sans|serif|color|border|style|nbsp|img|align|center|br|div|table|tr|td|span|strong|em)\b'

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(HTML_ARTIFACTS, '', text)
    text = re.sub(r'[\r\n\t]+', ' ', text)
    # text = re.sub(PRIORITY_TERMS, '', text) # Preserving keywords like problem, issue, failure
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 1]
    return ' '.join(tokens)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, '..', 'raw_data_sets')
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

print("Loading Data...")
a = pd.read_csv(os.path.join(RAW_DIR, 'aa_dataset-tickets-multi-lang-5-2-50-version.csv'))
b = pd.read_csv(os.path.join(RAW_DIR, 'dataset-tickets-multi-lang-4-20k.csv'))
c = pd.read_csv(os.path.join(RAW_DIR, 'email_dataset.csv'))

a = a[a['language'] == 'en']
b = b[b['language'] == 'en']

base_cols = ['type', 'priority', 'body', 'subject']
tag_cols = [f'tag_{i}' for i in range(1, 9)]
req_cols = base_cols + tag_cols
a_sel = a[[col for col in req_cols if col in a.columns]]
b_sel = b[[col for col in req_cols if col in b.columns]]
tickets = pd.concat([a_sel, b_sel], ignore_index=True)

def process_type_row(row):
    curr = row.get('type', '')
    if curr in ['Incident', 'Change']:
        tags = [str(row.get(f'tag_{i}', '')).lower() for i in range(1, 9)]
        if 'feedback' in tags:
            return 'Feedback'
    if curr == 'Problem':
        return 'Complaint'
    return curr

print("Processing Feedback...")
tickets['type'] = tickets.apply(process_type_row, axis=1)
valid_types = ['Request', 'Complaint', 'Feedback']
tickets = tickets[tickets['type'].isin(valid_types)].copy()

c = c.rename(columns={'Label': 'type', 'Body': 'body', 'Subject': 'subject'})
c = c[c['type'] == 'spam'].copy()
c['priority'] = 'low'
for t in tag_cols: c[t] = np.nan


print("Loading G2 Reviews...")
d = pd.read_csv(os.path.join(RAW_DIR, 'Hubspot (SaaS Company) - G2 Reviews.csv'))
d = d.dropna(subset=['Content'])

def parse_g2_content(content):
    # Extract Title between first """ and ""
    # Format: """Title"" Body..."
    try:
        if content.startswith('"""'):
            parts = content.split('""', 2)
            if len(parts) >= 2:
                title = parts[1].strip()
                body = parts[2].strip()
                # Remove common G2 boilerplate
                body = body.replace('Review collected by and hosted on G2.com.', '')
                return title, body
    except:
        pass
    return "Feedback Review", content

d_parsed = d['Content'].apply(parse_g2_content).apply(pd.Series)
d['subject'] = d_parsed[0]
d['body'] = d_parsed[1]
d['type'] = 'Feedback'
d['priority'] = 'low'
for t in tag_cols: d[t] = np.nan

print("Merging Datasets...")
master = pd.concat([tickets[base_cols], c[base_cols], d[base_cols]], ignore_index=True)
master.dropna(subset=['body', 'type', 'priority'], inplace=True)

# Standardize type names
master['type'] = master['type'].replace({'Problem': 'Complaint'})

print("Balancing (Upsampling to Max Class)...")
# Check counts
print(master['type'].value_counts())
max_cnt = master['type'].value_counts().max()

balanced = master.groupby('type').apply(lambda x: x.sample(max_cnt, replace=True, random_state=42)).reset_index(drop=True)
balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)
print("Balanced Counts:")
print(balanced['type'].value_counts())
balanced.to_csv(os.path.join(RAW_DIR, 'balanced_dataset.csv'), index=False)
print(f"Saved balanced dataset to {os.path.join(RAW_DIR, 'balanced_dataset.csv')}")


print("Cleaning text (Subject & Body)...")
balanced['body_cleaned'] = balanced['body'].apply(clean_text)
balanced['subject_cleaned'] = balanced['subject'].apply(clean_text)

def train_and_save(X, y, name, model_name):
    print(f"\n--- Training {name} ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)
    
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', multi_class='ovr')
    lr.fit(X_train_vec, y_train)
    
    print(classification_report(y_test, lr.predict(X_test_vec)))
    
    joblib.dump(lr, os.path.join(MODELS_DIR, f'{model_name}_model.pkl'))
    joblib.dump(tfidf, os.path.join(MODELS_DIR, f'{model_name}_tfidf.pkl'))
    print(f"Saved {model_name}")

# 1. Urgency (Body)
train_and_save(balanced['body_cleaned'], balanced['priority'], "Urgency (Body)", "urgency_body")
# 2. Urgency (Subject)
train_and_save(balanced['subject_cleaned'], balanced['priority'], "Urgency (Subject)", "urgency_subject")

# 3. Type (Body)
train_and_save(balanced['body_cleaned'], balanced['type'], "Type (Body)", "class_body")
# 4. Type (Subject)
train_and_save(balanced['subject_cleaned'], balanced['type'], "Type (Subject)", "class_subject")
