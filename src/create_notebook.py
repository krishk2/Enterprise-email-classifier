import json
import os

NOTEBOOK_PATH = "src/email_classifier_pipeline.ipynb"

# Helper to create code cell
def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    }

cells = [
    # 1. Imports
    code_cell([
        "import pandas as pd\n",
        "import numpy as np\n",
        "import re\n",
        "import nltk\n",
        "import joblib\n",
        "from nltk.corpus import stopwords\n",
        "from nltk.stem import WordNetLemmatizer\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.metrics import classification_report, accuracy_score\n",
        "\n",
        "nltk.download('stopwords')\n",
        "nltk.download('wordnet')\n",
        "nltk.download('omw-1.4')"
    ]),
    # 2. Cleaning Function
    code_cell([
        "lemmatizer = WordNetLemmatizer()\n",
        "stop_words = set(stopwords.words('english'))\n",
        "PRIORITY_TERMS = r'\\b(dear|support|team|ni|hello|hope|well|customer|data|please|message|team|nan|could|would|assistance|problem|issue|failure|system|update)\\b'\n",
        "HTML_ARTIFACTS = r'\\b(href|src|width|height|font|size|face|arial|sans|serif|color|border|style|nbsp|img|align|center|br|div|table|tr|td|span|strong|em)\\b'\n",
        "\n",
        "def clean_text(text):\n",
        "    text = str(text).lower()\n",
        "    text = re.sub(r'http\\S+|www\\.\\S+', '', text) # Remove URLs\n",
        "    text = re.sub(r'<.*?>', '', text) # Remove tags\n",
        "    text = re.sub(HTML_ARTIFACTS, '', text) # Remove attributes\n",
        "    text = re.sub(r'[\\r\\n\\t\\a\\b]+', ' ', text)\n",
        "    text = re.sub(PRIORITY_TERMS, '', text)\n",
        "    text = re.sub(r'[^a-z0-9\\s]', ' ', text)\n",
        "    text = re.sub(r'\\s+', ' ', text).strip()\n",
        "    \n",
        "    tokens = text.split()\n",
        "    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 1]\n",
        "    return ' '.join(tokens)"
    ]),
    # 3. Load Data
    code_cell([
        "# Load Data\n",
        "a = pd.read_csv('../raw_data_sets/aa_dataset-tickets-multi-lang-5-2-50-version.csv')\n",
        "b = pd.read_csv('../raw_data_sets/dataset-tickets-multi-lang-4-20k.csv')\n",
        "c = pd.read_csv('../raw_data_sets/email_dataset.csv')\n",
        "\n",
        "# Filter English\n",
        "a = a[a['language'] == 'en']\n",
        "b = b[b['language'] == 'en']\n",
        "\n",
        "# Combine Tickets\n",
        "base_cols = ['type', 'priority', 'body', 'subject']\n",
        "tag_cols = [f'tag_{i}' for i in range(1, 9)]\n",
        "req_cols = base_cols + tag_cols\n",
        "a_sel = a[[col for col in req_cols if col in a.columns]]\n",
        "b_sel = b[[col for col in req_cols if col in b.columns]]\n",
        "tickets = pd.concat([a_sel, b_sel], ignore_index=True)\n",
        "\n",
        "# Feedback Logic\n",
        "def process_type_row(row):\n",
        "    curr = row.get('type', '')\n",
        "    if curr in ['Incident', 'Change']:\n",
        "        tags = [str(row.get(f'tag_{i}', '')).lower() for i in range(1, 9)]\n",
        "        if 'feedback' in tags:\n",
        "            return 'Feedback'\n",
        "    if curr == 'Problem':\n",
        "        return 'Complaint'\n",
        "    return curr\n",
        "\n",
        "tickets['type'] = tickets.apply(process_type_row, axis=1)\n",
        "valid_types = ['Request', 'Complaint', 'Feedback']\n",
        "tickets = tickets[tickets['type'].isin(valid_types)].copy()\n",
        "\n",
        "# Prepare Spam\n",
        "c = c.rename(columns={'Label': 'type', 'Body': 'body', 'Subject': 'subject'})\n",
        "c = c[c['type'] == 'spam'].copy()\n",
        "c['priority'] = 'low'\n",
        "for t in tag_cols: c[t] = np.nan\n",
        "\n",
        "# Master DF\n",
        "master = pd.concat([tickets[base_cols], c[base_cols]], ignore_index=True)\n",
        "master.dropna(subset=['body', 'type', 'priority'], inplace=True)\n",
        "\n",
        "# Balance\n",
        "min_cnt = master['type'].value_counts().min()\n",
        "balanced = master.groupby('type').apply(lambda x: x.sample(min_cnt, random_state=42)).reset_index(drop=True)\n",
        "balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)\n",
        "\n",
        "print('Data Loading Complete. Shape:', balanced.shape)\n",
        "print(balanced['type'].value_counts())"
    ]),
    # 4. Clean Data (Separately)
    code_cell([
        "print('Cleaning Subject and Body...')\n",
        "balanced['body_cleaned'] = balanced['body'].apply(clean_text)\n",
        "balanced['subject_cleaned'] = balanced['subject'].apply(clean_text)\n",
        "print('Cleaning Complete!')"
    ]),
    # 5. Ensemble Training Helper
    code_cell([
        "def train_and_eval(X, y, name):\n",
        "    print(f'\\n--- Training {name} ---')\n",
        "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)\n",
        "    \n",
        "    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))\n",
        "    X_train_vec = tfidf.fit_transform(X_train)\n",
        "    X_test_vec = tfidf.transform(X_test)\n",
        "    \n",
        "    model = LogisticRegression(max_iter=1000, class_weight='balanced')\n",
        "    model.fit(X_train_vec, y_train)\n",
        "    \n",
        "    preds = model.predict(X_test_vec)\n",
        "    print(f'{name} Accuracy: {accuracy_score(y_test, preds):.4f}')\n",
        "    return model, tfidf, X_test, y_test"
    ]),
    # 6. Train Urgency Ensemble
    code_cell([
        "print('=== URGENCY ENSEMBLE ===')\n",
        "y_urgency = balanced['priority']\n",
        "\n",
        "# Train Body\n",
        "u_body_model, u_body_tfidf, X_test_u_body, y_test_u = train_and_eval(balanced['body_cleaned'], y_urgency, 'Urgency (Body)')\n",
        "\n",
        "# Train Subject (Ensure same split by random_state)\n",
        "u_subj_model, u_subj_tfidf, X_test_u_subj, _ = train_and_eval(balanced['subject_cleaned'], y_urgency, 'Urgency (Subject)')\n",
        "\n",
        "# Ensemble Prediction (Soft Voting)\n",
        "probs_body = u_body_model.predict_proba(u_body_tfidf.transform(X_test_u_body))\n",
        "probs_subj = u_subj_model.predict_proba(u_subj_tfidf.transform(X_test_u_subj))\n",
        "avg_probs = (probs_body + probs_subj) / 2\n",
        "final_preds_idx = np.argmax(avg_probs, axis=1)\n",
        "final_preds = u_body_model.classes_[final_preds_idx]\n",
        "\n",
        "print('\\n--- Ensemble Urgency Results ---')\n",
        "print(classification_report(y_test_u, final_preds))\n",
        "\n",
        "# Save Models\n",
        "joblib.dump(u_body_model, '../models/urgency_body_model.pkl')\n",
        "joblib.dump(u_body_tfidf, '../models/urgency_body_tfidf.pkl')\n",
        "joblib.dump(u_subj_model, '../models/urgency_subject_model.pkl')\n",
        "joblib.dump(u_subj_tfidf, '../models/urgency_subject_tfidf.pkl')\n",
        "print('Urgency Models Saved!')"
    ]),
    # 7. Train Classification Ensemble
    code_cell([
        "print('=== TYPE CLASSIFICATION ENSEMBLE ===')\n",
        "y_class = balanced['type']\n",
        "\n",
        "# Train Body\n",
        "c_body_model, c_body_tfidf, X_test_c_body, y_test_c = train_and_eval(balanced['body_cleaned'], y_class, 'Type (Body)')\n",
        "\n",
        "# Train Subject\n",
        "c_subj_model, c_subj_tfidf, X_test_c_subj, _ = train_and_eval(balanced['subject_cleaned'], y_class, 'Type (Subject)')\n",
        "\n",
        "# Ensemble Prediction\n",
        "probs_body = c_body_model.predict_proba(c_body_tfidf.transform(X_test_c_body))\n",
        "probs_subj = c_subj_model.predict_proba(c_subj_tfidf.transform(X_test_c_subj))\n",
        "avg_probs = (probs_body + probs_subj) / 2\n",
        "final_preds_idx = np.argmax(avg_probs, axis=1)\n",
        "final_preds = c_body_model.classes_[final_preds_idx]\n",
        "\n",
        "print('\\n--- Ensemble Type Results ---')\n",
        "print(classification_report(y_test_c, final_preds))\n",
        "\n",
        "# Save Models\n",
        "joblib.dump(c_body_model, '../models/class_body_model.pkl')\n",
        "joblib.dump(c_body_tfidf, '../models/class_body_tfidf.pkl')\n",
        "joblib.dump(c_subj_model, '../models/class_subject_model.pkl')\n",
        "joblib.dump(c_subj_tfidf, '../models/class_subject_tfidf.pkl')\n",
        "print('Type Models Saved!')"
    ]),
    # 8. Interactive Testing (ipywidgets)
    code_cell([
        "# Ensure widgets are installed\n",
        "!pip install ipywidgets\n",
        "import ipywidgets as widgets\n",
        "from IPython.display import display, clear_output\n",
        "\n",
        "print('=== Email Classifier Interface ===')\n",
        "\n",
        "# Widgets\n",
        "w_subject = widgets.Text(description='Subject:', placeholder='Enter email subject...')\n",
        "w_body = widgets.Textarea(description='Body:', placeholder='Enter email body...', layout=widgets.Layout(height='150px', width='600px'))\n",
        "w_btn = widgets.Button(description='Classify', button_style='primary')\n",
        "w_out = widgets.Output()\n",
        "\n",
        "def on_click(b):\n",
        "    with w_out:\n",
        "        clear_output()\n",
        "        subj_text = w_subject.value\n",
        "        body_text = w_body.value\n",
        "        \n",
        "        if not body_text:\n",
        "            print('Please enter some text in the Body field.')\n",
        "            return\n",
        "        \n",
        "        # Rule-Based Urgency Heuristic\n",
        "        text_combined = (str(subj_text) + ' ' + str(body_text)).lower()\n",
        "        high_triggers = ['deadline', 'asap', 'immediate', 'urgency', 'critical', 'breach', 'emergency', 'shuts down', 'exploded', 'security alert', 'system down', 'outage', 'unacceptable']\n",
        "        rule_triggered = False\n",
        "        for word in high_triggers:\n",
        "            if word in text_combined:\n",
        "                urgency = 'high'\n",
        "                u_conf = 0.99\n",
        "                rule_triggered = True\n",
        "                print(f'  [Rule Trigger] Found critical term: {word}')\n",
        "                break\n",
        "        \n",
        "        # Urgency Prediction (ML) if no rule triggered\n",
        "        if not rule_triggered:\n",
        "            b_vec_u = u_body_tfidf.transform([clean_text(body_text)])\n",
        "            s_vec_u = u_subj_tfidf.transform([clean_text(subj_text)])\n",
        "            p_b_u = u_body_model.predict_proba(b_vec_u)[0]\n",
        "            p_s_u = u_subj_model.predict_proba(s_vec_u)[0]\n",
        "            avg_u = (p_b_u + p_s_u) / 2\n",
        "            urgency = u_body_model.classes_[np.argmax(avg_u)]\n",
        "            u_conf = np.max(avg_u)\n",
        "        \n",
        "        # Type Prediction\n",
        "        b_vec_c = c_body_tfidf.transform([clean_text(body_text)])\n",
        "        s_vec_c = c_subj_tfidf.transform([clean_text(subj_text)])\n",
        "        p_b_c = c_body_model.predict_proba(b_vec_c)[0]\n",
        "        p_s_c = c_subj_model.predict_proba(s_vec_c)[0]\n",
        "        avg_c = (p_b_c + p_s_c) / 2\n",
        "        e_type = c_body_model.classes_[np.argmax(avg_c)]\n",
        "        t_conf = np.max(avg_c)\n",
        "        \n",
        "        # Display\n",
        "        print(f'Subject: {subj_text}')\n",
        "        print('-' * 40)\n",
        "        print(f'Predicted Urgency: {urgency} ({u_conf:.1%})')\n",
        "        print(f'Predicted Type:    {e_type} ({t_conf:.1%})')\n",
        "\n",
        "w_btn.on_click(on_click)\n",
        "display(w_subject, w_body, w_btn, w_out)"
    ])
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.5"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Ensure models dir exists
os.makedirs('../models', exist_ok=True)

with open(NOTEBOOK_PATH, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Notebook created at {NOTEBOOK_PATH}")
