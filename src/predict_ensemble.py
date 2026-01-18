import joblib
import re
import nltk
import numpy as np
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# 1. NLTK Setup
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
    # text = re.sub(PRIORITY_TERMS, '', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and len(w) > 1]
    return ' '.join(tokens)

# 2. Load Models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

print("Loading models...")
try:
    # Urgency
    u_body_model = joblib.load(os.path.join(MODELS_DIR, 'urgency_body_model.pkl'))
    u_body_tfidf = joblib.load(os.path.join(MODELS_DIR, 'urgency_body_tfidf.pkl'))
    u_subj_model = joblib.load(os.path.join(MODELS_DIR, 'urgency_subject_model.pkl'))
    u_subj_tfidf = joblib.load(os.path.join(MODELS_DIR, 'urgency_subject_tfidf.pkl'))

    # Type
    c_body_model = joblib.load(os.path.join(MODELS_DIR, 'class_body_model.pkl'))
    c_body_tfidf = joblib.load(os.path.join(MODELS_DIR, 'class_body_tfidf.pkl'))
    c_subj_model = joblib.load(os.path.join(MODELS_DIR, 'class_subject_model.pkl'))
    c_subj_tfidf = joblib.load(os.path.join(MODELS_DIR, 'class_subject_tfidf.pkl'))
except FileNotFoundError as e:
    print(f"Error loading models: {e}")
    print("Please make sure you have run train_models.py first.")
    exit(1)

def get_ensemble_pred(body_text, subj_text, body_model, body_tfidf, subj_model, subj_tfidf):
    # Vectorize
    b_vec = body_tfidf.transform([clean_text(body_text)])
    s_vec = subj_tfidf.transform([clean_text(subj_text)])
    
    # Predict Probas
    prob_b = body_model.predict_proba(b_vec)[0]
    prob_s = subj_model.predict_proba(s_vec)[0]
    
    # Average
    avg_prob = (prob_b + prob_s) / 2
    
    # Get Max Class
    idx = np.argmax(avg_prob)
    pred_class = body_model.classes_[idx]
    confidence = avg_prob[idx]
    
    return pred_class, confidence

# 3. Rule-Based Heuristics
def apply_urgency_rules(subject, body, current_pred, current_conf):
    """
    Boosts urgency to 'high' if critical keywords are found.
    """
    text = (str(subject) + " " + str(body)).lower()
    
    # Power Words for High Urgency
    high_triggers = [
        'deadline', 'asap', 'immediate', 'urgency', 'critical', 
        'breach', 'emergency', 'shuts down', 'exploded', 'security alert',
        'system down', 'outage', 'unacceptable'
    ]
    
    for word in high_triggers:
        if word in text:
            print(f"  [Rule Trigger] Found critical term: '{word}' -> Escalating to High")
            return 'high', 0.99 # Force High with high confidence
            
    return current_pred, current_conf

# 4. Test Function
def predict_email(subject, body):
    print(f"\n--- Analysis ---")
    print(f"Subject: {subject}")
    print(f"Body: {body[:100]}...") # Truncate body for display
    
    # Urgency (Ensemble)
    urgency, u_conf = get_ensemble_pred(
        body, subject, 
        u_body_model, u_body_tfidf, 
        u_subj_model, u_subj_tfidf
    )
    
    # Apply Rules
    urgency, u_conf = apply_urgency_rules(subject, body, urgency, u_conf)
    
    # Type (Ensemble)
    e_type, t_conf = get_ensemble_pred(
        body, subject, 
        c_body_model, c_body_tfidf, 
        c_subj_model, c_subj_tfidf
    )
    
    print(f"Predicted Urgency: {urgency} ({u_conf:.2%})")
    print(f"Predicted Type:    {e_type} ({t_conf:.2%})")

# 5. Main Loop
if __name__ == "__main__":
    print("\nModel Tester Ready (Type 'exit' to quit)")
    
    # Default Examples
    predict_email("URGENT: Server Down", "Our production server is not responding. Customers cannot login.")
    predict_email("Feedback on new feature", "I really liked the new dashboard update. Good job.")
    predict_email("Missed Deadline", "We have missed the critical submission deadline.")
    
    # Interactive
    while True:
        try:
            s = input("\nEnter Subject (or 'exit'): ")
            if s.lower() == 'exit': break
            b = input("Enter Body: ")
            predict_email(s, b)
        except EOFError:
            break
