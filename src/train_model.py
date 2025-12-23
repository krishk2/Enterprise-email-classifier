import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def train_model():
    input_path = 'cleaned_data_sets/labeled_email_dataset.csv'
    model_path = 'src/email_classifier_model.pkl'
    vectorizer_path = 'src/tfidf_vectorizer.pkl'
    
    print(f"Loading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Could not find {input_path}")
        return

    # Basic preprocessing
    print("Preprocessing data...")
    # Drop rows with missing values in 'body' or 'category'
    df = df.dropna(subset=['body', 'category'])
    
    # Text and Label
    X = df['body']
    y = df['category']
    
    # Split data
    print("Splitting data into Train and Test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Vectorization
    print("Vectorizing text using TF-IDF...")
    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    # Model Training
    print("Training Logistic Regression model...")
    # increased max_iter for convergence
    model = LogisticRegression(max_iter=1000, multi_class='ovr', class_weight='balanced', random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    # Predictions
    print("Evaluating model...")
    y_pred = model.predict(X_test_tfidf)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred)
    print(report)
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    # Save Model and Vectorizer
    print(f"\nSaving model to {model_path} and vectorizer to {vectorizer_path}...")
    joblib.dump(model, model_path)
    joblib.dump(tfidf, vectorizer_path)
    print("Training complete!")

if __name__ == "__main__":
    train_model()
