# Customer IT Support – Email Classification Dataset

## Overview

This project focuses on **automatic e-mail classification** for enterprise IT support systems.
The objective is to classify incoming emails into meaningful categories and assign priorities using **Natural Language Processing and Machine Learning** techniques.

The dataset simulates real corporate inbox traffic, including noisy text, inconsistent casing, HTML content, and contact details.

---

## Problem Statement

Enterprise IT support teams receive a high volume of emails daily. These emails vary in intent and urgency and include spam, complaints, feedback, and service requests.

Manual sorting and prioritization lead to delays, inconsistency, and operational inefficiency.
This project addresses the problem by building an **automated e-mail classification pipeline**.

---

## Objective of Email Classification

The main objectives are:

1. Automatically classify emails into predefined categories
2. Reduce manual effort in e-mail triaging
3. Enable faster response and escalation
4. Build a scalable baseline NLP system using TF-IDF

---

## Email Categories

Each email belongs to one of the following four classes:

SPAM
complaint
feedback
request

These categories represent common patterns observed in enterprise IT support inboxes.

---

## Priority Levels

Each email is also assigned a priority level:

low
medium
high

This allows models to support **priority-based routing and escalation**.

---

## Dataset Description

The dataset consists of enterprise-style email messages stored in CSV format.

Each row represents one email and contains:

id
subject
body
label
priority

The dataset is balanced across all four labels to ensure fair model training.

---

## Data Characteristics

The dataset intentionally includes realistic noise such as:

Random capitalization
HTML links
Email addresses
Lengthy and varied text content

These characteristics help test preprocessing robustness and model generalization.

---

## Project Structure

The project follows a clean ML workflow structure:

Raw datasets are stored separately
Cleaned datasets are maintained for modeling
Train and test splits are isolated
Experiments are conducted in Jupyter notebooks

This separation improves reproducibility and clarity.

---

## Email Classification Pipeline

The typical pipeline used in this project is:

1. Load raw email data
2. Clean and normalize text
3. Combine subject and body fields
4. Convert text to numerical features
5. Train classification models
6. Evaluate performance

---

## Feature Engineering (TF-IDF)

TF-IDF vectorization is used to convert email text into numerical representations.

TF-IDF helps by:

Reducing the impact of common words
Highlighting discriminative terms
Improving linear model performance

Unigrams and bigrams are used to capture both keywords and short phrases.

---

## Model Training and Evaluation

The following models are suitable for this dataset:

Logistic Regression
Naive Bayes
Linear Support Vector Machine

Evaluation is performed using accuracy, precision, recall, and F1-score.

---

## Installation and Setup

### Create a virtual environment

```bash
python -m venv .venv
```

### Activate the environment

Windows

```bash
.venv\Scripts\activate
```

Linux or macOS

```bash
source .venv/bin/activate
```

---

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## How to Run the Project

Open the notebook and execute cells sequentially:

```bash
jupyter notebook notebooks-kaggle/email-data.ipynb
```

The notebook covers preprocessing, TF-IDF vectorization, training, and evaluation.

---

## Use Cases

This project can be extended to:

Enterprise email triaging
IT support automation
Spam filtering systems
Priority-based ticket routing
Academic NLP projects

---

## Technologies Used

Python
pandas
NumPy
scikit-learn
Jupyter Notebook

---

## Results and Observations

TF-IDF combined with linear classifiers provides strong baseline performance for email classification tasks.
Balanced data improves per-class recall and interpretability.

---

## Limitations

TF-IDF does not capture semantic meaning or context.
Performance may drop on very short or ambiguous emails.

---

## Future Improvements

Possible enhancements include:

Using word embeddings or transformer models
Handling multilingual emails
Adding thread-level context
Deploying as an API-based service

---

## License

This project is released under the **MIT License**.
All data is synthetically generated and safe for academic and personal use.

---

## Author

Developed for **enterprise e-mail classification and NLP experimentation**.

---
