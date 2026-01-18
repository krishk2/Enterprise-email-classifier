import pandas as pd
try:
    df = pd.read_csv(r'c:\Users\Krish\Documents\GitHub\Enterprise-email-classifier\raw_data_sets\balanced_dataset.csv')
    print("--- DATASET COUNTS ---")
    print(df['type'].value_counts())
    print("----------------------")
except Exception as e:
    print(e)
