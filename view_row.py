import pandas as pd
import sys
import os

RAW_DIR = r'c:\Users\Krish\Documents\GitHub\Enterprise-email-classifier\raw_data_sets'
FILE = os.path.join(RAW_DIR, 'balanced_dataset.csv')

def view_row(index):
    if not os.path.exists(FILE):
        print(f"File not found: {FILE}")
        return

    try:
        df = pd.read_csv(FILE)
        if index < 0 or index >= len(df):
            print(f"Index {index} out of bounds. Dataset has {len(df)} rows.")
            return

        row = df.iloc[index]
        print(f"\n--- Row {index} ---")
        print(f"Type:     {row.get('type')}")
        print(f"Priority: {row.get('priority')}")
        print(f"Subject:  {row.get('subject')}")
        print("-" * 20)
        print(f"Body:\n{row.get('body')}")
        print("-" * 20)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python view_row.py <row_index>")
        sys.exit(1)
    
    try:
        idx = int(sys.argv[1])
        view_row(idx)
    except ValueError:
        print("Please provide a valid integer index.")
