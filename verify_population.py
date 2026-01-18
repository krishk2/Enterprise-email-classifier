import pandas as pd
import os

files = [
    "Unit_Test_Plan_v0.1.xlsx",
    "Defect_Tracker Template_v0.1.xlsx",
    "Agile_Template_v0.1.xlsx",
    "Sample-Kapil_Agile_Template.xlsx"
]

print("--- VERIFICATION REPORT ---")
for f in files:
    if not os.path.exists(f):
        print(f"[MISSING] {f}")
        continue
    
    try:
        xls = pd.ExcelFile(f)
        print(f"\nFile: {f}")
        for sheet in xls.sheet_names:
            df = pd.read_excel(f, sheet_name=sheet)
            if not df.empty:
                print(f"  Sheet: '{sheet}' - Valid (Rows: {len(df)})")
                # Print first non-header row to confirm data
                # Handle potential NaN
                first_row = df.iloc[0].tolist()
                print(f"    Sample: {first_row[:3]}...") # Print first 3 cols
            else:
                print(f"  Sheet: '{sheet}' - Empty")
    except Exception as e:
        print(f"  [ERROR] Reading {f}: {e}")
