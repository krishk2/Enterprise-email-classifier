import pandas as pd
import os

files = [
    "Unit_Test_Plan_v0.1.xlsx",
    "Defect_Tracker Template_v0.1.xlsx",
    "Agile_Template_v0.1.xls",
    "Sample-Kapil_Agile_Template.xls"
]

for f in files:
    if not os.path.exists(f):
        print(f"File not found: {f}")
        continue
    
    print(f"\n--- Inspecting {f} ---")
    try:
        # Load excel file to check sheet names
        xls = pd.ExcelFile(f)
        print(f"Sheets: {xls.sheet_names}")
        
        for sheet in xls.sheet_names:
            print(f"  Sheet: {sheet}")
            # Read first few rows/cols to guess headers
            df = pd.read_excel(f, sheet_name=sheet, nrows=5)
            print(f"  Columns: {list(df.columns)}")
            print("  First row values (sample):")
            if not df.empty:
                print(df.iloc[0].tolist())
            else:
                print("  (Empty sheet)")
            
    except Exception as e:
        print(f"Error reading {f}: {e}")
