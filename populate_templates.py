import pandas as pd
import os
import shutil
from datetime import datetime

# --- Data Definitions ---

test_cases = [
    {
        "Sl: No:": 1,
        "Test Case Name": "Load Raw Data",
        "Test Procedure": "Call load_data() with 'complaints-500.csv'",
        "Condition to be tested": "File exists and format is valid",
        "Expected Result": "DataFrame loaded with 500 rows",
        "Actual Result": "Pass"
    },
    {
        "Sl: No:": 2,
        "Test Case Name": "Clean Text - HTML Removal",
        "Test Procedure": "Pass string with <div> tags to clean_text()",
        "Condition to be tested": "Input contains HTML tags",
        "Expected Result": "String without HTML tags",
        "Actual Result": "Pass"
    },
    {
        "Sl: No:": 3,
        "Test Case Name": "TF-IDF Vectorization",
        "Test Procedure": "Fit TfidfVectorizer on clean text",
        "Condition to be tested": "Vocabulary is generated",
        "Expected Result": "Sparse matrix returned",
        "Actual Result": "Pass"
    },
    {
        "Sl: No:": 4,
        "Test Case Name": "Model Training - LogisticRegression",
        "Test Procedure": "Train model on training set",
        "Condition to be tested": "Training completes without error",
        "Expected Result": "Model object created",
        "Actual Result": "Pass"
    }
]

defects = [
    {
        "Sl No": 1,
        "Submitted By": "QA Team",
        "Submitted Date": "2025-10-01",
        "Description": "UnicodeDecodeError when reading some email bodies",
        "Detected Sprint": "Sprint 1",
        "Assigned To": "Data Engineer",
        "Type Of Defect": "Functional",
        "Action Taken": "Added encoding='utf-8' to read_csv",
        "Action Taken Date": "2025-10-02",
        "Status(Open/Closed)": "Closed",
        "Remarks": "Fixed"
    },
    {
        "Sl No": 2,
        "Submitted By": "QA Team",
        "Submitted Date": "2025-10-05",
        "Description": "Low recall on 'Request' category",
        "Detected Sprint": "Sprint 2",
        "Assigned To": "Data Scientist",
        "Type Of Defect": "Model Performance",
        "Action Taken": "Tuning hyperparameters",
        "Action Taken Date": "",
        "Status(Open/Closed)": "Open",
        "Remarks": "Investigating class imbalance impact"
    }
]

backlog = [
    {
        "Planned Sprint": "Sprint 1",
        "Actual Sprint": "Sprint 1",
        "US ID": "US-001",
        "User Story Description": "As a Data Engineer, I want to load and inspect the raw email dataset.",
        "MOSCOW": "Must Have",
        "Dependency": "None",
        "Assignee": "Krish",
        "Status": "Completed"
    },
    {
        "Planned Sprint": "Sprint 1",
        "Actual Sprint": "Sprint 1",
        "US ID": "US-002",
        "User Story Description": "As a Data Engineer, I want to clean email text (remove HTML, special chars).",
        "MOSCOW": "Must Have",
        "Dependency": "US-001",
        "Assignee": "Krish",
        "Status": "Completed"
    },
    {
        "Planned Sprint": "Sprint 2",
        "Actual Sprint": "Sprint 2",
        "US ID": "US-003",
        "User Story Description": "As a Data Scientist, I want to vectorize text using TF-IDF.",
        "MOSCOW": "Must Have",
        "Dependency": "US-002",
        "Assignee": "Krish",
        "Status": "In Progress"
    },
    {
        "Planned Sprint": "Sprint 2",
        "Actual Sprint": "",
        "US ID": "US-004",
        "User Story Description": "As a Data Scientist, I want to train a Logistic Regression model.",
        "MOSCOW": "Must Have",
        "Dependency": "US-003",
        "Assignee": "Krish",
        "Status": "To Do"
    }
]

# --- Processing Functions ---

def populate_xlsx(filename, data, sheet_name=0):
    print(f"Populating {filename}...")
    try:
        if os.path.exists(filename):
            # Read existing to preserve header style if possible, but pandas rewrite wipes style usually.
            # For simplicity, we strictly write data.
            # To preserve headers, we can use openpyxl directly, but pandas is easier.
            # Let's try to append to existing structure using openpyxl to keep styles if possible,
            # or just overwrite with pandas if header match.
            
            # Simple approach: Load into DF, append data, write back.
            # Note: This removes formatting.
            
            # Better approach for templates: existing file has headers.
            # We should load it, find the columns, fill the rows, and save.
            
            df_existing = pd.read_excel(filename, sheet_name=sheet_name)
            df_new = pd.DataFrame(data)
            
            # Align columns
            for col in df_existing.columns:
                if col not in df_new.columns:
                    df_new[col] = "" # Add missing cols as empty
            
            # Reorder columns to match template
            df_final = df_new[df_existing.columns]
            
            # Write back
            # Using openpyxl engine for xlsx
            with pd.ExcelWriter(filename, engine='openpyxl', mode='w') as writer:
                df_final.to_excel(writer, sheet_name=sheet_name if isinstance(sheet_name, str) else 'Sheet1', index=False)
            print(f"Successfully updated {filename}")
        else:
            print(f"File {filename} not found.")
    except Exception as e:
        print(f"Error updating {filename}: {e}")

def populate_xls(filename, data, sheet_name_contains="Backlog"):
    print(f"Populating {filename}...")
    try:
        # Saving as .xlsx to avoid xlwt issues and ensuring compatibility
        new_filename = filename + "x" if not filename.endswith("x") else filename
        
        # Read using read_excel (it handles xls via xlrd if installed, which we have)
        if os.path.exists(filename):
             # Find likely sheet
            xls_file = pd.ExcelFile(filename)
            target_sheet = None
            for s in xls_file.sheet_names:
                if sheet_name_contains.lower() in s.lower():
                    target_sheet = s
                    break
            if not target_sheet:
                target_sheet = xls_file.sheet_names[0]
            
            # Read existing
            try:
                df_existing = pd.read_excel(filename, sheet_name=target_sheet)
            except Exception:
                # Fallback if header issues
                df_existing = pd.read_excel(filename, sheet_name=target_sheet, header=1)
            
            df_new = pd.DataFrame(data)
            
             # Align columns
            for col in df_existing.columns:
                if col not in df_new.columns:
                    df_new[col] = "" 
            
            df_final = df_new[df_existing.columns] if set(df_existing.columns).issubset(df_new.columns) else df_new
            
            # Write using openpyxl (xlsx)
            with pd.ExcelWriter(new_filename, engine='openpyxl', mode='w') as writer:
                df_final.to_excel(writer, sheet_name=target_sheet, index=False)
            
            print(f"Successfully updated {filename} (saved as {new_filename})")

    except Exception as e:
        print(f"Error updating {filename}: {e}")

# --- Execution ---

if __name__ == "__main__":
    # 1. Unit Test Plan
    populate_xlsx("Unit_Test_Plan_v0.1.xlsx", test_cases, sheet_name="UT")

    # 2. Defect Tracker
    populate_xlsx("Defect_Tracker Template_v0.1.xlsx", defects, sheet_name="Defects")

    # 3. Agile Template (xls)
    # We populate 'Product Backlog' sheet roughly
    populate_xls("Agile_Template_v0.1.xls", backlog, sheet_name_contains="Product Backlog")
    
    # 4. Sample-Kapil (updates)
    populate_xls("Sample-Kapil_Agile_Template.xls", backlog, sheet_name_contains="Product Backlog")
