
import zipfile
import os

def zip_project(output_filename):
    print(f"Creating {output_filename}...")
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk('.'):
            # Exclude hidden directories and venv
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.gemini', '.vscode', 'venv', '.venv', '.streamlit']]
            
            for file in files:
                # Exclude secrets, the zip itself, and pyc files
                if file in ['.env', output_filename, 'package_project.py'] or file.endswith('.pyc'):
                    continue
                
                # Exclude raw data if it's huge? (Optionally) - keeping it for now as per request "project zip"
                
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, '.')
                zipf.write(file_path, arcname)
                print(f"Added: {arcname}")
    print("Done!")

if __name__ == "__main__":
    zip_project('Enterprise-email-classifier.zip')
