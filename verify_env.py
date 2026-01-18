import sys
import os

print(f"Executable: {sys.executable}")
print(f"CWD: {os.getcwd()}")
print("Sys Path:")
for p in sys.path:
    print(f"  {p}")

print("\nChecking site-packages for torch...")
site_packages = [p for p in sys.path if 'site-packages' in p]
for sp in site_packages:
    torch_path = os.path.join(sp, 'torch')
    if os.path.exists(torch_path):
        print(f"  Found torch at: {torch_path}")
    else:
        print(f"  No torch in: {sp}")

print("\nAttempting Import...")
try:
    import torch
    print(f"Successfully imported torch from {torch.__file__}")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
