import sys
import importlib.util

print(f"Python Executable: {sys.executable}")
print(f"Path: {sys.path}")

try:
    import xlwt
    print(f"xlwt version: {xlwt.__version__}")
    print(f"xlwt file: {xlwt.__file__}")
except ImportError as e:
    print(f"ImportError: {e}")

try:
    import pandas
    print(f"pandas version: {pandas.__version__}")
except ImportError:
    print("pandas not found")
