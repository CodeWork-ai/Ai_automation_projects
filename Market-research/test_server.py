import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from main import app
    print("✅ Main module imported successfully")
    print("✅ FastAPI app created successfully")
    print("🚀 Ready to start server!")
except Exception as e:
    print(f"❌ Error importing main module: {e}")
    import traceback
    traceback.print_exc()