#!/usr/bin/env python3
"""
Reliable server runner that handles dependency installation and server startup
"""
import subprocess
import sys
import os
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        # Install/upgrade pip first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def test_imports():
    """Test critical imports"""
    print("🔍 Testing imports...")
    try:
        # Test SentencePiece
        import sentencepiece
        print("✅ SentencePiece: OK")
        
        # Test transformers
        from transformers import T5Tokenizer
        print("✅ Transformers: OK")
        
        # Test FastAPI
        import fastapi
        print("✅ FastAPI: OK")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def start_server():
    """Start the FastAPI server"""
    print("🚀 Starting FastAPI server...")
    print("📍 Server will be available at: http://localhost:8000")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Use uvicorn directly
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--host", "127.0.0.1", 
            "--port", "8000", 
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")

def main():
    """Main function"""
    print("🔧 Market Research Assistant - Setup & Run")
    print("=" * 50)
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    print(f"📁 Working directory: {script_dir}")
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements. Please check the error messages above.")
        input("Press Enter to exit...")
        return
    
    # Test imports
    if not test_imports():
        print("❌ Import test failed. Please check the error messages above.")
        input("Press Enter to exit...")
        return
    
    print("✅ All checks passed!")
    print()
    
    # Start server
    start_server()

if __name__ == "__main__":
    main()