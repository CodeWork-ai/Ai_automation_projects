#!/usr/bin/env python3
"""
Complete solution for SentencePiece and dependency issues
"""

import subprocess
import sys
import os
import json

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(f"🔧 {title}")
    print("=" * 60)

def run_command(command, description, critical=True):
    """Run a command with proper error handling"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            return True
        else:
            print(f"❌ {description} - FAILED")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()}")
            if critical:
                print(f"⚠️ This is a critical step. Please fix manually.")
            return False
    except Exception as e:
        print(f"💥 {description} - EXCEPTION: {e}")
        return False

def test_python_environment():
    """Test Python environment"""
    print_header("PYTHON ENVIRONMENT CHECK")
    
    # Check Python version
    python_version = sys.version
    print(f"🐍 Python Version: {python_version}")
    
    # Check pip
    run_command("pip --version", "Checking pip version", critical=False)
    
    # Check current directory
    current_dir = os.getcwd()
    print(f"📁 Current Directory: {current_dir}")
    
    # Check if requirements.txt exists
    if os.path.exists("requirements.txt"):
        print("✅ requirements.txt found")
        with open("requirements.txt", "r") as f:
            content = f.read()
            if "sentencepiece" in content:
                print("✅ sentencepiece listed in requirements.txt")
            else:
                print("❌ sentencepiece NOT in requirements.txt")
    else:
        print("❌ requirements.txt not found")

def fix_sentencepiece():
    """Fix SentencePiece installation"""
    print_header("SENTENCEPIECE INSTALLATION FIX")
    
    # Step 1: Upgrade pip
    run_command("python -m pip install --upgrade pip", "Upgrading pip")
    
    # Step 2: Clean install of sentencepiece
    print("\n📦 Cleaning and reinstalling SentencePiece...")
    run_command("pip uninstall sentencepiece -y", "Uninstalling old SentencePiece", critical=False)
    
    # Try multiple installation approaches
    installation_commands = [
        ("pip install sentencepiece==0.1.99", "Installing SentencePiece v0.1.99"),
        ("pip install --no-cache-dir sentencepiece", "Installing SentencePiece (no cache)"),
        ("pip install --upgrade --force-reinstall sentencepiece", "Force reinstalling SentencePiece"),
    ]
    
    for command, description in installation_commands:
        if run_command(command, description, critical=False):
            # Test if it works
            try:
                import sentencepiece
                print("✅ SentencePiece import successful!")
                break
            except ImportError:
                print("⚠️ Installation completed but import still fails")
                continue
    
    # Step 3: Install all requirements
    run_command("pip install -r requirements.txt", "Installing all requirements")

def test_dependencies():
    """Test all critical dependencies"""
    print_header("DEPENDENCY TESTING")
    
    dependencies = [
        ("sentencepiece", "SentencePiece"),
        ("transformers", "Transformers"),
        ("torch", "PyTorch"),
        ("fastapi", "FastAPI"),
        ("selenium", "Selenium"),
        ("requests", "Requests"),
        ("beautifulsoup4", "BeautifulSoup4"),
    ]
    
    failed_imports = []
    
    for module, name in dependencies:
        try:
            __import__(module)
            print(f"✅ {name} - OK")
        except ImportError as e:
            print(f"❌ {name} - FAILED: {e}")
            failed_imports.append(name)
    
    return len(failed_imports) == 0, failed_imports

def test_t5_tokenizer():
    """Test T5Tokenizer specifically"""
    print_header("T5TOKENIZER TESTING")
    
    try:
        print("🤖 Testing T5Tokenizer import...")
        from transformers import T5Tokenizer
        print("✅ T5Tokenizer imported successfully")
        
        print("🔧 Testing T5Tokenizer initialization...")
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        print("✅ T5Tokenizer initialized successfully")
        
        print("📝 Testing tokenization...")
        test_text = "Hello world, this is a test."
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        print(f"✅ Tokenization test passed!")
        print(f"   Original: {test_text}")
        print(f"   Tokens: {len(tokens)} tokens")
        print(f"   Decoded: {decoded}")
        
        return True
        
    except Exception as e:
        print(f"❌ T5Tokenizer test failed: {e}")
        return False

def create_test_server():
    """Create a simple test server to verify everything works"""
    test_server_code = '''
import sys
try:
    from transformers import T5Tokenizer
    from fastapi import FastAPI
    import sentencepiece
    
    print("✅ All imports successful!")
    
    # Test tokenizer
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    test_result = tokenizer.encode("Test successful!")
    
    print(f"✅ T5Tokenizer test passed! Generated {len(test_result)} tokens")
    print("🎉 Your environment is ready!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    sys.exit(1)
'''
    
    with open("test_environment.py", "w") as f:
        f.write(test_server_code)
    
    print("📝 Created test_environment.py")
    return run_command("python test_environment.py", "Running environment test")

def main():
    """Main execution function"""
    print("🚀 COMPLETE SENTENCEPIECE FIX SOLUTION")
    print("This script will fix all SentencePiece related issues")
    
    # Step 1: Test environment
    test_python_environment()
    
    # Step 2: Fix SentencePiece
    fix_sentencepiece()
    
    # Step 3: Test dependencies
    deps_ok, failed_deps = test_dependencies()
    
    # Step 4: Test T5Tokenizer specifically
    tokenizer_ok = test_t5_tokenizer()
    
    # Step 5: Create and run test
    test_ok = create_test_server()
    
    # Final results
    print_header("FINAL RESULTS")
    
    if deps_ok and tokenizer_ok and test_ok:
        print("🎉 SUCCESS! Everything is working correctly!")
        print("\n✅ Next steps:")
        print("   1. Run: python main.py")
        print("   2. Open: http://127.0.0.1:8000")
        print("   3. Start a research task to test the logging system")
        
        print("\n🔍 The enhanced logging system will show you:")
        print("   • Real-time scraping progress")
        print("   • Detailed error messages")
        print("   • Content extraction statistics")
        print("   • AI processing status")
        
    else:
        print("❌ Some issues remain:")
        if not deps_ok:
            print(f"   • Missing dependencies: {', '.join(failed_deps)}")
        if not tokenizer_ok:
            print("   • T5Tokenizer not working")
        if not test_ok:
            print("   • Environment test failed")
        
        print("\n🔧 Manual solutions to try:")
        print("   1. Restart your terminal/command prompt")
        print("   2. pip install --upgrade pip setuptools wheel")
        print("   3. pip install --upgrade transformers sentencepiece torch")
        print("   4. Create a new virtual environment")
        print("   5. Restart your computer")

if __name__ == "__main__":
    main()