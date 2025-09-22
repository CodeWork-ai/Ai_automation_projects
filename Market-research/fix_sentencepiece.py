#!/usr/bin/env python3
"""
Fix SentencePiece installation and test the setup
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and return success status"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} - Success")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ {description} - Failed")
            if result.stderr.strip():
                print(f"   Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"💥 {description} - Exception: {e}")
        return False

def test_imports():
    """Test if all required libraries can be imported"""
    print("\n🧪 Testing imports...")
    
    test_cases = [
        ("sentencepiece", "SentencePiece library"),
        ("transformers", "Transformers library"),
        ("torch", "PyTorch"),
    ]
    
    all_passed = True
    
    for module, description in test_cases:
        try:
            __import__(module)
            print(f"✅ {description} - OK")
        except ImportError as e:
            print(f"❌ {description} - Failed: {e}")
            all_passed = False
    
    return all_passed

def test_t5_tokenizer():
    """Test T5Tokenizer specifically"""
    print("\n🤖 Testing T5Tokenizer...")
    try:
        from transformers import T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained('t5-small')
        test_text = "Hello world"
        tokens = tokenizer.encode(test_text)
        print(f"✅ T5Tokenizer works! Encoded '{test_text}' to {len(tokens)} tokens")
        return True
    except Exception as e:
        print(f"❌ T5Tokenizer failed: {e}")
        return False

def main():
    """Main fix and test function"""
    print("🚀 SentencePiece Installation Fix")
    print("=" * 50)
    
    # Step 1: Upgrade pip
    run_command("python -m pip install --upgrade pip", "Upgrading pip")
    
    # Step 2: Uninstall and reinstall sentencepiece
    print("\n📦 Fixing SentencePiece installation...")
    run_command("pip uninstall sentencepiece -y", "Uninstalling old SentencePiece")
    
    # Try different installation methods
    installation_methods = [
        ("pip install sentencepiece==0.1.99", "Installing SentencePiece v0.1.99"),
        ("pip install sentencepiece", "Installing latest SentencePiece"),
        ("pip install --no-cache-dir sentencepiece", "Installing SentencePiece (no cache)"),
    ]
    
    sentencepiece_installed = False
    for command, description in installation_methods:
        if run_command(command, description):
            # Test if it works
            try:
                import sentencepiece
                print(f"✅ SentencePiece successfully installed!")
                sentencepiece_installed = True
                break
            except ImportError:
                print(f"⚠️ Installation completed but import still fails")
                continue
    
    if not sentencepiece_installed:
        print("\n💡 Trying alternative installation method...")
        # Try installing from conda-forge if available
        run_command("pip install --upgrade --force-reinstall sentencepiece", 
                   "Force reinstalling SentencePiece")
    
    # Step 3: Install all requirements
    print("\n📋 Installing all requirements...")
    run_command("pip install -r requirements.txt", "Installing all requirements")
    
    # Step 4: Test everything
    print("\n" + "=" * 50)
    print("🧪 TESTING PHASE")
    print("=" * 50)
    
    imports_ok = test_imports()
    tokenizer_ok = test_t5_tokenizer()
    
    print("\n" + "=" * 50)
    print("📊 RESULTS")
    print("=" * 50)
    
    if imports_ok and tokenizer_ok:
        print("🎉 SUCCESS! All libraries are working correctly.")
        print("\n✅ You can now run your market research application:")
        print("   python main.py")
        print("\n🌐 Then open: http://127.0.0.1:8000")
    else:
        print("❌ Some issues remain. Try these solutions:")
        print("\n🔧 Manual fixes:")
        print("1. Restart your terminal/command prompt")
        print("2. Try: pip install --upgrade transformers sentencepiece")
        print("3. If using conda: conda install -c conda-forge sentencepiece")
        print("4. Restart your computer if needed")
        
        print("\n📞 If problems persist:")
        print("- Check Python version: python --version")
        print("- Check pip version: pip --version")
        print("- Try creating a new virtual environment")

if __name__ == "__main__":
    main()