#!/usr/bin/env python3
"""
Test script to verify all dependencies are installed correctly
"""

def test_imports():
    """Test if all required packages can be imported"""
    try:
        print("Testing imports...")
        
        # Test SentencePiece
        import sentencepiece
        print("✓ SentencePiece imported successfully")
        
        # Test transformers with T5
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        print("✓ Transformers (T5) imported successfully")
        
        # Test other dependencies
        import torch
        print("✓ PyTorch imported successfully")
        
        import fastapi
        print("✓ FastAPI imported successfully")
        
        import requests
        print("✓ Requests imported successfully")
        
        import bs4
        print("✓ BeautifulSoup4 imported successfully")
        
        print("\n✅ All dependencies are working correctly!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_tokenizer():
    """Test if T5 tokenizer works with SentencePiece"""
    try:
        print("\nTesting T5 tokenizer...")
        from transformers import T5Tokenizer
        
        # Try to load the tokenizer (this will test SentencePiece integration)
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base", legacy=False)
        
        # Test tokenization
        test_text = "This is a test sentence."
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        
        print(f"✓ Original: {test_text}")
        print(f"✓ Tokens: {tokens}")
        print(f"✓ Decoded: {decoded}")
        print("✅ T5 tokenizer is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Tokenizer error: {e}")
        return False

if __name__ == "__main__":
    print("=== Market Research Assistant - Dependency Test ===\n")
    
    imports_ok = test_imports()
    tokenizer_ok = test_tokenizer()
    
    if imports_ok and tokenizer_ok:
        print("\n🎉 Everything is set up correctly! You can now run the server.")
        print("Run: python main.py")
    else:
        print("\n❌ Some issues were found. Please install missing dependencies.")
        print("Run: pip install sentencepiece==0.1.99")