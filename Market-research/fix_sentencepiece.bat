@echo off
echo 🚀 Fixing SentencePiece Installation
echo ================================

echo.
echo 🔧 Step 1: Upgrading pip...
python -m pip install --upgrade pip

echo.
echo 📦 Step 2: Fixing SentencePiece...
pip uninstall sentencepiece -y
pip install --no-cache-dir sentencepiece==0.1.99

echo.
echo 📋 Step 3: Installing all requirements...
pip install -r requirements.txt

echo.
echo 🧪 Step 4: Testing installation...
python fix_sentencepiece.py

echo.
echo ✅ Fix completed! 
echo.
echo 🚀 To start the server:
echo    python main.py
echo.
echo 🌐 Then open: http://127.0.0.1:8000
echo.
pause