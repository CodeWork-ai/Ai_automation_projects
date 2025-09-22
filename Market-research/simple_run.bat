@echo off
echo === Quick Start - Market Research Assistant ===
echo.

REM Change to the correct directory
cd /d "C:\Users\TCARE\Downloads\Market-research\Market-research"

REM Install SentencePiece specifically
echo Installing SentencePiece...
python -m pip install sentencepiece==0.1.99

REM Install other requirements if needed
echo Installing other requirements...
python -m pip install fastapi uvicorn transformers torch

REM Start the server
echo.
echo Starting server at http://localhost:8000
echo Press Ctrl+C to stop
echo.
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload

pause