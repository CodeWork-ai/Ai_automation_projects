@echo off
echo Installing SentencePiece dependency...
.venv\Scripts\pip install sentencepiece==0.1.99

echo.
echo Starting the FastAPI server...
echo You can access the UI at: http://localhost:8000
echo.

.venv\Scripts\uvicorn main:app --host 127.0.0.1 --port 8000 --reload