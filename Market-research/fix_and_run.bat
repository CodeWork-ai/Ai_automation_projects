@echo off
echo === Market Research Assistant - Setup and Run ===
echo.

cd /d "C:\Users\TCARE\Downloads\Market-research\Market-research"

echo Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo Error: Python not found in PATH
    echo Please install Python or add it to your PATH
    pause
    exit /b 1
)

echo.
echo Installing/upgrading required packages...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo.
echo Testing imports...
python test_setup.py

echo.
echo Starting FastAPI server...
echo.
echo ========================================
echo   Server will start at: http://localhost:8000
echo   Press Ctrl+C to stop the server
echo ========================================
echo.

python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload