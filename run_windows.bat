@echo off
call .venv\Scripts\activate.bat
if errorlevel 1 (
  echo Virtual environment not found. Run setup_windows.bat first.
  pause
  exit /b 1
)
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
