@echo off
setlocal
py -3.11 --version >nul 2>nul
if %errorlevel%==0 (
  py -3.11 -m venv .venv
) else (
  python -m venv .venv
)
if errorlevel 1 (
  echo Could not create the virtual environment. Install Python 3.11+ and ensure Python is on PATH.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo.
  echo Installation failed. Check Python version and internet connection, then retry.
  pause
  exit /b 1
)
echo.
echo Setup complete. Run run_windows.bat to start the demo.
pause
