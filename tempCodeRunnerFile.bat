
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
