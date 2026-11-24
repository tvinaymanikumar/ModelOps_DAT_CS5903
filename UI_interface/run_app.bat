@echo off
REM Start the app. Run setup_windows.bat once before the first start.
cd /d "%~dp0"
if not exist ".venv\Scripts\activate.bat" (
    echo The virtual environment is missing. Run setup_windows.bat first.
    pause
    exit /b 1
)
call ".venv\Scripts\activate.bat"
python -m streamlit run app.py
pause
