@echo off
REM One-time setup: virtual environment, packages, .env, and a setup check.
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found. Install Python 3.10 or newer from python.org
    echo and tick "Add python.exe to PATH" during installation.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\activate.bat" (
    echo Creating virtual environment .venv ...
    python -m venv .venv
    if errorlevel 1 ( echo Could not create the virtual environment. & pause & exit /b 1 )
)
call ".venv\Scripts\activate.bat"

echo Installing packages ...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 ( echo Package installation failed. See the messages above. & pause & exit /b 1 )

if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo Created .env from .env.example
)

echo.
echo Checking the setup ...
python check_setup.py --network
echo.
echo If everything is OK, start the app with run_app.bat
pause
