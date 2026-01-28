@echo off
echo Setting up Face Detection App for production...

REM Install Python 3.9 if not present
echo Installing Python 3.9...
winget install Python.Python.3.9

REM Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    py -3.9 -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install requirements
echo Installing Python requirements...
pip install --no-cache-dir -r requirements.txt

echo Creating startup script for production...
(
echo @echo off
echo call venv\Scripts\activate.bat
echo echo Starting Face Detection App in production mode...
echo python production.py
echo pause
) > start_production.bat

REM Create directories if they don't exist
if not exist static\images mkdir static\images
if not exist templates mkdir templates

echo.
echo Setup complete! You can now:
echo 1. Run start_production.bat to start the server in production mode
echo 2. Access the application at http://localhost:8080
echo.
echo Note: The server will be accessible from other computers on your network.
echo      Use your computer's IP address instead of localhost to access from other devices.
echo.

pause