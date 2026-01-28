@echo off
echo Setting up Face Detection App for production...

REM Create virtual environment if it doesn't exist
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install Microsoft Visual C++ Build Tools silently
echo Installing Visual C++ Build Tools...
winget install Microsoft.VisualStudio.2019.BuildTools --override "--wait --quiet --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64"

REM Install requirements
echo Installing Python requirements...
pip install -r requirements.txt

REM Install production server
echo Installing production WSGI server...
pip install waitress

REM Create startup script for production
echo Creating production startup script...
(
echo @echo off
echo call venv\Scripts\activate.bat
echo echo Starting Face Detection App in production mode...
echo python -c "from waitress import serve; from app import app; serve(app, host='0.0.0.0', port=8080)"
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