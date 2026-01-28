@echo off
setlocal EnableDelayedExpansion

echo Setting up Face Detection App for production...

REM Check for Python 3.9
python -V 2>NUL
if errorlevel 1 (
    echo Python is not found. Please install Python 3.9 from https://www.python.org/downloads/release/python-3913/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if exist venv (
    echo Removing old virtual environment...
    rmdir /s /q venv
)

echo Creating new virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install wheel first
echo Installing wheel...
pip install wheel

REM Install PyTorch CPU for better wheel compatibility
echo Installing PyTorch CPU...
pip install torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/cpu

REM Install requirements
echo Installing Python requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install requirements
    pause
    exit /b 1
)

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