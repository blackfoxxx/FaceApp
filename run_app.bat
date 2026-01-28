@echo off
setlocal EnableDelayedExpansion

echo Setting up Face Detection App for production...

REM Use Python 3.9 explicitly
set PYTHON_CMD=py -3.9
%PYTHON_CMD% --version > nul 2>&1
if errorlevel 1 (
    echo Python 3.9 is required. Please install Python 3.9 from https://www.python.org/downloads/release/python-3913/
    echo Available Python versions:
    py -0
    pause
    exit /b 1
)

REM Check for Visual C++ Build Tools
where cl > nul 2>&1
if errorlevel 1 (
    echo Visual C++ Build Tools are required.
    echo Please install Visual Studio Build Tools from:
    echo https://visualstudio.microsoft.com/visual-cpp-build-tools/
    echo Make sure to select "Desktop development with C++"
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if exist venv (
    echo Removing old virtual environment...
    rmdir /s /q venv
)

echo Creating new virtual environment...
%PYTHON_CMD% -m venv venv

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip and install build tools
python -m pip install --upgrade pip wheel setuptools

REM Install requirements
echo Installing Python requirements...
pip install --no-cache-dir -r requirements.txt
if errorlevel 1 (
    echo Failed to install requirements
    pause
    exit /b 1
)

REM Create directories if they don't exist
if not exist static\images mkdir static\images
if not exist templates mkdir templates

echo Starting Face Detection App in production mode...
%PYTHON_CMD% production.py

pause