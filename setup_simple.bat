@echo off
echo Installing required packages...

REM Create virtual environment if it doesn't exist
if not exist venv (
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Upgrade pip
python -m pip install --upgrade pip

REM Install requirements
pip install -r requirements.txt

REM Run the simplified app
python app_simple.py

pause