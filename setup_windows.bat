@echo off
setlocal enabledelayedexpansion

echo =======================================
echo   FaceApp Windows Setup Script
echo =======================================

echo [INFO] Checking for Python 3.10+ on PATH...
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python was not found. Install Python 3.10 or newer from https://www.python.org/downloads/ and re-run this script.
    exit /b 1
)

for /f "tokens=2 delims= " %%P in ('python -V') do set PY_VERSION=%%P
echo [INFO] Detected Python %PY_VERSION%

set "VENV_DIR=venv"

if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [INFO] Using existing virtual environment: %VENV_DIR%
) else (
    echo [INFO] Creating virtual environment in %VENV_DIR% ...
    python -m venv "%VENV_DIR%"
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        exit /b 1
    )
)

echo [INFO] Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate the virtual environment.
    exit /b 1
)

echo [INFO] Upgrading pip, setuptools, and wheel...
python -m pip install --upgrade pip setuptools wheel
if %errorlevel% neq 0 (
    echo [ERROR] Failed to upgrade packaging tools.
    exit /b 1
)

echo [INFO] Installing project requirements...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [WARN] Initial requirements install failed. Attempting to install faiss-cpu from PyTorch wheel index and retry...
    pip install faiss-cpu==1.7.4 --extra-index-url https://download.pytorch.org/whl/cpu
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install faiss-cpu. Please install Visual C++ Build Tools and re-run.
        exit /b 1
    )
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Requirements installation still failing. Check the console output above for details.
        exit /b 1
    )
)

echo.
echo =======================================
echo   Setup Completed Successfully
echo =======================================
echo.
echo [INFO] To start the application in this shell, run:
echo    set FLASK_APP=app.py
echo    flask run --host=0.0.0.0 --port=5000
echo.
echo [INFO] The virtual environment will stay active for this window. To activate later, run:
echo    %VENV_DIR%\Scripts\activate
echo.
echo [NOTE] If you need InsightFace GPU support, install CUDA Toolkit that matches your GPU drivers before running the app.

endlocal
