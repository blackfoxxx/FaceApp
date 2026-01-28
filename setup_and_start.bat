@echo off
setlocal EnableDelayedExpansion

echo =======================================
echo   FaceApp Setup & Start
echo =======================================
echo.
echo Usage: setup_and_start.bat [gpu|cpu]
echo   gpu - install CUDA-enabled packages (PyTorch/ONNX Runtime GPU)
echo   cpu - install CPU-only packages (PyTorch/ONNX Runtime CPU)
echo   (default: auto, leaves existing packages)
echo.

REM Determine mode from first argument
set MODE=%1
if "%MODE%"=="" set MODE=gpu

REM Check Python availability
python -V >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9+ and add to PATH.
    pause
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist venv (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

REM Activate virtual environment
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate the virtual environment.
    pause
    exit /b 1
)

echo [INFO] Upgrading pip, setuptools, and wheel...
python -m pip install --upgrade pip setuptools wheel

echo [INFO] Installing base requirements...
pip install --no-cache-dir -r requirements.txt
if errorlevel 1 (
    echo [WARN] Base requirements install failed. Retrying...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install base requirements.
        pause
        exit /b 1
    )
)

REM Ensure production server package
pip install waitress

REM Optional GPU/CPU overlays
if /I "%MODE%"=="gpu" (
    echo [INFO] Installing GPU packages (PyTorch CUDA + ONNX Runtime GPU)...
    pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
    if errorlevel 1 (
        echo [ERROR] Failed to install PyTorch CUDA packages.
        pause
        exit /b 1
    )
    pip install onnxruntime-gpu==1.15.1
    if errorlevel 1 (
        echo [ERROR] Failed to install ONNX Runtime GPU.
        pause
        exit /b 1
    )
) else if /I "%MODE%"=="cpu" (
    echo [INFO] Installing CPU packages (PyTorch CPU + ONNX Runtime CPU)...
    pip install torch==2.0.1+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/cpu
    if errorlevel 1 (
        echo [ERROR] Failed to install PyTorch CPU packages.
        pause
        exit /b 1
    )
    pip install onnxruntime==1.15.1
    if errorlevel 1 (
        echo [ERROR] Failed to install ONNX Runtime CPU.
        pause
        exit /b 1
    )
) else (
    echo [INFO] Auto mode: keeping existing packages. Use 'gpu' or 'cpu' to force overlays.
)

REM Ensure required directories
if not exist static\images mkdir static\images
if not exist templates mkdir templates

echo.
REM Set processing mode environment variable based on chosen MODE
if /I "%MODE%"=="gpu" (
    set PROCESSING_MODE=gpu
    echo [INFO] Defaulting to GPU processing...
) else if /I "%MODE%"=="cpu" (
    set PROCESSING_MODE=cpu
    echo [INFO] Defaulting to CPU processing...
) else (
    set PROCESSING_MODE=gpu
    echo [INFO] Auto mode detected; defaulting to GPU processing...
)

echo [INFO] Starting FaceApp server (production)...
echo [INFO] Server will be available at: http://localhost:8080
echo [INFO] A new window will open for the server.
start "FaceApp Server" python production.py

echo [INFO] Opening browser to the app...
start "" http://localhost:8080/

echo.
echo [INFO] Setup complete. The server is running in a separate window.
echo       Use Ctrl+C in that window to stop the server.
pause