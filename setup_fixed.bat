@echo off
setlocal EnableDelayedExpansion

echo =======================================
echo   FaceApp GPU Setup Script
echo =======================================

REM Check for NVIDIA GPU and CUDA
echo [INFO] Checking for NVIDIA GPU and CUDA support...
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [WARNING] NVIDIA GPU or drivers not detected!
    echo [WARNING] This setup requires NVIDIA GPU with CUDA support.
    echo [WARNING] Please install NVIDIA drivers and CUDA Toolkit 11.8 first.
    echo.
    echo Continue anyway? Press Ctrl+C to cancel, or any key to continue...
    pause
)

REM Check for Python 3.9+
python -V 2>NUL
if errorlevel 1 (
    echo [ERROR] Python is not found. Please install Python 3.9+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

REM Remove old virtual environment to start fresh
if exist venv (
    echo [INFO] Removing old virtual environment...
    echo [INFO] Checking for running Python processes...
    powershell -Command "Get-Process | Where-Object {$_.ProcessName -like '*python*' -and $_.Path -like '*venv*'} | ForEach-Object { Write-Host 'Stopping Python process:' $_.Id; Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }"
    timeout /t 2 /nobreak >nul
    rmdir /s /q venv
    if exist venv (
        echo [ERROR] Failed to remove old virtual environment
        echo [INFO] Trying alternative removal method...
        powershell -Command "Remove-Item -Path 'venv' -Recurse -Force -ErrorAction SilentlyContinue"
        timeout /t 2 /nobreak >nul
        if exist venv (
            echo [ERROR] Virtual environment is still locked. Please close any Python processes and try again.
            pause
            exit /b 1
        )
    )
)

echo [INFO] Creating new virtual environment...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment
    pause
    exit /b 1
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip first
echo [INFO] Upgrading pip...
python -m pip install --upgrade pip setuptools wheel

REM Install basic dependencies first
echo [INFO] Installing basic Python packages...
pip install flask==2.3.3
pip install werkzeug==2.3.7
pip install flask-cors==4.0.0
pip install "numpy<2.0"
pip install pillow==8.4.0
pip install opencv-python==4.5.5.64
pip install waitress==2.0.0
pip install Cython==0.29.32
pip install imagehash==4.3.1

REM Install PyTorch GPU version (CUDA 11.8)
echo [INFO] Installing PyTorch GPU version with CUDA support...
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

REM Install ONNX Runtime GPU
echo [INFO] Installing ONNX Runtime GPU...
pip install onnxruntime-gpu==1.15.1

REM Install scikit-learn
echo [INFO] Installing scikit-learn...
pip install scikit-learn==1.0.2

REM Install FAISS GPU
echo [INFO] Installing FAISS GPU...
pip install faiss-gpu==1.7.4

REM Install InsightFace (this might take a while)
echo [INFO] Installing InsightFace...
pip install insightface==0.7.3

REM Create directories if they don't exist
if not exist static\images mkdir static\images
if not exist templates mkdir templates

REM Test the installation
echo [INFO] Testing GPU installation...
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA devices:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"
python -c "import flask, cv2, numpy, insightface, faiss, onnxruntime; print('All dependencies installed successfully!')"
if errorlevel 1 (
    echo [ERROR] Some dependencies failed to install properly
    pause
    exit /b 1
)

REM Create a simple test script
echo [INFO] Creating GPU test script...
(
echo @echo off
echo call venv\Scripts\activate.bat
echo echo Testing FaceApp GPU dependencies...
echo python -c "import torch; print('✓ PyTorch GPU available:', torch.cuda.is_available()); print('✓ CUDA devices:', torch.cuda.device_count())"
echo python -c "import flask, cv2, numpy as np, insightface, faiss, onnxruntime; print('✓ All core dependencies working'); print('✓ Ready to run FaceApp with GPU acceleration')"
echo pause
) > test_gpu_dependencies.bat

REM Create production startup script
echo [INFO] Creating production startup script...
(
echo @echo off
echo call venv\Scripts\activate.bat
echo echo Starting FaceApp in production mode...
echo python production.py
echo pause
) > start_production_fixed.bat

REM Create development startup script
echo [INFO] Creating development startup script...
(
echo @echo off
echo call venv\Scripts\activate.bat
echo echo Starting FaceApp in development mode...
echo python app.py
echo pause
) > start_development.bat

echo.
echo =======================================
echo   GPU Setup Complete!
echo =======================================
echo.
echo You can now:
echo 1. Run test_gpu_dependencies.bat to verify GPU support
echo 2. Run start_development.bat for development mode
echo 3. Run start_production_fixed.bat for production mode
echo.
echo The app will be available at:
echo - Development: http://localhost:5000
echo - Production: http://localhost:8080
echo.
echo GPU Performance Notes:
echo - First run downloads InsightFace models (~100MB)
echo - GPU acceleration significantly improves face processing speed
echo - Ensure CUDA Toolkit 11.8 is installed for optimal performance
echo.

pause