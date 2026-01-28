@echo off
echo =======================================
echo Building FaceApp GPU Standalone Executable
echo =======================================
echo.

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install GPU requirements first
echo [INFO] Installing GPU requirements...
pip install -r requirements.txt --upgrade

REM Run the GPU build script
echo [INFO] Starting GPU build process...
python build_exe_gpu.py

echo.
echo =======================================
echo GPU Build Process Complete!
echo =======================================
echo.
echo Check the 'dist\FaceApp' folder for the GPU-enabled executable.
echo Use 'run_faceapp_gpu.bat' to launch with GPU support.
echo.
pause