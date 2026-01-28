@echo off
echo =======================================
echo Testing FaceApp GPU Setup
echo =======================================
echo.

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Run the GPU test script
echo [INFO] Running GPU verification tests...
python test_gpu_setup.py

echo.
echo =======================================
echo GPU Test Complete!
echo =======================================
pause