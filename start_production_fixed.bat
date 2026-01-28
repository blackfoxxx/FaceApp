@echo off
setlocal

 =======================================
  FaceApp GPU Production Server
=======================================

echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat
set PROCESSING_MODE=gpu
echo [INFO] Defaulting to GPU processing...

echo [INFO] Starting FaceApp with GPU acceleration...
echo [INFO] Server will be available at: http://localhost:8080
echo [INFO] Press Ctrl+C to stop the server
echo.

python production.py

pause