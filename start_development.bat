@echo off
echo =======================================
  FaceApp GPU Development Server
=======================================

echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat
set PROCESSING_MODE=gpu
echo [INFO] Defaulting to GPU processing...

echo [INFO] Starting FaceApp in development mode...
echo [INFO] Server will be available at: http://localhost:5000
echo [INFO] Debug mode enabled - auto-reload on code changes
echo [INFO] Press Ctrl+C to stop the server
echo.

set FLASK_ENV=development
set FLASK_DEBUG=1
python app.py

pause