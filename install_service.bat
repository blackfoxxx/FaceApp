@echo off
echo =======================================
echo Installing FaceApp Windows Service
echo =======================================

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% == 0 (
    echo Running as Administrator - OK
) else (
    echo ERROR: This script must be run as Administrator
    echo Right-click and select "Run as administrator"
    pause
    exit /b 1
)

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install required Windows service dependencies
echo [INFO] Installing Windows service dependencies...
pip install pywin32

REM Install the service
echo [INFO] Installing FaceApp service...
python faceapp_service.py install

REM Start the service
echo [INFO] Starting FaceApp service...
python faceapp_service.py start

echo.
echo =======================================
echo FaceApp Service Installation Complete!
echo =======================================
echo.
echo Service Name: FaceAppService
echo Display Name: FaceApp Face Recognition Service
echo Status: Started
echo.
echo The FaceApp will now start automatically when Windows boots.
echo You can manage the service through Windows Services (services.msc)
echo or use the following commands:
echo.
echo   Start:   python faceapp_service.py start
echo   Stop:    python faceapp_service.py stop
echo   Remove:  python faceapp_service.py remove
echo.
echo FaceApp should be accessible at: http://localhost:8080
echo.
pause