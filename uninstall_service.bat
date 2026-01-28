@echo off
echo =======================================
echo Uninstalling FaceApp Windows Service
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

REM Stop the service
echo [INFO] Stopping FaceApp service...
python faceapp_service.py stop

REM Remove the service
echo [INFO] Removing FaceApp service...
python faceapp_service.py remove

echo.
echo =======================================
echo FaceApp Service Uninstallation Complete!
echo =======================================
echo.
echo The FaceApp service has been removed from Windows.
echo FaceApp will no longer start automatically with Windows.
echo.
echo You can still run FaceApp manually using:
echo   start_production_fixed.bat
echo.
pause