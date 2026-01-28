@echo off
echo =======================================
echo Starting FaceApp Executable
echo =======================================
echo.
echo FaceApp will be available at: http://localhost:8080
echo Press Ctrl+C to stop the application
echo.
cd dist\FaceApp
echo Starting FaceApp.exe...
start http://localhost:8080
FaceApp.exe
pause