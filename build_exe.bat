@echo off
echo =======================================
echo Building FaceApp Standalone Executable
echo =======================================
echo.

REM Activate virtual environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

REM Run the build script
echo [INFO] Starting build process...
python build_exe.py

echo.
echo =======================================
echo Build Process Complete!
echo =======================================
echo.
echo Check the 'dist' folder for the executable.
echo.
pause