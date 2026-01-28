@echo off
:menu
cls
echo =======================================
echo FaceApp Windows Service Manager
echo =======================================
echo.
echo 1. Install Service
echo 2. Start Service
echo 3. Stop Service
echo 4. Restart Service
echo 5. Check Service Status
echo 6. Uninstall Service
echo 7. Open Services Manager
echo 8. Exit
echo.
set /p choice="Enter your choice (1-8): "

if "%choice%"=="1" goto install
if "%choice%"=="2" goto start
if "%choice%"=="3" goto stop
if "%choice%"=="4" goto restart
if "%choice%"=="5" goto status
if "%choice%"=="6" goto uninstall
if "%choice%"=="7" goto services
if "%choice%"=="8" goto exit
goto menu

:install
echo.
echo Installing FaceApp Service...
call install_service.bat
pause
goto menu

:start
echo.
echo Starting FaceApp Service...
call venv\Scripts\activate.bat
python faceapp_service.py start
echo Service started successfully!
echo FaceApp should be accessible at: http://localhost:8080
pause
goto menu

:stop
echo.
echo Stopping FaceApp Service...
call venv\Scripts\activate.bat
python faceapp_service.py stop
echo Service stopped successfully!
pause
goto menu

:restart
echo.
echo Restarting FaceApp Service...
call venv\Scripts\activate.bat
python faceapp_service.py stop
timeout /t 3 /nobreak >nul
python faceapp_service.py start
echo Service restarted successfully!
echo FaceApp should be accessible at: http://localhost:8080
pause
goto menu

:status
echo.
echo Checking FaceApp Service Status...
sc query FaceAppService
echo.
echo Checking if FaceApp is accessible...
curl -s -o nul -w "HTTP Status: %%{http_code}\n" http://localhost:8080 2>nul || echo Could not connect to FaceApp
pause
goto menu

:uninstall
echo.
echo Uninstalling FaceApp Service...
call uninstall_service.bat
pause
goto menu

:services
echo.
echo Opening Windows Services Manager...
services.msc
goto menu

:exit
echo.
echo Goodbye!
exit /b 0