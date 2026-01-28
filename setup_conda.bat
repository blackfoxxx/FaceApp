@echo off
echo Installing Miniconda if not already installed...
powershell -Command "if (!(Test-Path $env:USERPROFILE\miniconda3\Scripts\conda.exe)) { Invoke-WebRequest -Uri 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x64.exe' -OutFile 'miniconda.exe'; Start-Process -Wait -FilePath '.\miniconda.exe' -ArgumentList '/S /D=%USERPROFILE%\miniconda3' }"

echo Creating conda environment...
call %USERPROFILE%\miniconda3\Scripts\activate.bat
call conda env create -f environment.yml

echo Activating environment...
call conda activate faceapp

echo Installing production WSGI server...
pip install waitress

echo Creating production startup script...
(
echo @echo off
echo call %%USERPROFILE%%\miniconda3\Scripts\activate.bat
echo call conda activate faceapp
echo echo Starting Face Detection App in production mode...
echo python production.py
echo pause
) > start_production.bat

echo Setup complete! To start the application:
echo 1. Run start_production.bat
echo 2. Access the application at http://localhost:8080
echo.
pause