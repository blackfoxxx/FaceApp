@echo off
echo Installing PyInstaller in virtual environment...
call venv\Scripts\activate.bat
pip install pyinstaller
echo PyInstaller installation complete!
pause