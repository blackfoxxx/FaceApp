@echo off
echo Downloading Visual Studio Build Tools...
curl -L "https://aka.ms/vs/17/release/vs_buildtools.exe" --output vs_buildtools.exe

echo Installing Visual Studio Build Tools...
start /wait vs_buildtools.exe --quiet --wait --norestart --nocache ^
    --installPath "%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools" ^
    --add Microsoft.VisualStudio.Component.VC.Tools.x86.x64 ^
    --add Microsoft.VisualStudio.Component.Windows10SDK.19041

echo Installation completed. Please run setup_py39.bat next.
pause