"""
Simple FaceApp Executable Builder
Creates a basic standalone executable for FaceApp using PyInstaller
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def build_simple_faceapp_exe():
    """Build FaceApp as a simple standalone executable"""
    
    print("=" * 50)
    print("Building Simple FaceApp Executable")
    print("=" * 50)
    
    # Get current directory
    app_dir = Path(__file__).parent.absolute()
    os.chdir(app_dir)
    
    # Clean previous builds
    print("[INFO] Cleaning previous builds...")
    for folder in ['build', 'dist']:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"[INFO] Removed {folder} directory")
    
    # Simple PyInstaller command - console mode for easier debugging
    pyinstaller_path = app_dir / "venv" / "Scripts" / "pyinstaller.exe"
    pyinstaller_cmd = [
        str(pyinstaller_path),
        "--onedir",                         # Directory mode (easier to debug)
        "--console",                        # Show console (for debugging)
        "--name=FaceApp",                   # Executable name
        "--add-data=templates;templates",   # Include templates
        "--add-data=static;static",         # Include static files
        "--add-data=face_training_data.db;.",  # Include database
        "--hidden-import=waitress",         # Include waitress server
        "--hidden-import=imagehash",        # Include imagehash
        "--hidden-import=PIL",              # Include Pillow
        "--hidden-import=cv2",              # Include OpenCV
        "--hidden-import=numpy",            # Include NumPy
        "app.py"                           # Main application file
    ]
    
    print("[INFO] Starting simple PyInstaller build...")
    print(f"[INFO] Command: {' '.join(pyinstaller_cmd)}")
    
    try:
        # Run PyInstaller
        result = subprocess.run(pyinstaller_cmd, check=True, capture_output=True, text=True)
        print("[SUCCESS] PyInstaller build completed!")
        
        # Check if executable was created
        exe_path = app_dir / "dist" / "FaceApp" / "FaceApp.exe"
        if exe_path.exists():
            print(f"[SUCCESS] Executable created: {exe_path}")
            print(f"[INFO] File size: {exe_path.stat().st_size / (1024*1024):.1f} MB")
            
            # Create a launcher script
            create_simple_launcher(app_dir)
            
            print("\n" + "=" * 50)
            print("BUILD COMPLETE!")
            print("=" * 50)
            print(f"Executable location: {exe_path}")
            print("Run 'Start_FaceApp_Simple.bat' to launch the application!")
            
        else:
            print("[ERROR] Executable not found in dist folder")
            
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] PyInstaller build failed: {e}")
        print(f"[ERROR] Output: {e.stdout}")
        print(f"[ERROR] Error: {e.stderr}")
        return False
    
    return True

def create_simple_launcher(app_dir):
    """Create a simple launcher script for the executable"""
    launcher_content = '''@echo off
echo =======================================
echo Starting FaceApp Simple Executable
echo =======================================
echo.
echo FaceApp will be available at: http://localhost:8080
echo Press Ctrl+C to stop the application
echo.
cd dist\\FaceApp
echo Starting FaceApp.exe...
start http://localhost:8080
FaceApp.exe
pause
'''
    
    launcher_path = app_dir / "Start_FaceApp_Simple.bat"
    with open(launcher_path, 'w') as f:
        f.write(launcher_content)
    
    print(f"[INFO] Created launcher script: {launcher_path}")

if __name__ == "__main__":
    success = build_simple_faceapp_exe()
    if success:
        print("\n[INFO] Simple build completed successfully!")
    else:
        print("\n[ERROR] Build failed!")
        sys.exit(1)