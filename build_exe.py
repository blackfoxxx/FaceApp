"""
FaceApp Executable Builder
Creates a standalone executable for FaceApp using PyInstaller
"""

import os
import sys
import subprocess
import shutil
import time
from pathlib import Path

def build_faceapp_exe():
    """Build FaceApp as a standalone executable"""
    
    print("=" * 50)
    print("Building FaceApp Standalone Executable")
    print("=" * 50)
    
    # Get current directory
    app_dir = Path(__file__).parent.absolute()
    os.chdir(app_dir)
    
    # Clean previous builds
    print("[INFO] Cleaning previous builds...")
    for folder in ['build']:  # Only clean build folder, leave dist alone if locked
        if os.path.exists(folder):
            try:
                shutil.rmtree(folder)
                print(f"[INFO] Removed {folder} directory")
            except PermissionError:
                print(f"[WARNING] Could not remove {folder} directory - it may be in use")
    
    # Handle dist folder separately - if it exists and is locked, work around it
    if os.path.exists('dist'):
        try:
            # Try to remove a test file to see if we can write to dist
            test_file = os.path.join('dist', 'test_write.tmp')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            print("[INFO] dist directory is accessible")
        except:
            print("[WARNING] dist directory is locked, will overwrite files during build")
    
    # PyInstaller command - use the venv version
    pyinstaller_path = app_dir / "venv" / "Scripts" / "pyinstaller.exe"
    pyinstaller_cmd = [
        str(pyinstaller_path),
        "--onefile",                    # Single executable file
        "--windowed",                   # No console window (GUI app)
        "--name=FaceApp",              # Executable name
        "--icon=static/images/placeholder.png",  # App icon (if available)
        "--add-data=templates;templates",        # Include templates
        "--add-data=static;static",             # Include static files
        "--add-data=face_training_data.db;.",   # Include database
        "--hidden-import=waitress",             # Include waitress server
        "--hidden-import=insightface",          # Include insightface
        "--hidden-import=faiss",               # Include FAISS
        "--hidden-import=imagehash",           # Include imagehash
        "--hidden-import=PIL",                 # Include Pillow
        "--hidden-import=cv2",                 # Include OpenCV
        "--hidden-import=numpy",               # Include NumPy
        "--hidden-import=torch",               # Include PyTorch
        "--hidden-import=torchvision",         # Include torchvision
        "--hidden-import=onnxruntime",         # Include ONNX Runtime
        "--collect-all=insightface",           # Collect all insightface files
        "--collect-all=onnxruntime",           # Collect all ONNX Runtime files
        "--exclude-module=tkinter",            # Exclude tkinter (not needed)
        "--exclude-module=matplotlib",         # Exclude matplotlib (not needed)
        "app.py"                              # Main application file
    ]
    
    print("[INFO] Starting PyInstaller build...")
    print(f"[INFO] Command: {' '.join(pyinstaller_cmd)}")
    
    try:
        # Run PyInstaller
        result = subprocess.run(pyinstaller_cmd, check=True, capture_output=True, text=True)
        print("[SUCCESS] PyInstaller build completed!")
        
        # Check if executable was created
        exe_path = app_dir / "dist" / "FaceApp.exe"
        if exe_path.exists():
            print(f"[SUCCESS] Executable created: {exe_path}")
            print(f"[INFO] File size: {exe_path.stat().st_size / (1024*1024):.1f} MB")
            
            # Create a launcher script
            create_launcher_script(app_dir)
            
            print("\n" + "=" * 50)
            print("BUILD COMPLETE!")
            print("=" * 50)
            print(f"Executable location: {exe_path}")
            print("You can now run FaceApp.exe directly!")
            print("The executable includes all dependencies and can run on other Windows machines.")
            
        else:
            print("[ERROR] Executable not found in dist folder")
            
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] PyInstaller build failed: {e}")
        print(f"[ERROR] Output: {e.stdout}")
        print(f"[ERROR] Error: {e.stderr}")
        return False
    
    return True

def create_launcher_script(app_dir):
    """Create a simple launcher script for the executable"""
    launcher_content = '''@echo off
echo Starting FaceApp...
echo.
echo FaceApp will be available at: http://localhost:8080
echo Press Ctrl+C to stop the application
echo.
start http://localhost:8080
FaceApp.exe
pause
'''
    
    launcher_path = app_dir / "dist" / "Start_FaceApp.bat"
    with open(launcher_path, 'w') as f:
        f.write(launcher_content)
    
    print(f"[INFO] Created launcher script: {launcher_path}")

if __name__ == "__main__":
    success = build_faceapp_exe()
    if success:
        print("\n[INFO] Build completed successfully!")
    else:
        print("\n[ERROR] Build failed!")
        sys.exit(1)