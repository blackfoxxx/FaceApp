"""
FaceApp GPU Executable Builder
Creates a standalone executable for FaceApp with GPU/CUDA support using PyInstaller
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def build_faceapp_gpu_exe():
    """Build FaceApp as a standalone executable with GPU support"""
    
    print("=" * 60)
    print("Building FaceApp GPU-Enabled Standalone Executable")
    print("=" * 60)
    
    # Get current directory
    app_dir = Path(__file__).parent.absolute()
    os.chdir(app_dir)
    
    # Clean previous builds
    print("[INFO] Cleaning previous builds...")
    for folder in ['build', 'dist']:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            print(f"[INFO] Removed {folder} directory")
    
    # Check for CUDA installation
    check_cuda_installation()
    
    # PyInstaller command with GPU support
    pyinstaller_path = app_dir / "venv" / "Scripts" / "pyinstaller.exe"
    pyinstaller_cmd = [
        str(pyinstaller_path),
        "--onedir",                     # Directory bundle (needed for CUDA DLLs)
        "--console",                    # Console window for GPU debug info
        "--name=FaceApp",              # Executable name
        "--icon=static/images/placeholder.png",  # App icon (if available)
        "--add-data=templates;templates",        # Include templates
        "--add-data=static;static",             # Include static files
        "--add-data=face_training_data.db;.",   # Include database
        
        # Core dependencies
        "--hidden-import=waitress",             # Include waitress server
        "--hidden-import=insightface",          # Include insightface
        "--hidden-import=faiss",               # Include FAISS (GPU version)
        "--hidden-import=imagehash",           # Include imagehash
        "--hidden-import=PIL",                 # Include Pillow
        "--hidden-import=cv2",                 # Include OpenCV
        "--hidden-import=numpy",               # Include NumPy
        
        # GPU-specific dependencies
        "--hidden-import=torch",               # Include PyTorch with CUDA
        "--hidden-import=torchvision",         # Include torchvision with CUDA
        "--hidden-import=onnxruntime",         # Include ONNX Runtime GPU
        "--hidden-import=onnxruntime.capi.onnxruntime_pybind11_state",  # ONNX Runtime internals
        "--hidden-import=faiss.swigfaiss",     # FAISS SWIG bindings
        
        # Collect all files for GPU libraries
        "--collect-all=insightface",           # Collect all insightface files
        "--collect-all=onnxruntime",           # Collect all ONNX Runtime files
        "--collect-all=torch",                 # Collect all PyTorch files (includes CUDA)
        "--collect-all=torchvision",           # Collect all torchvision files
        "--collect-all=faiss",                 # Collect all FAISS files
        
        # CUDA-specific libraries (if available)
        "--collect-binaries=torch",            # Include PyTorch CUDA binaries
        "--collect-binaries=onnxruntime",      # Include ONNX Runtime CUDA binaries
        "--collect-binaries=faiss",            # Include FAISS CUDA binaries
        
        # Exclude unnecessary modules
        "--exclude-module=tkinter",            # Exclude tkinter (not needed)
        "--exclude-module=matplotlib",         # Exclude matplotlib (not needed)
        "--exclude-module=jupyter",            # Exclude jupyter (not needed)
        "--exclude-module=IPython",            # Exclude IPython (not needed)
        
        "app.py"                              # Main application file
    ]
    
    print("[INFO] Starting PyInstaller GPU build...")
    print(f"[INFO] Command: {' '.join(pyinstaller_cmd)}")
    
    try:
        # Run PyInstaller
        result = subprocess.run(pyinstaller_cmd, check=True, capture_output=True, text=True)
        print("[SUCCESS] PyInstaller GPU build completed!")
        
        # Check if executable was created
        exe_path = app_dir / "dist" / "FaceApp" / "FaceApp.exe"
        if exe_path.exists():
            print(f"[SUCCESS] GPU-enabled executable created: {exe_path}")
            
            # Get directory size
            dist_dir = app_dir / "dist" / "FaceApp"
            total_size = sum(f.stat().st_size for f in dist_dir.rglob('*') if f.is_file())
            print(f"[INFO] Total package size: {total_size / (1024*1024):.1f} MB")
            
            # Create GPU-specific launcher script
            create_gpu_launcher_script(app_dir)
            
            # Copy CUDA DLLs if available
            copy_cuda_dlls(dist_dir)
            
            print("\n" + "=" * 60)
            print("GPU BUILD COMPLETE!")
            print("=" * 60)
            print(f"Executable location: {exe_path}")
            print("GPU Features:")
            print("- CUDA support for faster face processing")
            print("- GPU-accelerated FAISS indexing")
            print("- Automatic GPU/CPU fallback")
            print("- Console output for GPU debugging")
            print("\nThe entire 'dist/FaceApp' folder can be distributed to other Windows machines.")
            print("CUDA Toolkit 11.8+ required on target machines for GPU acceleration.")
            
        else:
            print("[ERROR] GPU executable not found in dist folder")
            
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] PyInstaller GPU build failed: {e}")
        print(f"[ERROR] Output: {e.stdout}")
        print(f"[ERROR] Error: {e.stderr}")
        return False
    
    return True

def check_cuda_installation():
    """Check if CUDA is properly installed"""
    print("[INFO] Checking CUDA installation...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"[SUCCESS] CUDA {torch.version.cuda} detected")
            print(f"[INFO] GPU devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"[INFO] GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("[WARNING] CUDA not available - building CPU fallback version")
    except ImportError:
        print("[WARNING] PyTorch not found - ensure GPU requirements are installed")

def copy_cuda_dlls(dist_dir):
    """Copy additional CUDA DLLs if available"""
    print("[INFO] Checking for additional CUDA DLLs...")
    
    # Common CUDA DLL locations
    cuda_paths = [
        os.environ.get('CUDA_PATH', ''),
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8',
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0',
    ]
    
    cuda_dlls = [
        'cublas64_11.dll', 'cublas64_12.dll',
        'cublasLt64_11.dll', 'cublasLt64_12.dll',
        'cudart64_110.dll', 'cudart64_120.dll',
        'curand64_10.dll',
        'cusparse64_11.dll', 'cusparse64_12.dll',
        'nvrtc64_112_0.dll', 'nvrtc64_120_0.dll'
    ]
    
    copied_dlls = 0
    for cuda_path in cuda_paths:
        if not cuda_path or not os.path.exists(cuda_path):
            continue
            
        bin_path = os.path.join(cuda_path, 'bin')
        if not os.path.exists(bin_path):
            continue
            
        for dll in cuda_dlls:
            dll_path = os.path.join(bin_path, dll)
            if os.path.exists(dll_path):
                dest_path = dist_dir / dll
                if not dest_path.exists():
                    shutil.copy2(dll_path, dest_path)
                    print(f"[INFO] Copied {dll}")
                    copied_dlls += 1
    
    if copied_dlls > 0:
        print(f"[SUCCESS] Copied {copied_dlls} additional CUDA DLLs")
    else:
        print("[INFO] No additional CUDA DLLs found to copy")

def create_gpu_launcher_script(app_dir):
    """Create a GPU-specific launcher script for the executable"""
    launcher_content = '''@echo off
echo =======================================
echo Starting FaceApp with GPU Support
echo =======================================
echo.
echo GPU Features:
echo - CUDA acceleration for face processing
echo - GPU-accelerated FAISS indexing  
echo - Automatic GPU/CPU fallback
echo.
echo FaceApp will be available at: http://localhost:8080
echo Press Ctrl+C to stop the application
echo.
echo GPU Debug Information:
echo =======================================

REM Set environment variable for GPU mode
set PROCESSING_MODE=auto

REM Start the application
dist\\FaceApp\\FaceApp.exe

echo.
echo =======================================
echo FaceApp GPU Session Ended
echo =======================================
pause
'''
    
    launcher_path = app_dir / "run_faceapp_gpu.bat"
    with open(launcher_path, 'w') as f:
        f.write(launcher_content)
    
    print(f"[INFO] Created GPU launcher script: {launcher_path}")

if __name__ == "__main__":
    success = build_faceapp_gpu_exe()
    if success:
        print("\n[INFO] GPU build completed successfully!")
        print("[INFO] Use 'run_faceapp_gpu.bat' to launch with GPU support")
    else:
        print("\n[ERROR] GPU build failed!")
        sys.exit(1)