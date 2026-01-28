@echo off
echo =======================================
  GPU Dependencies Test
=======================================

echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

echo [INFO] Testing GPU dependencies...
echo.

python -c "
import sys
print('Python version:', sys.version)
print()

# Test Flask
try:
    import flask
    print('✅ Flask:', flask.__version__)
except ImportError as e:
    print('❌ Flask import failed:', e)

# Test PyTorch
try:
    import torch
    print('✅ PyTorch:', torch.__version__)
    print('   CUDA available:', torch.cuda.is_available())
    if torch.cuda.is_available():
        print('   CUDA devices:', torch.cuda.device_count())
        print('   GPU name:', torch.cuda.get_device_name(0))
except ImportError as e:
    print('❌ PyTorch import failed:', e)

# Test ONNX Runtime
try:
    import onnxruntime
    print('✅ ONNX Runtime available')
    providers = onnxruntime.get_available_providers()
    if 'CUDAExecutionProvider' in providers:
        print('   GPU acceleration: Available')
    else:
        print('   GPU acceleration: Not available')
except ImportError as e:
    print('❌ ONNX Runtime import failed:', e)

# Test InsightFace
try:
    import insightface
    print('✅ InsightFace:', insightface.__version__)
except ImportError as e:
    print('❌ InsightFace import failed:', e)

# Test NumPy
try:
    import numpy as np
    print('✅ NumPy:', np.__version__)
except ImportError as e:
    print('❌ NumPy import failed:', e)

# Test OpenCV
try:
    import cv2
    print('✅ OpenCV:', cv2.__version__)
except ImportError as e:
    print('❌ OpenCV import failed:', e)

print()
print('GPU Dependencies Test Complete!')
"

echo.
echo [INFO] Test completed. Check results above.
pause