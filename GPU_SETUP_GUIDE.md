# FaceApp GPU Setup Guide

This guide will help you configure FaceApp to utilize GPU acceleration with CUDA for significantly improved performance.

## 🚀 Performance Benefits

**GPU vs CPU Performance:**
- **Face Detection**: 5-10x faster
- **Face Recognition**: 3-8x faster  
- **FAISS Indexing**: 2-5x faster
- **Overall Processing**: 3-7x improvement

## 📋 System Requirements

### Minimum Requirements
- **OS**: Windows 10/11 (64-bit)
- **GPU**: NVIDIA GPU with CUDA Compute Capability 3.5+
- **VRAM**: 4GB+ recommended
- **RAM**: 8GB+ system memory
- **Storage**: 5GB+ free space (including CUDA toolkit)

### Recommended GPUs
- RTX 30/40 series (RTX 3060, 3070, 4070, etc.)
- GTX 16 series (GTX 1660, 1660 Ti)
- RTX 20 series (RTX 2060, 2070, 2080)
- Quadro/Tesla professional cards

## 🔧 Installation Steps

### Step 1: Install NVIDIA Drivers
1. Download latest drivers from [NVIDIA Driver Downloads](https://www.nvidia.com/drivers/)
2. Install and restart your computer
3. Verify installation: `nvidia-smi` in command prompt

### Step 2: Install CUDA Toolkit 11.8
1. Download [CUDA Toolkit 11.8](https://developer.nvidia.com/cuda-11-8-0-download-archive)
2. Run installer with default settings
3. Add to PATH (usually automatic):
   ```
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin
   C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\libnvvp
   ```

### Step 3: Install cuDNN 8.6+ (Optional but Recommended)
1. Download [cuDNN 8.6+](https://developer.nvidia.com/cudnn) (requires NVIDIA account)
2. Extract to CUDA installation directory
3. Copy files to respective CUDA folders

### Step 4: Install FaceApp GPU Dependencies
```bash
# Activate virtual environment
venv\Scripts\activate.bat

# Install GPU-optimized packages
pip install -r requirements.txt --upgrade

# Verify installation
python test_gpu_setup.py
```

## 🏗️ Building GPU-Enabled Executable

### Option 1: Quick Build (Recommended)
```bash
# Run the GPU build script
.\build_exe_gpu.bat
```

### Option 2: Manual Build
```bash
# Activate environment
venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt --upgrade

# Build GPU executable
python build_exe_gpu.py
```

## 🧪 Testing GPU Setup

### Automated Testing
```bash
# Run comprehensive GPU tests
.\test_gpu.bat
```

### Manual Verification
```bash
# Activate environment
venv\Scripts\activate.bat

# Run test script
python test_gpu_setup.py
```

### Expected Test Results
✅ **Fully GPU-Ready:**
```
Overall GPU Status: ✓ READY
✓ PyTorch + CUDA
✓ ONNX Runtime GPU  
✓ FAISS GPU
✓ InsightFace
✓ FaceApp GPU Detection
```

⚠️ **Partial/CPU Mode:**
```
Overall GPU Status: ⚠ PARTIAL/CPU ONLY
✗ PyTorch + CUDA (Missing CUDA)
✗ ONNX Runtime GPU (CPU fallback)
```

## 🚀 Running GPU-Enabled FaceApp

### From Executable
```bash
# Launch GPU-enabled executable
.\run_faceapp_gpu.bat

# Or run directly
dist\FaceApp\FaceApp.exe
```

### From Source Code
```bash
# Set GPU mode explicitly
set PROCESSING_MODE=gpu
python app.py

# Or use auto-detection (default)
set PROCESSING_MODE=auto
python app.py
```

## ⚙️ Configuration Options

### Environment Variables
- `PROCESSING_MODE=auto` - Auto-detect GPU (default)
- `PROCESSING_MODE=gpu` - Force GPU mode
- `PROCESSING_MODE=cpu` - Force CPU mode

### Runtime Switching
Access the API endpoint to switch modes dynamically:
```
POST /switch_processing_mode
{
  "mode": "gpu"  // or "cpu"
}
```

## 📊 Performance Monitoring

### GPU Usage Monitoring
- **Task Manager**: Performance → GPU
- **NVIDIA-SMI**: `nvidia-smi -l 1` (updates every second)
- **GPU-Z**: Third-party monitoring tool

### FaceApp Debug Output
The GPU executable runs in console mode and shows:
```
DEBUG: torch imported successfully, CUDA available: True
DEBUG: Auto-detected processing mode: gpu (GPU available: True)
DEBUG: Final processing mode: gpu
```

## 🔧 Troubleshooting

### Common Issues

#### 1. "CUDA not available" Error
**Symptoms:**
```
DEBUG: torch imported successfully, CUDA available: False
WARNING: GPU mode requested but CUDA not available, falling back to CPU
```

**Solutions:**
- Install/update NVIDIA drivers
- Install CUDA Toolkit 11.8
- Restart computer after installation
- Verify: `nvidia-smi` shows GPU info

#### 2. "No module named 'torch'" Error
**Solutions:**
```bash
venv\Scripts\activate.bat
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

#### 3. ONNX Runtime GPU Issues
**Symptoms:**
```
⚠ GPU providers found: []
⚠ No GPU providers found - using CPU
```

**Solutions:**
```bash
pip install onnxruntime-gpu==1.15.1
# Ensure CUDA 11.8 is installed
```

#### 4. FAISS GPU Not Working
**Symptoms:**
```
✓ FAISS GPU count: 0
⚠ No FAISS GPU resources found - using CPU
```

**Solutions:**
```bash
pip install faiss-gpu==1.7.4
# Check GPU memory availability
```

#### 5. Out of Memory Errors
**Solutions:**
- Reduce batch size in processing
- Close other GPU applications
- Use smaller model: `buffalo_s` instead of `buffalo_l`
- Add more VRAM or use CPU mode

### Performance Issues

#### Slow GPU Performance
1. **Check GPU utilization**: Should be >80% during processing
2. **Verify CUDA version**: Must match PyTorch requirements
3. **Update drivers**: Use latest NVIDIA drivers
4. **Check thermal throttling**: Monitor GPU temperature

#### Memory Leaks
1. **Restart application** periodically for long-running sessions
2. **Monitor VRAM usage** with `nvidia-smi`
3. **Use CPU mode** for very large datasets

## 📁 Distribution

### GPU Executable Distribution
The built executable includes:
- **Location**: `dist\FaceApp\` folder
- **Size**: ~2-4GB (includes CUDA libraries)
- **Portability**: Entire folder can be copied to other Windows machines

### Target Machine Requirements
- Windows 10/11 (64-bit)
- NVIDIA GPU with updated drivers
- CUDA Toolkit 11.8+ (for full GPU acceleration)
- 4GB+ available VRAM

### Fallback Behavior
- Automatically detects GPU availability
- Falls back to CPU if CUDA unavailable
- No manual configuration needed

## 🔄 Updating GPU Setup

### Update GPU Dependencies
```bash
venv\Scripts\activate.bat
pip install -r requirements.txt --upgrade --force-reinstall
```

### Rebuild GPU Executable
```bash
# Clean previous builds
rmdir /s build dist

# Rebuild with latest dependencies
.\build_exe_gpu.bat
```

## 📞 Support

### Getting Help
1. **Run diagnostics**: `.\test_gpu.bat`
2. **Check logs**: Console output shows detailed GPU status
3. **Verify hardware**: `nvidia-smi` for GPU info
4. **Test components**: Individual library imports in Python

### Reporting Issues
Include the following information:
- GPU model and VRAM
- CUDA Toolkit version
- Driver version (`nvidia-smi`)
- Test script output (`test_gpu_setup.py`)
- Error messages and stack traces

---

## 🎯 Quick Start Checklist

- [ ] Install NVIDIA drivers
- [ ] Install CUDA Toolkit 11.8
- [ ] Run `.\test_gpu.bat` to verify setup
- [ ] Build GPU executable: `.\build_exe_gpu.bat`
- [ ] Launch with: `.\run_faceapp_gpu.bat`
- [ ] Verify GPU usage in Task Manager

**🎉 Enjoy significantly faster face processing with GPU acceleration!**