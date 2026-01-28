# FaceApp GPU Deployment Guide

## GPU Requirements

### Hardware Requirements
- **NVIDIA GPU**: GTX 1060 / RTX 2060 or better
- **VRAM**: Minimum 4GB, recommended 6GB+
- **CUDA Compute Capability**: 3.5 or higher
- **System RAM**: 8GB+ recommended

### Software Requirements
- **NVIDIA Drivers**: 450.80.02+ (Linux) or 452.39+ (Windows)
- **CUDA Toolkit**: 11.8 (required for PyTorch compatibility)
- **Python**: 3.9+ (3.10 recommended)

## Pre-Installation Steps

### 1. Install NVIDIA Drivers
Download the latest drivers from [NVIDIA's website](https://www.nvidia.com/drivers/)

### 2. Install CUDA Toolkit 11.8
Download from [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-11-8-0-download-archive)

### 3. Verify CUDA Installation
```bash
nvidia-smi
nvcc --version
```

## GPU Deployment Steps

### Step 1: Run GPU Setup
```bash
setup_fixed.bat
```

This will:
- Check for NVIDIA GPU and CUDA support
- Install GPU-accelerated versions of all libraries
- Test GPU functionality
- Create GPU-specific startup scripts

### Step 2: Verify GPU Installation
```bash
 
```

Expected output:
```
✓ PyTorch GPU available: True
✓ CUDA devices: 1 (or more)
✓ All core dependencies working
✓ Ready to run FaceApp with GPU acceleration
```

### Step 3: Start the Application
```bash
start_production_fixed.bat
```

## GPU-Specific Libraries

### PyTorch GPU (2.0.1+cu118)
- **Purpose**: Neural network operations, tensor computations
- **GPU Benefit**: 5-10x faster than CPU for deep learning operations
- **Memory Usage**: ~1-2GB VRAM for face detection models

### ONNX Runtime GPU (1.15.1)
- **Purpose**: Optimized inference for ONNX models
- **GPU Benefit**: 3-5x faster inference
- **Memory Usage**: ~500MB-1GB VRAM

### FAISS GPU (1.7.4)
- **Purpose**: Fast similarity search and clustering
- **GPU Benefit**: 10-50x faster for large datasets
- **Memory Usage**: Scales with number of face embeddings

### InsightFace (0.7.3)
- **Purpose**: Face detection and recognition
- **GPU Benefit**: Automatic GPU acceleration when available
- **Memory Usage**: ~1-2GB VRAM for face models

## Performance Optimizations

### Environment Variables
Add these to your system or in the startup scripts:

```bash
# Optimize CUDA memory allocation
set PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Enable cuDNN benchmarking for consistent input sizes
set TORCH_CUDNN_BENCHMARK=1

# Reduce CUDA memory fragmentation
set CUDA_LAUNCH_BLOCKING=0
```

### Application Configuration
In your `app.py`, you can add GPU-specific optimizations:

```python
import torch
import os

# Set GPU device if available
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Optimize memory usage
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')
    print("GPU not available, using CPU")
```

## Performance Expectations

### Face Detection Speed
- **CPU**: ~200-500ms per image
- **GPU**: ~50-100ms per image (3-5x faster)

### Face Recognition/Matching
- **CPU**: ~100-200ms per comparison
- **GPU**: ~20-50ms per comparison (4-5x faster)

### Batch Processing
- **CPU**: Linear scaling with batch size
- **GPU**: Significant speedup with larger batches (up to VRAM limits)

## Troubleshooting

### Common Issues

**"CUDA out of memory"**
- Reduce batch sizes in face processing
- Add `torch.cuda.empty_cache()` calls
- Lower image resolution for processing

**"No CUDA-capable device"**
- Verify NVIDIA drivers are installed
- Check GPU is not disabled in Device Manager
- Ensure GPU supports CUDA (GTX/RTX series)

**"CUDA version mismatch"**
- Uninstall existing PyTorch: `pip uninstall torch torchvision`
- Reinstall with correct CUDA version
- Verify CUDA Toolkit version matches PyTorch requirements

**Slow performance despite GPU**
- Check GPU utilization with `nvidia-smi`
- Verify models are actually running on GPU
- Ensure sufficient VRAM is available

### Performance Monitoring

Monitor GPU usage during operation:
```bash
# Real-time GPU monitoring
nvidia-smi -l 1

# Check GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

## Memory Management

### VRAM Usage Guidelines
- **4GB VRAM**: Basic face detection and recognition
- **6GB VRAM**: Comfortable for most operations
- **8GB+ VRAM**: Batch processing and multiple models

### Memory Optimization Tips
1. Process images in smaller batches
2. Use mixed precision (FP16) when possible
3. Clear GPU cache between operations
4. Resize images to optimal dimensions (512x512 or 640x640)

## Production Considerations

### Scaling
- Single GPU: 50-100 concurrent face operations
- Multiple GPUs: Use CUDA_VISIBLE_DEVICES to distribute load
- Load balancing: Consider multiple instances on different GPUs

### Monitoring
- Monitor GPU temperature and utilization
- Set up alerts for VRAM usage
- Log GPU performance metrics

### Backup Strategy
- Keep CPU fallback option available
- Monitor GPU health and driver stability
- Have rollback plan to CPU-only deployment

## Known Issues & Solutions

### Version Compatibility Issues
- **Flask-Werkzeug Compatibility**: Updated to Flask 2.3.3 + Werkzeug 2.3.7 for compatibility
- **NumPy 2.x Issues**: Using NumPy 1.x (`numpy<2.0`) for compatibility with ONNX Runtime and ML libraries
- **FAISS Installation**: GPU version requires proper CUDA installation

## Next Steps

### 🚀 Starting Your FaceApp

#### **Production Mode (Recommended)**
For production deployment with optimized performance:
```bash
start_production_fixed.bat
```
- **URL**: http://localhost:8080
- **Features**: Production-optimized, Waitress WSGI server
- **Performance**: Full GPU acceleration enabled

#### **Development Mode**
For development and testing:
```bash
start_development.bat
```
- **URL**: http://localhost:5000
- **Features**: Debug mode, auto-reload on code changes
- **Performance**: Full GPU acceleration enabled

### 🧪 Testing GPU Setup

Run the comprehensive GPU test:
```bash
test_gpu_dependencies.bat
```

This will verify:
- ✅ All dependencies are installed correctly
- ✅ GPU acceleration is working
- ✅ CUDA is available and functional
- ✅ All ML libraries are compatible

### 📊 Performance Expectations

With GPU acceleration enabled, you should see:

| Operation | CPU Performance | GPU Performance | Speedup |
|-----------|----------------|-----------------|---------|
| Face Detection | ~200ms | ~50ms | **4x faster** |
| Face Recognition | ~500ms | ~100ms | **5x faster** |
| Similarity Search | ~2000ms | ~40ms | **50x faster** |

### 🔧 Maintenance

#### **Updating Dependencies**
To update GPU dependencies:
```bash
setup_fixed.bat
```

#### **Checking GPU Status**
Monitor GPU usage during operation:
```bash
nvidia-smi
```

### 🎯 Ready to Use!

Your FaceApp is now fully configured with GPU acceleration. The application provides:

- **High-performance face detection and recognition**
- **Real-time similarity search**
- **Production-ready deployment**
- **Comprehensive error handling**
- **GPU-optimized processing**

Start with `start_production_fixed.bat` and enjoy blazing-fast face recognition! 🚀

1. Run `setup_fixed.bat` to install GPU libraries
2. Test with `test_gpu_dependencies.bat`
3. Start application with `start_production_fixed.bat`
4. Monitor performance with `nvidia-smi`
5. Optimize based on your specific use case

Your FaceApp will now leverage GPU acceleration for significantly improved performance!