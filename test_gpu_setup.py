"""
GPU Setup Verification Script for FaceApp
Tests CUDA availability, GPU libraries, and FaceApp GPU functionality
"""

import sys
import os
import traceback
from pathlib import Path

def test_basic_imports():
    """Test basic Python library imports"""
    print("=" * 60)
    print("Testing Basic Library Imports")
    print("=" * 60)
    
    libraries = [
        ('numpy', 'np'),
        ('cv2', 'cv2'),
        ('PIL', 'PIL'),
        ('flask', 'flask'),
        ('sqlite3', 'sqlite3')
    ]
    
    results = {}
    for lib_name, import_name in libraries:
        try:
            __import__(import_name)
            print(f"✓ {lib_name}: OK")
            results[lib_name] = True
        except ImportError as e:
            print(f"✗ {lib_name}: FAILED - {e}")
            results[lib_name] = False
    
    return results

def test_torch_cuda():
    """Test PyTorch and CUDA availability"""
    print("\n" + "=" * 60)
    print("Testing PyTorch and CUDA")
    print("=" * 60)
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        # Test CUDA availability
        cuda_available = torch.cuda.is_available()
        print(f"✓ CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU count: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"✓ GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            
            # Test GPU tensor operations
            try:
                x = torch.randn(100, 100).cuda()
                y = torch.randn(100, 100).cuda()
                z = torch.mm(x, y)
                print("✓ GPU tensor operations: OK")
                return True
            except Exception as e:
                print(f"✗ GPU tensor operations: FAILED - {e}")
                return False
        else:
            print("⚠ CUDA not available - will use CPU mode")
            return False
            
    except ImportError as e:
        print(f"✗ PyTorch import: FAILED - {e}")
        return False
    except Exception as e:
        print(f"✗ PyTorch/CUDA test: FAILED - {e}")
        return False

def test_onnxruntime_gpu():
    """Test ONNX Runtime GPU support"""
    print("\n" + "=" * 60)
    print("Testing ONNX Runtime GPU")
    print("=" * 60)
    
    try:
        import onnxruntime as ort
        print(f"✓ ONNX Runtime version: {ort.__version__}")
        
        # Check available providers
        providers = ort.get_available_providers()
        print(f"✓ Available providers: {providers}")
        
        gpu_providers = [p for p in providers if 'CUDA' in p or 'GPU' in p]
        if gpu_providers:
            print(f"✓ GPU providers found: {gpu_providers}")
            
            # Test GPU session creation
            try:
                session = ort.InferenceSession(
                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
                )
                print("✓ GPU session creation: OK")
                return True
            except Exception as e:
                print(f"⚠ GPU session creation: FAILED - {e}")
                print("⚠ Will fallback to CPU")
                return False
        else:
            print("⚠ No GPU providers found - using CPU")
            return False
            
    except ImportError as e:
        print(f"✗ ONNX Runtime import: FAILED - {e}")
        return False
    except Exception as e:
        print(f"✗ ONNX Runtime test: FAILED - {e}")
        return False

def test_faiss_gpu():
    """Test FAISS GPU support"""
    print("\n" + "=" * 60)
    print("Testing FAISS GPU")
    print("=" * 60)
    
    try:
        import faiss
        print(f"✓ FAISS version: {faiss.__version__}")
        
        # Test GPU resources
        try:
            gpu_count = faiss.get_num_gpus()
            print(f"✓ FAISS GPU count: {gpu_count}")
            
            if gpu_count > 0:
                # Test GPU index creation
                import numpy as np
                
                # Create test data
                d = 128  # dimension
                nb = 1000  # database size
                np.random.seed(1234)
                xb = np.random.random((nb, d)).astype('float32')
                
                # Create GPU index
                res = faiss.StandardGpuResources()
                index_flat = faiss.IndexFlatL2(d)
                gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
                
                # Add vectors and search
                gpu_index.add(xb)
                k = 5
                xq = np.random.random((1, d)).astype('float32')
                distances, indices = gpu_index.search(xq, k)
                
                print("✓ FAISS GPU operations: OK")
                return True
            else:
                print("⚠ No FAISS GPU resources found - using CPU")
                return False
                
        except Exception as e:
            print(f"⚠ FAISS GPU test: FAILED - {e}")
            print("⚠ Will fallback to CPU")
            return False
            
    except ImportError as e:
        print(f"✗ FAISS import: FAILED - {e}")
        return False

def test_insightface():
    """Test InsightFace library"""
    print("\n" + "=" * 60)
    print("Testing InsightFace")
    print("=" * 60)
    
    try:
        from insightface.app import FaceAnalysis
        print("✓ InsightFace import: OK")
        
        # Test model initialization
        try:
            app = FaceAnalysis(name='buffalo_l')
            print("✓ InsightFace model creation: OK")
            
            # Test GPU context
            try:
                app.prepare(ctx_id=0, det_size=(640, 640))  # GPU context
                print("✓ InsightFace GPU context: OK")
                return True
            except Exception as e:
                print(f"⚠ InsightFace GPU context: FAILED - {e}")
                try:
                    app.prepare(ctx_id=-1, det_size=(640, 640))  # CPU context
                    print("✓ InsightFace CPU context: OK")
                    return False
                except Exception as e2:
                    print(f"✗ InsightFace CPU context: FAILED - {e2}")
                    return False
                    
        except Exception as e:
            print(f"✗ InsightFace model creation: FAILED - {e}")
            return False
            
    except ImportError as e:
        print(f"✗ InsightFace import: FAILED - {e}")
        return False

def test_faceapp_gpu_detection():
    """Test FaceApp's GPU detection logic"""
    print("\n" + "=" * 60)
    print("Testing FaceApp GPU Detection")
    print("=" * 60)
    
    try:
        # Add current directory to path to import app
        sys.path.insert(0, str(Path(__file__).parent))
        
        # Import the GPU detection function
        from app import detect_gpu_availability, PROCESSING_MODE, TORCH_AVAILABLE
        
        print(f"✓ FaceApp imports: OK")
        print(f"✓ Torch available: {TORCH_AVAILABLE}")
        
        gpu_detected = detect_gpu_availability()
        print(f"✓ GPU detected by FaceApp: {gpu_detected}")
        print(f"✓ Processing mode: {PROCESSING_MODE}")
        
        return gpu_detected
        
    except Exception as e:
        print(f"✗ FaceApp GPU detection: FAILED - {e}")
        traceback.print_exc()
        return False

def generate_report(results):
    """Generate a comprehensive test report"""
    print("\n" + "=" * 60)
    print("GPU SETUP VERIFICATION REPORT")
    print("=" * 60)
    
    gpu_ready = all([
        results.get('torch_cuda', False),
        results.get('onnxruntime_gpu', False),
        results.get('faiss_gpu', False),
        results.get('insightface', False)
    ])
    
    print(f"Overall GPU Status: {'✓ READY' if gpu_ready else '⚠ PARTIAL/CPU ONLY'}")
    print()
    
    print("Component Status:")
    components = [
        ('Basic Libraries', results.get('basic_imports', False)),
        ('PyTorch + CUDA', results.get('torch_cuda', False)),
        ('ONNX Runtime GPU', results.get('onnxruntime_gpu', False)),
        ('FAISS GPU', results.get('faiss_gpu', False)),
        ('InsightFace', results.get('insightface', False)),
        ('FaceApp GPU Detection', results.get('faceapp_gpu', False))
    ]
    
    for component, status in components:
        status_icon = "✓" if status else "✗"
        print(f"  {status_icon} {component}")
    
    print()
    if gpu_ready:
        print("🚀 GPU acceleration is fully configured!")
        print("   - Face processing will use GPU")
        print("   - FAISS indexing will use GPU")
        print("   - Significant performance improvements expected")
    else:
        print("⚠ GPU acceleration partially configured or unavailable")
        print("   - Application will run in CPU mode")
        print("   - Consider installing CUDA Toolkit 11.8+")
        print("   - Check GPU drivers and dependencies")
    
    print("\nNext Steps:")
    if gpu_ready:
        print("1. Run: python build_exe_gpu.py")
        print("2. Test the GPU-enabled executable")
        print("3. Distribute the 'dist/FaceApp' folder")
    else:
        print("1. Install CUDA Toolkit 11.8+")
        print("2. Update GPU drivers")
        print("3. Reinstall GPU packages: pip install -r requirements.txt")
        print("4. Re-run this test script")

def main():
    """Main test function"""
    print("FaceApp GPU Setup Verification")
    print("Testing GPU capabilities and dependencies...")
    
    results = {}
    
    # Run all tests
    basic_ok = test_basic_imports()
    results['basic_imports'] = all(basic_ok.values())
    
    results['torch_cuda'] = test_torch_cuda()
    results['onnxruntime_gpu'] = test_onnxruntime_gpu()
    results['faiss_gpu'] = test_faiss_gpu()
    results['insightface'] = test_insightface()
    results['faceapp_gpu'] = test_faceapp_gpu_detection()
    
    # Generate final report
    generate_report(results)

if __name__ == "__main__":
    main()