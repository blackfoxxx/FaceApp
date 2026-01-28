# FaceApp — Face Matching Service

A Flask-based face matching and training app using InsightFace and FAISS with SQLite-backed persistence. Includes a simple web UI and a JSON API with **GPU acceleration support**.

## 🚀 GPU Acceleration

FaceApp now supports **NVIDIA GPU acceleration** for significantly faster performance:
- **3-7x faster** face processing
- **Automatic GPU detection** and fallback
- **CUDA 11.8+ support** with PyTorch and ONNX Runtime

### Quick GPU Setup
```bash
# Install CUDA Toolkit 11.8+ and NVIDIA drivers
# Then run:
.\build_exe_gpu.bat    # Build GPU-enabled executable
.\test_gpu.bat         # Verify GPU setup
```

📖 **See [GPU_SETUP_GUIDE.md](GPU_SETUP_GUIDE.md) for detailed GPU installation instructions.**

## Quick Start

- Python 3.9+
- Windows/macOS/Linux
- Optional: NVIDIA GPU with CUDA 11.8+ for acceleration

Install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Run the app:

```bash
FLASK_APP=app.py flask run --host=0.0.0.0 --port=5000
```

Open the UI at `http://localhost:5000`.

## Configuration

Environment variables:

- `APP_VERSION` — app version string (default `0.1.0`)
- `DEFAULT_MODEL` — starting model (`buffalo_l`, `buffalo_m`, `buffalo_sc`)
- `PROCESSING_MODE` — `auto` (detect GPU), `gpu`, or `cpu` (default `auto`)
- `UPLOAD_FOLDER` — where images are stored (default `static/images`)
- `MAX_CONTENT_LENGTH` — upload size in bytes (default 10MB)
- `CORS_ALLOWED_ORIGINS` — comma-separated list of allowed origins (optional)

The server ensures `UPLOAD_FOLDER` exists on startup.

## Key Endpoints

- `GET /` — web UI
- `GET /health` — health check, DB connectivity, loaded models, counts
- `GET /version` — returns app version
- `POST /add_face` — add a face (form-data: `file`, `name`, `model`, optional `align`, `process_all_faces`)
- `POST /match_face` — match face from an uploaded image (see UI for params)
- `POST /train_model` — batch add faces from files/folder
- `POST /check_duplicates` — identify duplicate images, optionally delete
- `DELETE /delete_face/<id>` — delete a face and its image
- `GET /get_face_details/<id>` — face details by id
- `POST /switch_model` — switch active model
- `POST /switch_processing_mode` — switch between CPU/GPU processing modes

## 🏗️ Building Executables

### Standard Build (CPU-only)
```bash
.\build_exe.bat
```

### GPU-Enabled Build
```bash
.\build_exe_gpu.bat
```

The GPU build includes CUDA libraries and creates a `run_faceapp_gpu.bat` launcher for easy deployment.

## 🧪 Testing & Verification

Test your GPU setup:
```bash
.\test_gpu.bat
```

This will verify CUDA availability, GPU libraries, and FaceApp's GPU detection functionality.

## 📁 Project Structure

```
FaceApp/
├── app.py                 # Main Flask application
├── requirements.txt       # CPU dependencies
├── requirements_gpu.txt   # GPU dependencies (reference)
├── build_exe.py          # Standard PyInstaller build
├── build_exe_gpu.py      # GPU-enabled PyInstaller build
├── test_gpu_setup.py     # GPU verification script
├── GPU_SETUP_GUIDE.md    # Detailed GPU setup guide
├── static/               # Web UI assets
├── templates/            # Flask templates
└── venv/                # Virtual environment
```

## 🔧 Troubleshooting

- **GPU not detected**: Check NVIDIA drivers and CUDA installation
- **Build failures**: Ensure all dependencies are installed in virtual environment
- **Performance issues**: Monitor GPU usage and memory with `nvidia-smi`
- **CUDA errors**: Verify CUDA version compatibility (11.8+ recommended)

For detailed troubleshooting, see [GPU_SETUP_GUIDE.md](GPU_SETUP_GUIDE.md).

## Enhancements Added

- Health and version endpoints
- Upload folder creation on startup
- Upload size limit via `MAX_CONTENT_LENGTH`
- File type validation in `POST /add_face`
- Safer image deletion (normalize URL paths to filesystem paths)
- Configurable CORS origins via `CORS_ALLOWED_ORIGINS`

## Notes

- FAISS/InsightFace are CPU by default; set `PROCESSING_MODE=gpu` if your environment is configured for GPU and InsightFace can use it.
- Images are stored under `static/images` by default, and entries are persisted in `face_training_data.db`.

## License

Internal/private project by default. Add a license if you plan to distribute.
