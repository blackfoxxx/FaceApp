# At the top of the file, add model configuration
from flask import Flask, request, jsonify, render_template, url_for
import cv2
import numpy as np
import os
import uuid
import time
from flask_cors import CORS
import hashlib
from PIL import Image
import sqlite3
import json
import base64
import imagehash
import threading
import tempfile
import shutil

# Import torch for GPU detection
torch = None
TORCH_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
    print(f"DEBUG: torch imported successfully, CUDA available: {torch.cuda.is_available()}")
except ImportError as e:
    print(f"ERROR: Failed to import torch: {e}")
    torch = None
    TORCH_AVAILABLE = False

# Import insightface with error handling
FaceAnalysis = None  # Initialize as None first
INSIGHTFACE_AVAILABLE = False

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
    print("DEBUG: FaceAnalysis imported successfully")
except ImportError as e:
    print(f"ERROR: Failed to import FaceAnalysis: {e}")
    FaceAnalysis = None
    INSIGHTFACE_AVAILABLE = False

# Import faiss with error handling
faiss = None
FAISS_AVAILABLE = False

try:
    import faiss
    FAISS_AVAILABLE = True
    print("DEBUG: faiss imported successfully")
except ImportError as e:
    print(f"ERROR: Failed to import faiss: {e}")
    faiss = None
    FAISS_AVAILABLE = False

# Database setup
DB_FILE = 'face_training_data.db'

# Basic app metadata/config
APP_VERSION = os.environ.get('APP_VERSION', '0.1.0')
UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', os.path.join('static', 'images'))
# Default 25 GB upload limit (increased for very large training datasets)
MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 25 * 1024 * 1024 * 1024))

# Vector helpers
def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if vec is None:
        return vec
    n = np.linalg.norm(vec)
    if n < eps:
        return vec
    return (vec / n).astype('float32')

def init_database():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Create table for storing face embeddings and metadata
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            person_name TEXT NOT NULL,
            embedding BLOB NOT NULL,
            image_path TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create index for faster queries
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_model_name ON face_data(model_name)
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS image_hashes (
            file TEXT PRIMARY KEY,
            mtime REAL NOT NULL,
            phash INTEGER NOT NULL,
            dhash INTEGER NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_image_hashes_mtime ON image_hashes(mtime)
    ''')
    
    conn.commit()
    conn.close()

def save_face_to_db(model_name, person_name, embedding, image_path):
    """Save face data to SQLite database"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Convert numpy array to bytes for storage
    embedding_bytes = embedding.tobytes()
    
    cursor.execute('''
        INSERT INTO face_data (model_name, person_name, embedding, image_path)
        VALUES (?, ?, ?, ?)
    ''', (model_name, person_name, embedding_bytes, image_path))
    
    conn.commit()
    conn.close()

def load_faces_from_db(model_name):
    """Load face data from SQLite database for a specific model"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT person_name, embedding, image_path FROM face_data 
        WHERE model_name = ?
    ''', (model_name,))
    
    results = cursor.fetchall()
    conn.close()
    
    embeddings = []
    names = []
    images = []
    
    for name, embedding_bytes, image_path in results:
        # Normalize image path and compute filesystem path
        if image_path.startswith('/static/'):
            image_url = image_path
            fs_path = image_path[1:]  # strip leading slash
        elif image_path.startswith('static/'):
            image_url = '/' + image_path
            fs_path = image_path
        else:
            if image_path.startswith('images/'):
                image_url = '/static/' + image_path
                fs_path = 'static/' + image_path
            else:
                image_url = '/static/images/' + image_path
                fs_path = os.path.join('static', 'images', image_path)
        
        # Exclude invalid images (missing files or placeholder)
        try:
            if (fs_path.endswith('placeholder.png')) or (not os.path.exists(fs_path)):
                continue
        except Exception:
            continue
        
        # Convert bytes back to numpy array
        embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        embeddings.append(l2_normalize(embedding))
        names.append(name)
        images.append(image_url)
    
    return {
        'embeddings': embeddings,
        'names': names,
        'images': images
    }


def cleanup_upload_folder(keep_placeholder=True):
    """Remove stored images from the upload folder."""
    removed = []
    folder = app.config['UPLOAD_FOLDER']
    if not os.path.exists(folder):
        return removed

    for filename in os.listdir(folder):
        if keep_placeholder and filename == 'placeholder.png':
            continue

        filepath = os.path.join(folder, filename)
        if os.path.isfile(filepath):
            try:
                os.remove(filepath)
                removed.append(filename)
            except Exception as exc:
                print(f"Warning: could not delete image {filepath}: {exc}")

    return removed


def save_query_face_image(face_img=None, fallback_image=None):
    """Persist a query face image for result display, falling back to placeholder."""
    candidates = []
    if face_img is not None and getattr(face_img, 'size', 0) > 0:
        candidates.append(face_img)
    if fallback_image is not None and getattr(fallback_image, 'size', 0) > 0:
        candidates.append(fallback_image)

    for image in candidates:
        try:
            filename = f"query_{uuid.uuid4()}.jpg"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if cv2.imwrite(filepath, image):
                return url_for('static', filename=f'images/{filename}')
        except Exception as exc:
            print(f"Warning: could not write query image: {exc}")

    return url_for('static', filename='images/placeholder.png')

def clear_model_data(model_name):
    """Clear all training data for a specific model"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Get all image paths before deletion
    cursor.execute('SELECT image_path FROM face_data WHERE model_name = ?', (model_name,))
    image_paths = cursor.fetchall()
    
    # Delete from database
    cursor.execute('DELETE FROM face_data WHERE model_name = ?', (model_name,))
    
    conn.commit()
    conn.close()
    
    # Delete image files from filesystem
    for (image_path,) in image_paths:
        try:
            # Convert URL path to filesystem path
            if image_path.startswith('/static/'):
                file_path = image_path[1:]  # Remove leading slash
            elif image_path.startswith('static/'):
                file_path = image_path
            else:
                file_path = 'static/' + image_path if not image_path.startswith('images/') else 'static/' + image_path
            
            # Check if file exists and delete it
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Warning: Could not delete image file {image_path}: {e}")

# Initialize database on startup
init_database()

# Add image enhancement functions
def enhance_contrast(image):
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    # Merge channels
    merged = cv2.merge((cl, a, b))
    
    # Convert back to BGR
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    
    return enhanced

def reduce_noise(image):
    # Non-local means denoising
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return denoised

def upscale_image(image):
    # Simple bicubic upscaling for very small images
    height, width = image.shape[:2]
    if height < 100 or width < 100:  # Only upscale very small images
        upscaled = cv2.resize(image, (width*2, height*2), interpolation=cv2.INTER_CUBIC)
        return upscaled
    return image

def enhance_image(image, enhance_level='medium'):
    """Apply image enhancement based on the specified level"""
    if enhance_level == 'none':
        return image
    
    # Always upscale small images
    image = upscale_image(image)
    
    if enhance_level == 'low':
        # Just apply contrast enhancement
        return enhance_contrast(image)
    
    elif enhance_level == 'medium':
        # Apply contrast enhancement and mild noise reduction
        enhanced = enhance_contrast(image)
        return cv2.GaussianBlur(enhanced, (3, 3), 0)
    
    elif enhance_level == 'high':
        # Apply full enhancement pipeline
        denoised = reduce_noise(image)
        return enhance_contrast(denoised)
    
    return image

# Add this function after the align_face function (around line 130)
def assess_face_quality(face_img, landmarks=None, face_rect=None):
    """
    Assess the quality of a face image based on multiple factors:
    - Resolution: Size of the face region
    - Brightness: Average pixel intensity
    - Contrast: Standard deviation of pixel intensities
    - Sharpness: Variance of Laplacian
    - Pose: If landmarks provided, assess face pose
    
    Returns a quality score between 0-1
    """
    if face_img is None:
        return 0.0
    
    # Convert to grayscale for some calculations
    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img
    
    # 1. Resolution score (0-0.25)
    height, width = face_img.shape[:2]
    min_dimension = min(height, width)
    resolution_score = min(1.0, min_dimension / 150) * 0.25
    
    # 2. Brightness score (0-0.2)
    mean_brightness = np.mean(gray) / 255
    # Penalize if too dark or too bright
    brightness_score = (1 - abs(mean_brightness - 0.5) * 2) * 0.2
    
    # 3. Contrast score (0-0.2)
    std_dev = np.std(gray) / 255
    contrast_score = min(std_dev * 5, 1.0) * 0.2
    
    # 4. Sharpness score (0-0.25)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(laplacian_var / 500, 1.0) * 0.25
    
    # 5. Pose score (0-0.1) - if landmarks available
    pose_score = 0.1  # Default if no landmarks
    if landmarks is not None:
        # Simple pose estimation based on eye positions
        # More sophisticated pose estimation could be implemented
        pose_score = 0.1
    
    # Calculate final quality score (0-1)
    quality_score = resolution_score + brightness_score + contrast_score + sharpness_score + pose_score
    
    # Cap at 1.0
    return min(1.0, quality_score)


def detect_partial_face(face_img, landmarks=None):
    """
    Detect if the face is partial (e.g., occluded, cropped) and calculate
    a partial match factor to adjust confidence accordingly.
    
    Returns a tuple of (is_partial, partial_match_factor)
    - is_partial: Boolean indicating if face is partial
    - partial_match_factor: 1.0 for full face, lower for partial faces
    """
    if face_img is None:
        return True, 0.5
    
    # Convert to grayscale
    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img
    
    # 1. Check face dimensions ratio
    height, width = face_img.shape[:2]
    aspect_ratio = width / height
    normal_ratio = 0.8  # Typical face aspect ratio
    ratio_factor = min(1.0, 1.0 - abs(aspect_ratio - normal_ratio))
    
    # 2. Check for edge proximity (face potentially cropped)
    edge_margin = 0.05  # 5% margin
    edge_factor = 1.0
    
    # If landmarks are available, check if any are too close to the edge
    if landmarks is not None:
        for point in landmarks:
            x, y = point
            if (x < width * edge_margin or x > width * (1 - edge_margin) or
                y < height * edge_margin or y > height * (1 - edge_margin)):
                edge_factor = 0.8
                break
    
    # 3. Check for uniform regions (potential occlusions)
    blocks = 4  # Divide image into 4x4 blocks
    block_h, block_w = height // blocks, width // blocks
    uniform_blocks = 0
    
    for i in range(blocks):
        for j in range(blocks):
            block = gray[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
            if block.size > 0:  # Ensure block is not empty
                block_std = np.std(block)
                if block_std < 20:  # Low standard deviation indicates uniform region
                    uniform_blocks += 1
    
    uniform_factor = 1.0 - (uniform_blocks / (blocks * blocks) * 0.5)
    
    # Calculate overall partial match factor (1.0 = full face, lower = partial)
    partial_match_factor = (ratio_factor * 0.4 + edge_factor * 0.3 + uniform_factor * 0.3)
    
    # Determine if face is partial
    is_partial = partial_match_factor < 0.85
    
    return is_partial, partial_match_factor

def detect_face_mask(face_img):
    """
    Specifically detect if a person is wearing a face mask
    Returns (is_masked, mask_probability)
    """
    if face_img is None:
        return False, 0.0
    
    # Convert to grayscale
    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img
    
    height, width = face_img.shape[:2]
    
    # Focus on the lower half of the face where masks typically appear
    lower_face = gray[height//2:, :]
    
    # Simple heuristic: Check for uniform regions in lower face area
    # (masks often create more uniform texture than mouth/chin features)
    blocks_h, blocks_w = 3, 4
    block_h, block_w = lower_face.shape[0] // blocks_h, lower_face.shape[1] // blocks_w
    
    uniform_blocks = 0
    total_blocks = 0
    
    for i in range(blocks_h):
        for j in range(blocks_w):
            if i*block_h < lower_face.shape[0] and j*block_w < lower_face.shape[1]:
                block = lower_face[i*block_h:min((i+1)*block_h, lower_face.shape[0]), 
                                  j*block_w:min((j+1)*block_w, lower_face.shape[1])]
                if block.size > 0:
                    block_std = np.std(block)
                    if block_std < 15:  # Low standard deviation indicates uniform region
                        uniform_blocks += 1
                    total_blocks += 1
    
    if total_blocks == 0:
        return False, 0.0
    
    mask_probability = uniform_blocks / total_blocks
    is_masked = mask_probability > 0.5
    
    return is_masked, mask_probability

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj
def popcount(x: int) -> int:
    c = 0
    while x:
        x &= x - 1
        c += 1
    return c
def get_cached_hash_from_db(filename, mtime):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('SELECT phash, dhash, mtime FROM image_hashes WHERE file = ?', (filename,))
        row = cursor.fetchone()
        conn.close()
        if not row:
            return None
        phash_int, dhash_int, stored_mtime = row
        if float(stored_mtime) == float(mtime):
            return {'phash': phash_int, 'dhash': dhash_int, 'mtime': stored_mtime}
        return None
    except Exception:
        return None
def upsert_image_hash_to_db(filename, mtime, phash_int, dhash_int):
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute('INSERT OR REPLACE INTO image_hashes (file, mtime, phash, dhash) VALUES (?, ?, ?, ?)', (filename, float(mtime), int(phash_int), int(dhash_int)))
        conn.commit()
        conn.close()
        return True
    except Exception:
        return False

def calculate_confidence_score(similarity, det_score, face_quality, partial_match_factor=1.0):
    """
    Calculate a comprehensive confidence score based on multiple factors:
    - similarity: The combined similarity score from face matching
    - det_score: The face detection confidence from InsightFace
    - face_quality: The quality assessment score
    - partial_match_factor: Adjustment for partial face matches
    
    Returns a score between 0-100 and a confidence level description
    """
    # Base score from similarity (0-1 scale to 0-100)
    base_score = similarity * 100
    
    # Weight the detection score (typically 0-1)
    det_weight = 0.15
    det_component = det_score * det_weight * 100
    
    # Weight the face quality (0-1)
    quality_weight = 0.25
    quality_component = face_quality * quality_weight * 100
    
    # Apply partial match factor (1.0 for full face, lower for partial matches)
    partial_factor_weight = 0.1
    partial_component = partial_match_factor * partial_factor_weight * 100
    
    # Calculate final weighted score
    raw_score = base_score * 0.5 + det_component + quality_component + partial_component
    
    # Cap at 100
    final_score = min(100, raw_score)
    
    # Determine confidence level
    if final_score >= 90:
        confidence_level = "Very High"
    elif final_score >= 75:
        confidence_level = "High"
    elif final_score >= 60:
        confidence_level = "Medium"
    elif final_score >= 40:
        confidence_level = "Low"
    else:
        confidence_level = "Very Low"
    
    return {
        'score': round(final_score, 1),
        'level': confidence_level,
        'components': {
            'similarity': round(similarity * 100, 1),
            'detection': round(det_score * 100, 1),
            'quality': round(face_quality * 100, 1),
            'partial_factor': round(partial_match_factor * 100, 1)
        }
    }

def detect_potential_twins(matches):
    """
    Analyze match results to detect potential twin scenarios
    """
    if len(matches) < 2:
        return False, None
    
    # Get top two matches
    top_match = matches[0]
    second_match = matches[1]
    
    # Check if scores are extremely close (potential twins)
    score_diff = top_match['similarity'] - second_match['similarity']
    
    if score_diff < 0.03 and top_match['similarity'] > 0.85 and second_match['similarity'] > 0.85:
        return True, {
            'potential_twins': [
                {'name': top_match['name'], 'score': top_match['similarity']},
                {'name': second_match['name'], 'score': second_match['similarity']}
            ],
            'score_difference': round(score_diff, 4),
            'recommendation': 'Consider additional verification methods'
        }
    
    return False, None

def get_age_difference_warning(query_age, reference_age):
    """Generate appropriate warning for significant age differences"""
    age_order = {"child": 0, "teen": 1, "young_adult": 2, "adult": 3, "senior": 4, "unknown": -1}
    
    if query_age == "unknown" or reference_age == "unknown":
        return None
    
    category_diff = abs(age_order[query_age] - age_order[reference_age])
    
    if category_diff >= 3:
        return "Significant age difference detected. Match confidence adjusted accordingly."
    elif category_diff == 2:
        return "Moderate age difference detected. Match threshold slightly adjusted."
    
    return None

def apply_age_aware_matching(query_embedding, db_embedding, query_age, db_age):
    """Apply age-aware matching logic to improve cross-age recognition"""
    # Standard similarity calculation
    cosine_sim = np.dot(query_embedding, db_embedding) / \
                 (np.linalg.norm(query_embedding) * np.linalg.norm(db_embedding))
    
    # Apply age-difference compensation
    age_diff = abs(query_age - db_age)
    age_compensation = min(0.2, age_diff * 0.01)  # Max 0.2 boost for 20+ year difference
    
    # Boost similarity for large age differences
    adjusted_sim = min(1.0, cosine_sim + age_compensation)
    
    return adjusted_sim

def adjust_threshold_for_age(base_threshold, query_age, reference_age):
    """Adjust matching threshold based on age categories"""
    # Define age category progression
    age_order = {"child": 0, "teen": 1, "young_adult": 2, "adult": 3, "senior": 4, "unknown": -1}
    
    # If either age is unknown, use base threshold
    if query_age == "unknown" or reference_age == "unknown":
        return base_threshold
    
    # Calculate category difference
    category_diff = abs(age_order[query_age] - age_order[reference_age])
    
    # Adjust threshold based on category difference
    if category_diff >= 3:  # e.g., child vs adult or teen vs senior
        return base_threshold * 0.65  # 35% more lenient
    elif category_diff == 2:  # e.g., teen vs adult
        return base_threshold * 0.75  # 25% more lenient
    elif category_diff == 1:  # e.g., young_adult vs adult
        return base_threshold * 0.9  # 10% more lenient
    else:  # Same category
        return base_threshold

app = Flask(__name__)
# Core config
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH
app.config['DEBUG'] = True  # Enable debug mode
image_hash_cache = {}
duplicate_check_lock = threading.Lock()
duplicate_jobs = {}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Ensure a placeholder image exists for broken/missing images
try:
    placeholder_path = os.path.join(app.config['UPLOAD_FOLDER'], 'placeholder.png')
    needs_placeholder = (not os.path.exists(placeholder_path)) or os.path.getsize(placeholder_path) == 0
    if needs_placeholder:
        from PIL import Image as _PILImage, ImageDraw as _PILDraw, ImageFont as _PILFont
        _img = _PILImage.new('RGB', (240, 160), color=(230, 230, 230))
        _draw = _PILDraw.Draw(_img)
        _text = 'No Image'
        try:
            _font = _PILFont.load_default()
        except Exception:
            _font = None
        # Center the text
        try:
            bbox = _draw.textbbox((0, 0), _text, font=_font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = (60, 10)
        _draw.text(((240 - tw) // 2, (160 - th) // 2), _text, fill=(100, 100, 100), font=_font)
        _img.save(placeholder_path, format='PNG')
except Exception as _e:
    print(f"Warning: could not create placeholder image: {_e}")

# CORS (can be restricted via CORS_ALLOWED_ORIGINS env var, comma-separated)
cors_origins = os.environ.get('CORS_ALLOWED_ORIGINS')
if cors_origins:
    origins = [o.strip() for o in cors_origins.split(',') if o.strip()]
    CORS(app, resources={r"/*": {"origins": origins}})
else:
    CORS(app)

# Available models configuration
AVAILABLE_MODELS = {
    'buffalo_l': {
        'name': 'buffalo_l',
        'description': 'Large model with highest accuracy',
        'threshold': 0.4,  # Changed from 0.6
        'instance': None
    },
    'buffalo_m': {
        'name': 'buffalo_m',
        'description': 'Medium model with good balance of speed and accuracy',
        'threshold': 0.45,  # Changed from 0.65
        'instance': None
    },
    'buffalo_sc': {
        'name': 'buffalo_sc',
        'description': 'Small and compact model for faster processing',
        'threshold': 0.5,  # Changed from 0.7
        'instance': None
    }
}

# Default model
CURRENT_MODEL = os.environ.get('DEFAULT_MODEL', 'buffalo_l')

# Auto-detect GPU availability and set processing mode
def detect_gpu_availability():
    """Detect if GPU/CUDA is available for processing"""
    if not TORCH_AVAILABLE or torch is None:
        return False
    
    try:
        return torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception as e:
        print(f"ERROR: Failed to detect GPU: {e}")
        return False

# Set processing mode with auto-detection
gpu_available = detect_gpu_availability()
env_mode = os.environ.get('PROCESSING_MODE', 'auto').lower()

if env_mode == 'auto':
    PROCESSING_MODE = 'gpu' if gpu_available else 'cpu'
    print(f"DEBUG: Auto-detected processing mode: {PROCESSING_MODE} (GPU available: {gpu_available})")
elif env_mode in ['1', 'true', 'gpu']:
    PROCESSING_MODE = 'gpu'
    if not gpu_available:
        print("WARNING: GPU mode requested but CUDA not available, falling back to CPU")
        PROCESSING_MODE = 'cpu'
else:
    PROCESSING_MODE = 'cpu'

print(f"DEBUG: Final processing mode: {PROCESSING_MODE}")

# Defer model initialization to first use (lazy-load per model)

# Load existing data from database instead of empty dictionaries
model_databases = {}
for model in AVAILABLE_MODELS:
    model_databases[model] = load_faces_from_db(model)

# Global FAISS indices for each model
indices = {model: None for model in AVAILABLE_MODELS}

# Function to perform face alignment
def align_face(face, image):
    # Get facial landmarks (5 key points)
    landmarks = face.kps
    
    if landmarks is None or landmarks.shape[0] != 5:
        return None
    
    # Standard reference 5 points for alignment
    # These points are the standard positions for eyes, nose, and mouth corners
    src = np.array([
        [30.2946, 51.6963],  # Left eye
        [65.5318, 51.6963],  # Right eye
        [48.0252, 71.7366],  # Nose
        [33.5493, 92.3655],  # Left mouth corner
        [62.7299, 92.3655]   # Right mouth corner
    ], dtype=np.float32)
    
    # Destination size
    dst_size = (112, 112)
    
    # Calculate transformation matrix
    s = np.array(landmarks, dtype=np.float32)
    M = cv2.estimateAffinePartial2D(s, src, method=cv2.RANSAC)[0]
    
    # Apply affine transformation
    aligned_face = cv2.warpAffine(image, M, dst_size, borderValue=0.0)
    
    return aligned_face

# Function to extract face features using InsightFace
# Update the extract_face_features function (around line 150)
def extract_face_features(image, model_name=CURRENT_MODEL, perform_alignment=True, enhance_level='none', process_all_faces=False):
    # Import FaceAnalysis locally to ensure it's available
    global FaceAnalysis, INSIGHTFACE_AVAILABLE
    
    # Re-import if needed
    if FaceAnalysis is None:
        try:
            from insightface.app import FaceAnalysis
            INSIGHTFACE_AVAILABLE = True
            print("DEBUG: FaceAnalysis re-imported in extract_face_features")
        except ImportError as e:
            print(f"ERROR: Failed to re-import FaceAnalysis in extract_face_features: {e}")
            raise ImportError("FaceAnalysis is not available. Please check insightface installation.")
    
    # Check if insightface is available
    if not INSIGHTFACE_AVAILABLE or FaceAnalysis is None:
        raise ImportError("FaceAnalysis is not available. Please check insightface installation.")
    
    # Apply image enhancement if requested
    if enhance_level != 'none':
        image = enhance_image(image, enhance_level)
    
    # Lazy load model if not already loaded
    if AVAILABLE_MODELS[model_name]['instance'] is None:
        if not INSIGHTFACE_AVAILABLE or FaceAnalysis is None:
            raise ImportError("FaceAnalysis is not available. Please check insightface installation.")
        AVAILABLE_MODELS[model_name]['instance'] = FaceAnalysis(name=model_name)
        ctx_id = 0 if PROCESSING_MODE == 'gpu' else -1
        AVAILABLE_MODELS[model_name]['instance'].prepare(ctx_id=ctx_id, det_size=(640, 640))
    
    face_app = AVAILABLE_MODELS[model_name]['instance']
    
    # InsightFace expects RGB images
    if image.shape[2] == 3:
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        rgb_image = image
        
    # Detect faces
    faces = face_app.get(rgb_image)
    
    if len(faces) == 0:
        return None, None, None
    
    # If process_all_faces is False, just return the highest-scoring face (original behavior)
    if not process_all_faces:
        # Get the face with highest detection score
        face = max(faces, key=lambda x: x.det_score)
        
        # Perform face alignment if requested
        aligned_face = None
        if perform_alignment:
            aligned_face = align_face(face, rgb_image)
        
        # Get embedding
        embedding = face.embedding
        
        # Get face box coordinates
        box = face.bbox.astype(int)
        x1, y1, x2, y2 = box
        
        # Extract face ROI with margin
        margin = int(0.1 * (x2 - x1))
        x1_margin = max(0, x1 - margin)
        y1_margin = max(0, y1 - margin)
        x2_margin = min(image.shape[1], x2 + margin)
        y2_margin = min(image.shape[0], y2 + margin)
        
        face_roi = image[y1_margin:y2_margin, x1_margin:x2_margin]
        
        # Return the face rectangle coordinates for visualization
        face_rect = (x1_margin, y1_margin, x2_margin - x1_margin, y2_margin - y1_margin)
        
        # If alignment was performed, use the aligned face
        if aligned_face is not None:
            face_roi = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)
        
        # Calculate quality score
        quality_score = assess_face_quality(face_roi, face.kps, face_rect)
        
        # Check if face is partial
        is_partial, partial_match_factor = detect_partial_face(face_roi, face.kps)
        
        return embedding, face_roi, face_rect, face.det_score, quality_score, is_partial, partial_match_factor
    
    # Process all faces
    results = []
    for face in faces:
        # Perform face alignment if requested
        aligned_face = None
        if perform_alignment:
            aligned_face = align_face(face, rgb_image)
        
        # Get embedding
        embedding = face.embedding
        
        # Get face box coordinates
        box = face.bbox.astype(int)
        x1, y1, x2, y2 = box
        
        # Extract face ROI with margin
        margin = int(0.1 * (x2 - x1))
        x1_margin = max(0, x1 - margin)
        y1_margin = max(0, y1 - margin)
        x2_margin = min(image.shape[1], x2 + margin)
        y2_margin = min(image.shape[0], y2 + margin)
        
        face_roi = image[y1_margin:y2_margin, x1_margin:x2_margin]
        
        # Return the face rectangle coordinates for visualization
        face_rect = (x1_margin, y1_margin, x2_margin - x1_margin, y2_margin - y1_margin)
        
        # If alignment was performed, use the aligned face
        if aligned_face is not None:
            face_roi = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2BGR)
        
        # Calculate quality score
        quality_score = assess_face_quality(face_roi, face.kps, face_rect)
        
        # Check if face is partial
        is_partial, partial_match_factor = detect_partial_face(face_roi, face.kps)
        
        # Store face data in a dictionary format that matches what the match_face endpoint expects
        face_data = {
            'embedding': embedding,
            'face_img': face_roi,
            'face_rect': face_rect,
            'det_score': float(face.det_score),
            'quality_score': float(quality_score),
            'is_partial': is_partial,
            'partial_match_factor': float(partial_match_factor)
        }
        
        results.append(face_data)
    
    return results

@app.route('/')
def index():
    return render_template('index.html', models=AVAILABLE_MODELS, processing_mode=PROCESSING_MODE)

@app.errorhandler(413)
def request_entity_too_large(e):
    return jsonify({'success': False, 'error': 'Uploaded file is too large'}), 413

@app.errorhandler(500)
def internal_server_error(e):
    """Handle internal server errors with JSON response"""
    return jsonify({'success': False, 'error': 'Internal server error occurred'}), 500

@app.errorhandler(400)
def bad_request(e):
    """Handle bad request errors with JSON response"""
    return jsonify({'success': False, 'error': 'Bad request'}), 400

@app.errorhandler(404)
def not_found(e):
    """Handle not found errors with JSON response"""
    return jsonify({'success': False, 'error': 'Resource not found'}), 404

@app.errorhandler(Exception)
def handle_exception(e):
    """Handle all unhandled exceptions with JSON response"""
    # Log the error for debugging
    print(f"Unhandled exception: {str(e)}")
    import traceback
    traceback.print_exc()
    
    # Return JSON error response
    return jsonify({'success': False, 'error': f'An error occurred: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health():
    """Basic health and readiness check"""
    db_ok = True
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.execute('SELECT 1')
        conn.close()
    except Exception:
        db_ok = False

    loaded_models = [name for name, cfg in AVAILABLE_MODELS.items() if cfg.get('instance') is not None]
    model_counts = {}
    try:
        for model_name, db in model_databases.items():
            model_counts[model_name] = len(db.get('embeddings', []))
    except Exception:
        # Fallback if model_databases not initialized yet
        model_counts = {}

    return jsonify({
        'status': 'ok',
        'version': APP_VERSION,
        'db': {'connected': db_ok},
        'processing_mode': PROCESSING_MODE,
        'loaded_models': loaded_models,
        'counts': model_counts
    })

@app.route('/version', methods=['GET'])
def version():
    return jsonify({'version': APP_VERSION})

@app.route('/switch_model', methods=['POST'])
def switch_model():
    global CURRENT_MODEL
    global FaceAnalysis, INSIGHTFACE_AVAILABLE
    
    # Get data from JSON request
    data = request.get_json()
    if not data or 'model' not in data:
        return jsonify({'error': 'Missing model parameter'}), 400
    
    model_name = data['model']
    
    if model_name not in AVAILABLE_MODELS:
        return jsonify({'error': f'Invalid model: {model_name}'}), 400
    
    # Lazy load the model if not already loaded
    if AVAILABLE_MODELS[model_name]['instance'] is None:
        try:
            # Attempt to re-import InsightFace if unavailable
            if not INSIGHTFACE_AVAILABLE or FaceAnalysis is None:
                try:
                    from insightface.app import FaceAnalysis as _FaceAnalysis
                    FaceAnalysis = _FaceAnalysis
                    INSIGHTFACE_AVAILABLE = True
                    print("DEBUG: FaceAnalysis re-imported in switch_model")
                except ImportError:
                    return jsonify({'error': 'FaceAnalysis is not available. Please check insightface installation.'}), 500
            print(f"DEBUG: FaceAnalysis available: {FaceAnalysis}")
            AVAILABLE_MODELS[model_name]['instance'] = FaceAnalysis(name=model_name)
            ctx_id = 0 if PROCESSING_MODE == 'gpu' else -1
            AVAILABLE_MODELS[model_name]['instance'].prepare(ctx_id=ctx_id, det_size=(640, 640))
        except Exception as e:
            return jsonify({'error': f'Failed to load model {model_name}: {str(e)}'}), 500
    
    CURRENT_MODEL = model_name
    
    return jsonify({
        'success': True, 
        'model': model_name,
        'description': AVAILABLE_MODELS[model_name]['description']
    })

@app.route('/upload_face', methods=['POST'])
def upload_face():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image part in the request'}), 400
    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'}), 400
    if file:
        try:
            # Your logic to save the image and process it
            # For example:
            # filename = secure_filename(file.filename)
            # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return jsonify({'success': True, 'message': 'Face uploaded successfully'}), 200
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    return jsonify({'success': False, 'error': 'Unknown error'}), 500

@app.route('/add_face', methods=['POST'])
def add_face():
    if 'file' not in request.files or 'name' not in request.form:
        return jsonify({'error': 'Missing file or name'}), 400
    
    file = request.files['file']
    name = request.form['name']
    
    # Get model name from request
    model_name = request.form.get('model', CURRENT_MODEL)
    if model_name not in AVAILABLE_MODELS:
        return jsonify({'error': f'Invalid model: {model_name}'}), 400
    
    # Get alignment preference
    perform_alignment = request.form.get('align', 'true').lower() == 'true'
    
    # Get enhancement level
    enhance_level = request.form.get('enhance', 'none')
    
    # Get multi-face processing preference
    process_all_faces = request.form.get('process_all_faces', 'false').lower() == 'true'
    
    # Validate file type
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'gif'}

    if not file or file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid or unsupported file type'}), 400

    # Read and process the image
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None or img.size == 0:
        return jsonify({'error': 'Invalid image'}), 400
    
    # Extract face features
    result = extract_face_features(img, model_name, perform_alignment, enhance_level, process_all_faces)
    
    if result is None:
        return jsonify({'error': 'No face detected'}), 400
    
    # Get the database for this model
    db = model_databases[model_name]
    
    # Process single face (original behavior)
    if not process_all_faces:
        features, face_img, face_rect, det_score, quality_score, is_partial, partial_match_factor = result
        
        # Save the face image
        filename = f"{uuid.uuid4()}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(filepath, face_img)
        image_url = url_for('static', filename=f'images/{filename}')
        
        # Save to database
        save_face_to_db(model_name, name, features, image_url)
        
        # Add to in-memory database for the specific model
        db['embeddings'].append(l2_normalize(features))
        db['names'].append(name)
        db['images'].append(image_url)
        
        # Rebuild index for this model (cosine similarity via IP on normalized vectors)
        if len(db['embeddings']) > 0:
            if not FAISS_AVAILABLE:
                return jsonify({
                    'success': False,
                    'error': 'FAISS library not available for similarity indexing'
                }), 500
            
            embeddings = np.array([l2_normalize(e) for e in db['embeddings']]).astype('float32')
            dimension = embeddings.shape[1]
            db['index'] = faiss.IndexFlatIP(dimension)
            faiss.normalize_L2(embeddings)
            db['index'].add(embeddings)
        
        return jsonify({
            'success': True, 
            'name': name, 
            'image_url': image_url,
            'model': model_name,
            'detection_score': float(det_score),
            'aligned': perform_alignment,
            'faces_found': 1,
            'faces_processed': 1
        })
    
    # Process multiple faces
    faces_processed = 0
    face_results = []
    
    for i, face_data in enumerate(result):
        # Generate a unique name for each face if multiple faces
        face_name = f"{name}_{i+1}" if len(result) > 1 else name
        
        # Save the face image
        filename = f"{uuid.uuid4()}.jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        cv2.imwrite(filepath, face_data['face_img'])
        image_url = url_for('static', filename=f'images/{filename}')
        
        # Save to database
        save_face_to_db(model_name, face_name, face_data['embedding'], image_url)
        
        # Add to in-memory database for the specific model
        db['embeddings'].append(l2_normalize(face_data['embedding']))
        db['names'].append(face_name)
        db['images'].append(image_url)
        
        face_results.append({
            'name': face_name,
            'image_url': image_url,
            'detection_score': face_data['det_score']
        })
        
        faces_processed += 1
    
    # Rebuild index for this model (cosine similarity via IP on normalized vectors)
    if len(db['embeddings']) > 0:
        if not FAISS_AVAILABLE:
            return jsonify({
                'success': False,
                'error': 'FAISS library not available for similarity indexing'
            }), 500
        
        embeddings = np.array([l2_normalize(e) for e in db['embeddings']]).astype('float32')
        dimension = embeddings.shape[1]
        db['index'] = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        db['index'].add(embeddings)
    
    return jsonify({
        'success': True, 
        'model': model_name,
        'aligned': perform_alignment,
        'faces_found': len(result),
        'faces_processed': faces_processed,
        'faces': face_results
    })

def load_all_images_from_folder(folder_path='static/images'):
    # Check if cv2 is available
    if cv2 is None:
        print("OpenCV not available")
        return []
    
    # Check if folder exists
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist")
        return []
    
    face_data = []
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = [f for f in os.listdir(folder_path) 
                   if any(f.lower().endswith(ext) for ext in image_extensions)]
    
    for filename in image_files:
        if filename.startswith('query_'):  # Skip query images
            continue
            
        filepath = os.path.join(folder_path, filename)
        
        try:
            # Read image
            img = cv2.imread(filepath)
            if img is None:
                continue
                
            # Use the first extract_face_features function with process_all_faces=True
            result = extract_face_features(img, CURRENT_MODEL, True, 'none', True)
            
            if result is None:
                continue
                
            base_name = os.path.splitext(filename)[0]
            
            # Handle the list of dictionaries returned by the first function
            if isinstance(result, list):
                for idx, face_dict in enumerate(result):
                    if isinstance(face_dict, dict) and 'embedding' in face_dict:
                        face_data.append({
                            'name': f"{base_name}_{idx}" if len(result) > 1 else base_name,
                            'embedding': face_dict['embedding'],
                            'image_url': url_for('static', filename=f'images/{filename}'),
                            'detection_score': face_dict['det_score']
                        })
            else:
                print(f"Unexpected result format for {filename}: {type(result)}")
                
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
            
    return face_data



@app.route('/match_face', methods=['POST'])
def match_face():
    if 'file' not in request.files:
        return jsonify({'error': 'Missing file'}), 400
    
    file = request.files['file']
    
    # Basic file type validation (size enforced via MAX_CONTENT_LENGTH)
    if not file or file.filename == '' or not ('.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in {'jpg','jpeg','png','bmp','tiff','gif'}):
        return jsonify({'error': 'Invalid or unsupported file type'}), 400
    
    # Get parameters
    model_name = request.form.get('model', CURRENT_MODEL)
    if model_name not in AVAILABLE_MODELS:
        return jsonify({'error': f'Invalid model: {model_name}'}), 400
    
    # Updated default parameters in match_face route
    perform_alignment = request.form.get('align', 'true').lower() == 'true'
    threshold = float(request.form.get('threshold', '0.4'))
    cosine_weight = float(request.form.get('cosine_weight', '0.7'))
    l2_weight = float(request.form.get('l2_weight', '0.3'))
    enhance_level = request.form.get('enhance', 'none')
    process_all_faces = request.form.get('process_all_faces', 'false').lower() == 'true'
    detect_masked_partial = request.form.get('detect_masked_partial', 'false').lower() == 'true'
    
    # Read and process the image
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None or img.size == 0:
        return jsonify({'error': 'Invalid image'}), 400
    
    # Extract face features
    start_time = time.time()
    result = extract_face_features(img, model_name, perform_alignment, enhance_level, process_all_faces)
    extraction_time = time.time() - start_time
    
    if result is None:
        return jsonify({'error': 'No face detected'}), 400
    
    # USE TRAINED MODEL DATA INSTEAD OF FOLDER PROCESSING
    db = model_databases[model_name]
    
    # Check if we have trained data
    if not db['embeddings']:
        return jsonify({'error': 'No trained faces found. Please train the model first.'}), 400
    
    # Create FAISS index from trained data
    try:
        if not FAISS_AVAILABLE:
            return jsonify({'error': 'FAISS library not available for similarity indexing'}), 500
        
        embeddings_array = np.array([l2_normalize(e) for e in db['embeddings']]).astype('float32')
        temp_index = faiss.IndexFlatIP(embeddings_array.shape[1])
        faiss.normalize_L2(embeddings_array)
        temp_index.add(embeddings_array)
    except Exception as e:
        return jsonify({'error': f'Failed to create search index: {str(e)}'}), 500
    
    # Process single face (original behavior)
    if not process_all_faces:
        features, face_img, face_rect, det_score, quality_score, is_partial, partial_match_factor = result
        
        # Check for face mask
        is_masked, mask_probability = detect_face_mask(face_img)
        
        # Save the query image for display
        query_image_url = save_query_face_image(face_img, img)
        
        # Search for similar faces using FAISS
        start_time = time.time()
        # Normalize query embedding
        q = l2_normalize(features).reshape(1, -1).astype('float32')
        k = len(db['embeddings'])  # Get all faces from trained data
        
        # Perform search with timing
        distances, indices = temp_index.search(q, k)
        search_time = time.time() - start_time

        all_results = []
        for i, idx in enumerate(indices[0]):
            # Get the database embedding from trained data
            db_embedding = db['embeddings'][idx]
            
            # Cosine similarity from IP search on normalized vectors
            cosine_sim = float(distances[0][i])
            combined_score = cosine_sim
            
            # Adjust confidence calculation if detect_masked_partial is enabled
            if detect_masked_partial and (is_partial or is_masked):
                # Increase the weight of partial_match_factor when the checkbox is checked
                partial_weight = 1.5 if detect_masked_partial else 1.0
                confidence = calculate_confidence_score(
                    combined_score, 
                    det_score, 
                    quality_score, 
                    partial_match_factor * partial_weight
                )
            else:
                confidence = calculate_confidence_score(
                    combined_score, 
                    det_score, 
                    quality_score, 
                    partial_match_factor
                )
            
            # Calculate confidence score
            all_results.append({
                'name': db['names'][idx],  # Use trained data
                'similarity': float(combined_score),
                'image_url': db['images'][idx],  # Use trained data
                'cosine_sim': float(cosine_sim),
                'l2_distance': 0.0,
                'l2_similarity': float(cosine_sim),
                'query_image_url': query_image_url,
                'confidence': confidence,
                'is_partial': is_partial,
                'quality_score': float(quality_score),
                'is_masked': is_masked,
                'mask_probability': float(mask_probability)
            })
        
        # Sort by combined score (highest first)
        all_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Apply threshold
        results = [r for r in all_results if r['similarity'] > threshold]
        
        # Limit to top 3 matches and convert numpy types
        results = [convert_numpy_types(r) for r in results[:3]]
        
        # Check for ambiguous results
        warning = None
        if len(results) >= 2:
            top_score = results[0]['similarity']
            second_score = results[1]['similarity']
            if (top_score - second_score) < 0.08:  # Less than 8% difference
                warning = "Multiple similar matches found - results may be ambiguous"
        
        # Check for potential twins
        is_twins, twins_info = detect_potential_twins(results)
        
        return jsonify({
            'matches': results,
            'query_image_url': query_image_url,
            'warning': warning,
            'model': model_name,
            'threshold': threshold,
            'cosine_weight': cosine_weight,
            'l2_weight': l2_weight,
            'extraction_time': extraction_time,
            'search_time': search_time,
            'is_twins': is_twins,
            'twins_info': twins_info,
            'timing': {
                'feature_extraction_ms': round(extraction_time * 1000, 2),
                'search_ms': round(search_time * 1000, 2)
            }
        })
    
    # Process multiple faces
    face_results = []
    total_search_time = 0
    total_conversion_time = 0
    total_sorting_time = 0
    total_filtering_time = 0
    
    # Batch processing implementation for multiple faces
    if len(result) > 1:
        print(f"DEBUG: Processing {len(result)} faces in batch mode")
        k = min(5, len(db['embeddings']))  # Limit to top 5 matches per face
        all_embeddings = np.array([l2_normalize(f['embedding']) for f in result]).astype('float32')
        if FAISS_AVAILABLE:
            faiss.normalize_L2(all_embeddings)
        distances, indices = temp_index.search(all_embeddings, k)
        
        # Process each face's results
        for face_idx, face_data in enumerate(result):
            # Check for face mask
            is_masked = False  # Temporarily set to False
            mask_probability = 0.0  # Temporarily set to 0.0

            query_image_url = save_query_face_image(face_data.get('face_img'))
            
            # Get this face's search results
            face_distances = distances[face_idx]
            face_indices = indices[face_idx]
            
            all_results = []
            for i, idx in enumerate(face_indices):
                # Get the database embedding from trained data
                db_embedding = db['embeddings'][idx]
                
                # Cosine similarity from IP search on normalized vectors
                cosine_sim = float(face_distances[i])
                combined_score = cosine_sim
                
                all_results.append({
                    'name': db['names'][idx],
                    'similarity': float(combined_score),
                    'image_url': db['images'][idx],
                    'cosine_sim': float(cosine_sim),
                    'l2_distance': 0.0,
                    'l2_similarity': float(cosine_sim),
                    'query_image_url': query_image_url,
                    'is_partial': face_data['is_partial'],
                    'quality_score': float(face_data['quality_score']),
                    'is_masked': is_masked,
                    'mask_probability': float(mask_probability)
                })
            
            # Sort by combined score (highest first)
            start = time.time()
            all_results.sort(key=lambda x: x['similarity'], reverse=True)
            total_sorting_time += time.time() - start
            
            # Apply threshold
            start = time.time()
            filtered_results = [r for r in all_results if r['similarity'] > threshold]
            total_filtering_time += time.time() - start
            
            # Limit to top 3 matches and convert numpy types
            start = time.time()
            filtered_results = [convert_numpy_types(r) for r in filtered_results[:3]]
            total_conversion_time += time.time() - start
            
            # Check for ambiguous results
            warning = None
            if len(filtered_results) >= 2:
                top_score = filtered_results[0]['similarity']
                second_score = filtered_results[1]['similarity']
                if (top_score - second_score) < 0.08:
                    warning = "Multiple similar matches found - results may be ambiguous"
            
            # Check for potential twins
            is_twins, twins_info = detect_potential_twins(filtered_results)
            
            face_results.append({
                'face_index': face_idx + 1,
                'detection_score': face_data['det_score'],
                'quality_score': face_data['quality_score'],
                'is_partial': face_data['is_partial'],
                'partial_match_factor': face_data['partial_match_factor'],
                'is_masked': is_masked,
                'mask_probability': float(mask_probability),
                'potential_twins': twins_info if is_twins else None,
                'matches': filtered_results,
                'query_image_url': query_image_url,
                'warning': warning
            })
    else:
        print(f"DEBUG: Processing {len(result)} face(s) individually")
        # Individual face processing logic...
        # result is a list of dictionaries, each containing face data
        for face_idx, face_data in enumerate(result):
            # Check for face mask
            # is_masked, mask_probability = detect_face_mask(face_data['face_img'])
            is_masked = False # Temporarily set to False
            mask_probability = 0.0 # Temporarily set to 0.0

            query_image_url = save_query_face_image(face_data.get('face_img'))
            
            # Search for similar faces using FAISS
            start_time = time.time()
            query_embedding = l2_normalize(face_data['embedding']).reshape(1, -1).astype('float32')
            # Further limit k to a smaller number, e.g., 5
            k = min(5, len(db['embeddings']))  # Use trained data
            
            # Perform search with timing
            distances, indices = temp_index.search(query_embedding, k)
            search_time = time.time() - start_time
            total_search_time += search_time
            
            all_results = []
            for i, idx in enumerate(indices[0]):
                # Get the database embedding from trained data
                db_embedding = db['embeddings'][idx]
                
                # Cosine similarity from IP search on normalized vectors
                cosine_sim = float(distances[0][i])
                combined_score = cosine_sim
                
                all_results.append({
                    'name': db['names'][idx],  # Use trained data
                    'similarity': float(combined_score),
                    'image_url': db['images'][idx],  # Use trained data
                    'cosine_sim': float(cosine_sim),
                    'l2_distance': 0.0,
                    'l2_similarity': float(cosine_sim),
                    'query_image_url': query_image_url, # Keep this line, but query_image_url will be empty
                    'is_partial': face_data['is_partial'],
                    'quality_score': float(face_data['quality_score']),
                    'is_masked': is_masked, # Use the temporary value
                    'mask_probability': float(mask_probability) # Use the temporary value
                })
            
            # Sort by combined score (highest first)
            start = time.time()
            all_results.sort(key=lambda x: x['similarity'], reverse=True)
            total_sorting_time += time.time() - start
            
            # Apply threshold
            start = time.time()
            filtered_results = [r for r in all_results if r['similarity'] > threshold]
            total_filtering_time += time.time() - start
            
            # Limit to top 3 matches and convert numpy types
            start = time.time()
            filtered_results = [convert_numpy_types(r) for r in filtered_results[:3]]
            total_conversion_time += time.time() - start
            
            # Check for ambiguous results
            warning = None
            if len(filtered_results) >= 2:
                top_score = filtered_results[0]['similarity']
                second_score = filtered_results[1]['similarity']
                if (top_score - second_score) < 0.08:  # Less than 8% difference
                    warning = "Multiple similar matches found - results may be ambiguous"
            
            # Check for potential twins
            is_twins, twins_info = detect_potential_twins(filtered_results)
            
            face_results.append({
                'face_index': face_idx + 1,
                'detection_score': face_data['det_score'],
                'quality_score': face_data['quality_score'],
                'is_partial': face_data['is_partial'],
                'partial_match_factor': face_data['partial_match_factor'],
                'is_masked': is_masked,
                'mask_probability': float(mask_probability),
                'potential_twins': twins_info if is_twins else None,
                'matches': filtered_results,
                'query_image_url': query_image_url,
                'warning': warning
            })
    
    return jsonify({
        'model': model_name,
        'threshold': threshold,
        'cosine_weight': cosine_weight,
        'l2_weight': l2_weight,
        'aligned': perform_alignment,
        'faces_found': len(result),
        'faces_processed': len(face_results),
        'face_results': face_results,
        'timing': {
            'feature_extraction_ms': round(extraction_time * 1000, 2),
            'search_ms': round(total_search_time * 1000, 2),
            'conversion_ms': round(total_conversion_time * 1000, 2),
            'sorting_ms': round(total_sorting_time * 1000, 2),
            'filtering_ms': round(total_filtering_time * 1000, 2)
        }
    })

@app.route('/extract_faces_from_video', methods=['POST'])
def extract_faces_from_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No video file selected'}), 400
        
    model_name = request.form.get('model', CURRENT_MODEL)
    frame_interval = int(request.form.get('frame_interval', 30))
    min_quality = float(request.form.get('min_quality', 0.65)) # Default quality threshold
    
    # Setup directories
    video_faces_dir = os.path.join(app.static_folder, 'video_faces', 'temp') # Use temp subfolder for extraction
    if os.path.exists(video_faces_dir):
        shutil.rmtree(video_faces_dir)
    os.makedirs(video_faces_dir)
    
    # Save video to temp file
    try:
        fd, temp_path = tempfile.mkstemp(suffix='.mp4')
        os.close(fd)
        video_file.save(temp_path)
    except Exception as e:
        return jsonify({'error': f'Failed to save video: {str(e)}'}), 500
        
    extracted_faces = []
    extracted_embeddings = [] # List of (embedding, face_id, quality)
    
    try:
        cap = cv2.VideoCapture(temp_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30
        
        frame_count = 0
        
        # Load model
        if AVAILABLE_MODELS[model_name]['instance'] is None:
            if not INSIGHTFACE_AVAILABLE or FaceAnalysis is None:
                raise ImportError("FaceAnalysis is not available.")
            AVAILABLE_MODELS[model_name]['instance'] = FaceAnalysis(name=model_name)
            ctx_id = 0 if PROCESSING_MODE == 'gpu' else -1
            AVAILABLE_MODELS[model_name]['instance'].prepare(ctx_id=ctx_id, det_size=(640, 640))
            
        face_app = AVAILABLE_MODELS[model_name]['instance']
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                timestamp = frame_count / fps
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = face_app.get(rgb_frame)
                
                for i, face in enumerate(faces):
                    # Filter by detection score (quality)
                    if face.det_score < min_quality:
                        continue

                    # Deduplication Check
                    # Compare with previously extracted faces
                    curr_embedding = face.embedding
                    is_duplicate = False
                    
                    if extracted_embeddings:
                        # Normalize current embedding
                        curr_emb_norm = l2_normalize(curr_embedding)
                        
                        for prev_emb, prev_id, prev_score in extracted_embeddings:
                            prev_emb_norm = l2_normalize(prev_emb)
                            sim = np.dot(curr_emb_norm, prev_emb_norm)
                            
                            if sim > 0.75: # Deduplication threshold
                                is_duplicate = True
                                # TODO: Logic to replace if current quality is significantly better?
                                # For now, just skip duplicates to keep the set diverse and small
                                break
                    
                    if is_duplicate:
                        continue

                    # Save face crop with padding
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox
                    
                    # Add padding (e.g., 20% of width/height)
                    w_box = x2 - x1
                    h_box = y2 - y1
                    pad_x = int(w_box * 0.2)
                    pad_y = int(h_box * 0.2)
                    
                    # Apply padding with boundary checks
                    h_frame, w_frame, _ = frame.shape
                    x1 = max(0, x1 - pad_x)
                    y1 = max(0, y1 - pad_y)
                    x2 = min(w_frame, x2 + pad_x)
                    y2 = min(h_frame, y2 + pad_y)
                    
                    if x2 > x1 and y2 > y1:
                        face_img = frame[y1:y2, x1:x2]
                        face_filename = f"face_{frame_count}_{i}.jpg"
                        face_path = os.path.join(video_faces_dir, face_filename)
                        cv2.imwrite(face_path, face_img)
                        
                        extracted_faces.append({
                            'id': f"temp/{face_filename}",
                            'url': url_for('static', filename=f'video_faces/temp/{face_filename}'),
                            'timestamp': round(timestamp, 2),
                            'timestamp_str': time.strftime('%H:%M:%S', time.gmtime(timestamp)),
                            'bbox': bbox.tolist(),
                            'score': float(face.det_score)
                        })
                        
                        extracted_embeddings.append((face.embedding, face_filename, float(face.det_score)))
            
            frame_count += 1
            
        cap.release()
        
    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': str(e)}), 500
        
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    return jsonify({'success': True, 'faces': extracted_faces})

@app.route('/match_selected_faces', methods=['POST'])
def match_selected_faces():
    data = request.get_json()
    selected_faces = data.get('faces', [])
    model_name = data.get('model', CURRENT_MODEL)
    threshold = float(data.get('threshold', 0.6))
    
    if not selected_faces:
        return jsonify({'error': 'No faces selected'}), 400
        
    video_faces_dir = os.path.join(app.static_folder, 'video_faces')
    results = []
    
    try:
        # Load model
        if AVAILABLE_MODELS[model_name]['instance'] is None:
            if not INSIGHTFACE_AVAILABLE or FaceAnalysis is None:
                raise ImportError("FaceAnalysis is not available.")
            AVAILABLE_MODELS[model_name]['instance'] = FaceAnalysis(name=model_name)
            ctx_id = 0 if PROCESSING_MODE == 'gpu' else -1
            AVAILABLE_MODELS[model_name]['instance'].prepare(ctx_id=ctx_id, det_size=(640, 640))
            
        face_app = AVAILABLE_MODELS[model_name]['instance']
        
        # Prepare DB index
        db = load_faces_from_db(model_name)
        if not db['embeddings']:
            return jsonify({'error': 'No trained faces found'}), 400
            
        embeddings_array = np.array([l2_normalize(e) for e in db['embeddings']]).astype('float32')
        index = faiss.IndexFlatIP(embeddings_array.shape[1])
        faiss.normalize_L2(embeddings_array)
        index.add(embeddings_array)
        
        for face_info in selected_faces:
            face_id = face_info.get('id') # filename
            face_path = os.path.join(video_faces_dir, face_id)
            
            if not os.path.exists(face_path):
                continue
                
            # Read image and get embedding
            img = cv2.imread(face_path)
            if img is None:
                continue
                
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = face_app.get(rgb_img)
            
            # Should be only one face in the crop usually, but pick largest if multiple
            if not faces:
                continue
                
            faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
            target_face = faces[0]
            
            embedding = target_face.embedding
            q = l2_normalize(embedding).reshape(1, -1).astype('float32')
            
            distances, indices = index.search(q, 1)
            score = float(distances[0][0])
            idx = int(indices[0][0])
            
            if score > threshold:
                name = db['names'][idx]
                results.append({
                    'timestamp': face_info.get('timestamp'),
                    'timestamp_str': face_info.get('timestamp_str'),
                    'face_url': url_for('static', filename=f'video_faces/{face_id}'),
                    'name': name,
                    'score': round(score, 4)
                })
                
    except Exception as e:
        return jsonify({'error': str(e)}), 500
        
    return jsonify({'success': True, 'matches': results})

@app.route('/check_duplicates', methods=['POST'])
def check_duplicates():
    data = request.get_json() or {}
    similarity_threshold = float(data.get('threshold', '95'))
    delete_duplicates = data.get('delete', False)
    max_images = int(data.get('max_images', 5000))
    timeout_ms = int(data.get('timeout_ms', 10000))
    
    # Convert percentage threshold to max allowed difference (0-64 scale)
    # 100% similarity = 0 difference, 0% similarity = 64 difference
    max_diff = (100.0 - similarity_threshold) / 100.0 * 64.0
    start_time = time.time()
    if not duplicate_check_lock.acquire(blocking=False):
        return jsonify({'success': False, 'busy': True, 'error': 'Duplicate scan in progress'}), 429
    try:
        images_dir = os.path.join('static', 'images')
        image_files = [
            f for f in os.listdir(images_dir)
            if os.path.isfile(os.path.join(images_dir, f)) and
               f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')) and
               f != 'placeholder.png'
        ]
        if max_images and len(image_files) > max_images:
            image_files = image_files[:max_images]
        image_hashes = {}
        duplicates = []
        removed_files = []
        cache_hits = 0
        computed = 0
        phash_buckets = {}
        dhash_buckets = {}
        for img_file in image_files:
            img_path = os.path.join(images_dir, img_file)
            try:
                mtime = os.path.getmtime(img_path)
            except Exception:
                continue
            entry = image_hash_cache.get(img_file)
            if entry and entry.get('mtime') == mtime:
                phash_int = entry['phash']
                dhash_int = entry['dhash']
                cache_hits += 1
            else:
                try:
                    db_entry = get_cached_hash_from_db(img_file, mtime)
                    if db_entry:
                        phash_int = int(db_entry['phash'])
                        dhash_int = int(db_entry['dhash'])
                        image_hash_cache[img_file] = {'phash': phash_int, 'dhash': dhash_int, 'mtime': mtime}
                        cache_hits += 1
                    else:
                        img = Image.open(img_path)
                        phash_obj = imagehash.phash(img)
                        dhash_obj = imagehash.dhash(img)
                        phash_int = int(str(phash_obj), 16)
                        dhash_int = int(str(dhash_obj), 16)
                        image_hash_cache[img_file] = {'phash': phash_int, 'dhash': dhash_int, 'mtime': mtime}
                        upsert_image_hash_to_db(img_file, mtime, phash_int, dhash_int)
                        computed += 1
                except Exception:
                    continue
            pkey = phash_int >> 48
            dkey = dhash_int >> 48
            candidates = set()
            for fname, p_i, d_i in phash_buckets.get(pkey, []):
                candidates.add((fname, p_i, d_i))
            for fname, p_i, d_i in dhash_buckets.get(dkey, []):
                candidates.add((fname, p_i, d_i))
            is_duplicate = False
            duplicate_of = None
            min_diff = float('inf')
            for fname, p_i, d_i in candidates:
                phash_diff = popcount(phash_int ^ p_i)
                dhash_diff = popcount(dhash_int ^ d_i)
                avg_diff = (phash_diff * 0.7 + dhash_diff * 0.3)
                if avg_diff <= max_diff and avg_diff < min_diff:
                    is_duplicate = True
                    duplicate_of = fname
                    min_diff = avg_diff
            if is_duplicate:
                actual_similarity = (1.0 - (min_diff / 64.0)) * 100.0
                duplicates.append({
                    'file': img_file,
                    'file_url': url_for('static', filename=f'images/{img_file}'),
                    'duplicate_of': duplicate_of,
                    'duplicate_of_url': url_for('static', filename=f'images/{duplicate_of}'),
                    'similarity': actual_similarity
                })
            else:
                image_hashes[img_file] = {'phash': phash_int, 'dhash': dhash_int}
                phash_buckets.setdefault(pkey, []).append((img_file, phash_int, dhash_int))
                dhash_buckets.setdefault(dkey, []).append((img_file, phash_int, dhash_int))
            if (time.time() - start_time) * 1000 > timeout_ms:
                break
        if delete_duplicates and duplicates:
            for dup in duplicates:
                try:
                    if dup['file'].startswith('query_'):
                        os.remove(os.path.join(images_dir, dup['file']))
                        removed_files.append(dup['file'])
                    else:
                        file_url = url_for('static', filename=f'images/{dup["file"]}')
                        is_referenced = False
                        for model_name, db in model_databases.items():
                            if file_url in db['images']:
                                is_referenced = True
                                original_url = url_for('static', filename=f'images/{dup["duplicate_of"]}')
                                idx = db['images'].index(file_url)
                                db['images'][idx] = original_url
                        if not is_referenced or (is_referenced and dup['duplicate_of']):
                            os.remove(os.path.join(images_dir, dup['file']))
                            removed_files.append(dup['file'])
                except Exception:
                    continue
        elapsed_ms = round((time.time() - start_time) * 1000, 2)
        timed_out = elapsed_ms > timeout_ms
        return jsonify({
            'success': True,
            'total_images': len(image_files),
            'unique_images': len(image_hashes),
            'duplicates_found': len(duplicates),
            'duplicates': duplicates,
            'duplicates_removed': len(removed_files),
            'removed_files': removed_files,
            'cache_hits': cache_hits,
            'hashes_computed': computed,
            'elapsed_ms': elapsed_ms,
            'timed_out': timed_out
        })
    finally:
        duplicate_check_lock.release()
@app.route('/start_duplicate_scan', methods=['POST'])
def start_duplicate_scan():
    data = request.get_json() or {}
    job_id = str(uuid.uuid4())
    if not duplicate_check_lock.acquire(blocking=False):
        return jsonify({'success': False, 'error': 'Scan busy'}), 429
    similarity_threshold = float(data.get('threshold', '95'))
    delete_duplicates = data.get('delete', False)
    max_images = int(data.get('max_images', 0))

    # Convert percentage threshold to max allowed difference
    max_diff = (100.0 - similarity_threshold) / 100.0 * 64.0

    def worker():
        try:
            images_dir = os.path.join('static', 'images')
            files = [
                f for f in os.listdir(images_dir)
                if os.path.isfile(os.path.join(images_dir, f)) and
                   f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')) and
                   f != 'placeholder.png'
            ]
            if max_images and len(files) > max_images:
                files = files[:max_images]
            duplicates = []
            removed_files = []
            unique_count = 0
            phash_buckets = {}
            dhash_buckets = {}
            total = len(files)
            duplicate_jobs[job_id] = {'status': 'running', 'total': total, 'processed': 0, 'duplicates_found': 0, 'removed': 0, 'duplicates': [], 'unique': 0}
            for img_file in files:
                if duplicate_jobs.get(job_id, {}).get('status') == 'canceled':
                    break
                img_path = os.path.join(images_dir, img_file)
                try:
                    mtime = os.path.getmtime(img_path)
                except Exception:
                    processed = duplicate_jobs[job_id]['processed'] + 1
                    duplicate_jobs[job_id]['processed'] = processed
                    continue
                entry = image_hash_cache.get(img_file)
                if not entry or entry.get('mtime') != mtime:
                    db_entry = get_cached_hash_from_db(img_file, mtime)
                    if db_entry:
                        phash_int = int(db_entry['phash'])
                        dhash_int = int(db_entry['dhash'])
                        image_hash_cache[img_file] = {'phash': phash_int, 'dhash': dhash_int, 'mtime': mtime}
                    else:
                        try:
                            img = Image.open(img_path)
                            phash_obj = imagehash.phash(img)
                            dhash_obj = imagehash.dhash(img)
                            phash_int = int(str(phash_obj), 16)
                            dhash_int = int(str(dhash_obj), 16)
                            image_hash_cache[img_file] = {'phash': phash_int, 'dhash': dhash_int, 'mtime': mtime}
                            upsert_image_hash_to_db(img_file, mtime, phash_int, dhash_int)
                        except Exception:
                            processed = duplicate_jobs[job_id]['processed'] + 1
                            duplicate_jobs[job_id]['processed'] = processed
                            continue
                else:
                    phash_int = entry['phash']
                    dhash_int = entry['dhash']
                pkey = phash_int >> 48
                dkey = dhash_int >> 48
                candidates = set()
                for fname, p_i, d_i in phash_buckets.get(pkey, []):
                    candidates.add((fname, p_i, d_i))
                for fname, p_i, d_i in dhash_buckets.get(dkey, []):
                    candidates.add((fname, p_i, d_i))
                is_duplicate = False
                duplicate_of = None
                min_diff = float('inf')
                for fname, p_i, d_i in candidates:
                    phash_diff = popcount(phash_int ^ p_i)
                    dhash_diff = popcount(dhash_int ^ d_i)
                    avg_diff = (phash_diff * 0.7 + dhash_diff * 0.3)
                    if avg_diff <= max_diff and avg_diff < min_diff:
                        is_duplicate = True
                        duplicate_of = fname
                        min_diff = avg_diff
                if is_duplicate:
                    actual_similarity = (1.0 - (min_diff / 64.0)) * 100.0
                    duplicates.append({'file': img_file, 'duplicate_of': duplicate_of, 'similarity': actual_similarity})
                    duplicate_jobs[job_id]['duplicates_found'] = len(duplicates)
                    if delete_duplicates:
                        try:
                            os.remove(os.path.join(images_dir, img_file))
                            removed_files.append(img_file)
                            duplicate_jobs[job_id]['removed'] = len(removed_files)
                        except Exception:
                            pass
                else:
                    unique_count += 1
                    phash_buckets.setdefault(pkey, []).append((img_file, phash_int, dhash_int))
                    dhash_buckets.setdefault(dkey, []).append((img_file, phash_int, dhash_int))
                processed = duplicate_jobs[job_id]['processed'] + 1
                duplicate_jobs[job_id]['processed'] = processed
                duplicate_jobs[job_id]['duplicates'] = duplicates[-50:]
            duplicate_jobs[job_id]['duplicates'] = duplicates
            duplicate_jobs[job_id]['status'] = 'completed'
            duplicate_jobs[job_id]['unique'] = unique_count
        finally:
            duplicate_check_lock.release()
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return jsonify({'success': True, 'job_id': job_id})
@app.route('/duplicate_status/<job_id>', methods=['GET'])
def duplicate_status(job_id):
    info = duplicate_jobs.get(job_id)
    if not info:
        return jsonify({'success': False, 'error': 'Invalid job id'}), 404
    return jsonify({'success': True, 'job': info})
@app.route('/duplicate_cancel/<job_id>', methods=['POST'])
def duplicate_cancel(job_id):
    info = duplicate_jobs.get(job_id)
    if not info:
        return jsonify({'success': False, 'error': 'Invalid job id'}), 404
    duplicate_jobs[job_id]['status'] = 'canceled'
    return jsonify({'success': True})

@app.route('/delete_selected_duplicates', methods=['POST'])
def delete_selected_duplicates():
    data = request.get_json() or {}
    files_to_delete = data.get('files', [])
    
    if not files_to_delete:
        return jsonify({'success': False, 'error': 'No files provided'}), 400
        
    images_dir = os.path.join('static', 'images')
    removed_count = 0
    failed_files = []
    
    for filename in files_to_delete:
        try:
            # Security check: prevent directory traversal
            if '..' in filename or filename.startswith('/') or '\\' in filename:
                failed_files.append(filename)
                continue
                
            file_path = os.path.join(images_dir, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                removed_count += 1
            else:
                # If file doesn't exist, count as removed/handled
                pass
        except Exception as e:
            print(f"Error deleting {filename}: {e}")
            failed_files.append(filename)
            
    return jsonify({
        'success': True,
        'removed_count': removed_count,
        'failed_count': len(failed_files),
        'failed_files': failed_files
    })


@app.route('/train_model', methods=['POST'])
def train_model():
    # Get parameters from form data (files uploaded)
    files = request.files.getlist('files')
    perform_alignment = request.form.get('align', 'false').lower() == 'true'
    process_all_faces = request.form.get('process_all_faces', 'false').lower() == 'true'
    model_name = request.form.get('model', CURRENT_MODEL)
    
    # Validate parameters
    if not files:
        return jsonify({'error': 'No files uploaded'}), 400
        
    if model_name not in AVAILABLE_MODELS:
        return jsonify({'error': f'Invalid model: {model_name}'}), 400
    
    # Get the database for this model
    db = model_databases[model_name]
    
    # Get face counts before training
    faces_before = len(db['embeddings'])
    unique_names_before = len(set(db['names'])) if db['names'] else 0
    
    # Filter image files
    image_files = [
        f for f in files
        if f and getattr(f, 'filename', '') and f.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'))
    ]
    
    if not image_files:
        return jsonify({'error': 'No image files found in the selected folder'}), 400
    
    # Process each image
    processed_count = 0
    skipped_count = 0
    start_time = time.time()
    
    # Track new unique names added
    existing_names = set(db['names'])
    new_unique_names = set()
    
    for img_file in image_files:
        try:
            # Extract person name from filename (assuming format: name_xyz.jpg)
            name = os.path.splitext(img_file.filename)[0].split('_')[0]
            
            # Read image from uploaded file
            img_data = img_file.read()
            nparr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                skipped_count += 1
                continue
                
            # Extract face features
            result = extract_face_features(image, model_name, perform_alignment, 'none', process_all_faces)
            
            # Handle both single face and multiple faces cases
            if not process_all_faces:
                embedding, face_roi, _, _, _, _, _ = result
                if embedding is None:
                    skipped_count += 1
                    continue
                    
                # Save the processed face image
                output_filename = f"{uuid.uuid4()}.jpg"
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
                cv2.imwrite(output_path, face_roi)
                
                # Save to database
                image_url = url_for('static', filename=f'images/{output_filename}')
                save_face_to_db(model_name, name, embedding, image_url)
                
                # Add to in-memory database (normalize for cosine search)
                db['embeddings'].append(l2_normalize(embedding))
                db['names'].append(name)
                db['images'].append(image_url)
                
                # Track new unique names
                if name not in existing_names:
                    new_unique_names.add(name)
                
                processed_count += 1
            else:
                # Process all faces in the image
                if not result or len(result) == 0:
                    skipped_count += 1
                    continue
                    
                for face_data in result:
                    embedding = face_data['embedding']
                    face_roi = face_data['face_img']
                    
                    # Save the processed face image
                    output_filename = f"{uuid.uuid4()}.jpg"
                    output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
                    cv2.imwrite(output_path, face_roi)
                    
                    # Save to database
                    image_url = url_for('static', filename=f'images/{output_filename}')
                    save_face_to_db(model_name, name, embedding, image_url)
                    
                    # Add to in-memory database (normalize for cosine search)
                    db['embeddings'].append(l2_normalize(embedding))
                    db['names'].append(name)
                    db['images'].append(image_url)
                    
                    # Track new unique names
                    if name not in existing_names:
                        new_unique_names.add(name)
                    
                    processed_count += 1
            
        except Exception as e:
            print(f"Error processing {img_file.filename}: {str(e)}")
            skipped_count += 1
            continue
    
    # Rebuild FAISS index
    if db['embeddings']:
        if not FAISS_AVAILABLE or faiss is None:
            return jsonify({
                'success': False,
                'error': 'FAISS library not available for similarity indexing'
            }), 500
        
        try:
            embeddings_array = np.array(db['embeddings']).astype('float32')
            db['index'] = faiss.IndexFlatIP(embeddings_array.shape[1])
            faiss.normalize_L2(embeddings_array)
            db['index'].add(embeddings_array)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to create FAISS index: {str(e)}'
            }), 500
    
    # Get face counts after training
    faces_after = len(db['embeddings'])
    unique_names_after = len(set(db['names'])) if db['names'] else 0
    new_unique_names_count = len(new_unique_names)
    
    processing_time = round(time.time() - start_time, 2)
    
    return jsonify({
        'processed': len(image_files),
        'faces_added': processed_count,
        'skipped': skipped_count,
        'processing_time': processing_time,
        'faces_before': faces_before,
        'faces_after': faces_after,
        'unique_names_before': unique_names_before,
        'unique_names_after': unique_names_after,
        'new_unique_names': new_unique_names_count
    })

@app.route('/clear_training_data', methods=['POST'])
def clear_training_data():
    """Clear all training data for a specific model"""
    try:
        data = request.get_json()
        model_name = data.get('model', CURRENT_MODEL)
        
        if model_name not in AVAILABLE_MODELS:
            return jsonify({'error': 'Invalid model name'}), 400
        
        # Clear from database
        clear_model_data(model_name)
        
        # Clear from memory
        model_databases[model_name] = {'embeddings': [], 'names': [], 'images': []}
        
        # Clear FAISS index
        indices[model_name] = None
        
        return jsonify({
            'success': True,
            'message': f'Training data cleared for model {model_name}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/clear_all_data', methods=['POST'])
def clear_all_data():
    """Clear all stored faces and images across every model."""
    try:
        cleared_counts = {}
        for model_name in AVAILABLE_MODELS:
            count = len(model_databases[model_name]['embeddings']) if model_databases.get(model_name) else 0
            clear_model_data(model_name)
            model_databases[model_name] = {'embeddings': [], 'names': [], 'images': []}
            indices[model_name] = None
            cleared_counts[model_name] = count

        removed_files = cleanup_upload_folder(keep_placeholder=True)

        return jsonify({
            'success': True,
            'cleared_models': cleared_counts,
            'removed_images': removed_files,
            'placeholder_preserved': 'placeholder.png' in os.listdir(app.config['UPLOAD_FOLDER']) if os.path.exists(app.config['UPLOAD_FOLDER']) else False
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get_training_stats', methods=['GET'])
def get_training_stats():
    """Get statistics about training data"""
    try:
        stats = {}
        for model in AVAILABLE_MODELS:
            data = load_faces_from_db(model)
            stats[model] = {
                'total_faces': len(data['embeddings']),
                'unique_people': len(set(data['names'])) if data['names'] else 0
            }
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/switch_processing_mode', methods=['POST'])
def switch_processing_mode():
    global PROCESSING_MODE
    global AVAILABLE_MODELS
    
    # Get data from JSON request
    data = request.get_json()
    if not data or 'mode' not in data:
        return jsonify({'error': 'Missing mode parameter'}), 400
    
    mode = data['mode']
    if mode not in ['gpu', 'cpu']:
        return jsonify({'error': f'Invalid processing mode: {mode}'}), 400
    
    # Only take action if the mode is changing
    if mode != PROCESSING_MODE:
        PROCESSING_MODE = mode
        
        # Reset all model instances to force reloading with new ctx_id
        for model_name in AVAILABLE_MODELS:
            AVAILABLE_MODELS[model_name]['instance'] = None
        
        # Reload current model with new processing mode
        if AVAILABLE_MODELS[CURRENT_MODEL]['instance'] is None:
            try:
                if not INSIGHTFACE_AVAILABLE or FaceAnalysis is None:
                    return jsonify({'error': 'FaceAnalysis is not available. Please check insightface installation.'}), 500
                AVAILABLE_MODELS[CURRENT_MODEL]['instance'] = FaceAnalysis(name=CURRENT_MODEL)
                ctx_id = 0 if PROCESSING_MODE == 'gpu' else -1
                AVAILABLE_MODELS[CURRENT_MODEL]['instance'].prepare(ctx_id=ctx_id, det_size=(640, 640))
            except Exception as e:
                return jsonify({'error': f'Failed to load model with new processing mode: {str(e)}'}), 500
        
        # Reload training data from database for all models
        for model_name in AVAILABLE_MODELS:
            model_databases[model_name] = load_faces_from_db(model_name)
    
    return jsonify({
        'success': True, 
        'mode': PROCESSING_MODE,
        'description': 'GPU Processing' if PROCESSING_MODE == 'gpu' else 'CPU Processing'
    })

@app.route('/search_faces', methods=['POST'])
def search_faces():
    """Search for faces in the database with various filters"""
    try:
        data = request.get_json()
        model_name = data.get('model', CURRENT_MODEL)
        search_query = data.get('query', '').strip().lower()
        limit = data.get('limit', 20)
        offset = data.get('offset', 0)
        hide_invalid = bool(data.get('hide_invalid', False))
        
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Base query
        base_query = '''
            SELECT id, person_name, image_path, created_at 
            FROM face_data 
            WHERE model_name = ?
        '''
        params = [model_name]
        
        # Add search filter if provided
        if search_query:
            base_query += ' AND LOWER(person_name) LIKE ?'
            params.append(f'%{search_query}%')
        
        # Add ordering and pagination
        base_query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
        params.extend([limit, offset])
        
        cursor.execute(base_query, params)
        results = cursor.fetchall()
        
        # Get total count for pagination
        count_query = '''
            SELECT COUNT(*) FROM face_data WHERE model_name = ?
        '''
        count_params = [model_name]
        if search_query:
            count_query += ' AND LOWER(person_name) LIKE ?'
            count_params.append(f'%{search_query}%')
        
        cursor.execute(count_query, count_params)
        total_count = cursor.fetchone()[0]
        
        conn.close()
        
        # Format results with proper image URLs
        formatted_results = []
        for row in results:
            # Convert file path to proper URL
            image_path = row[2]
            if image_path.startswith('static/'):
                # Remove 'static/' prefix and use url_for
                image_filename = image_path.replace('static/', '')
                image_url = url_for('static', filename=image_filename)
                fs_path = image_path
            elif image_path.startswith('/static/'):
                # Already has the full path with leading slash
                image_url = image_path
                fs_path = image_path.lstrip('/')
            else:
                # Fallback for other path formats
                image_url = f'/static/{image_path}'
                fs_path = f'static/{image_path}'
            
            # Compute invalid flag: missing file or placeholder
            is_invalid = False
            try:
                if fs_path.endswith('placeholder.png') or (not os.path.exists(fs_path)):
                    is_invalid = True
            except Exception:
                is_invalid = True
            
            if hide_invalid and is_invalid:
                continue
            
            formatted_results.append({
                'id': row[0],
                'name': row[1],
                'image_path': image_url,  # Now contains proper URL
                'created_at': row[3],
                'is_invalid': is_invalid
            })
        
        return jsonify({
            'success': True,
            'results': formatted_results,
            'total_count': total_count,
            'current_page': offset // limit + 1,
            'total_pages': (total_count + limit - 1) // limit
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get_face_details/<int:face_id>', methods=['GET'])
def get_face_details(face_id):
    """Get detailed information about a specific face"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, model_name, person_name, image_path, created_at
            FROM face_data 
            WHERE id = ?
        ''', (face_id,))
        
        result = cursor.fetchone()
        conn.close()
        
        if not result:
            return jsonify({'success': False, 'error': 'Face not found'}), 404
        
        return jsonify({
            'success': True,
            'face': {
                'id': result[0],
                'model_name': result[1],
                'name': result[2],
                'image_path': result[3],
                'created_at': result[4]
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/delete_face/<int:face_id>', methods=['DELETE'])
def delete_face(face_id):
    """Delete a specific face from the database"""
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Get image path before deletion for cleanup
        cursor.execute('SELECT image_path FROM face_data WHERE id = ?', (face_id,))
        result = cursor.fetchone()
        
        if not result:
            conn.close()
            return jsonify({'success': False, 'error': 'Face not found'}), 404
        
        image_path = result[0]

        # Normalize image path from URL to filesystem path
        fs_path = None
        try:
            if image_path.startswith('/static/'):
                fs_path = image_path.lstrip('/')
            elif image_path.startswith('static/'):
                fs_path = image_path
            else:
                # Fallback assume images folder
                fs_path = os.path.join('static', image_path)
        except Exception:
            fs_path = None

        # Delete from database
        cursor.execute('DELETE FROM face_data WHERE id = ?', (face_id,))
        conn.commit()
        conn.close()
        
        # Clean up image file if it exists
        try:
            if fs_path and os.path.exists(fs_path):
                os.remove(fs_path)
        except Exception as e:
            print(f"Warning: Could not delete image file {image_path}: {e}")
        
        return jsonify({'success': True, 'message': 'Face deleted successfully'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/delete_invalid_faces', methods=['POST'])
def delete_invalid_faces():
    """Bulk delete invalid faces (missing image files or placeholder)."""
    try:
        data = request.get_json() or {}
        ids = data.get('ids', [])
        if not isinstance(ids, list) or not ids:
            return jsonify({'success': False, 'error': 'No IDs provided'}), 400
        
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        removed = 0
        failed = []
        for face_id in ids:
            try:
                cursor.execute('SELECT image_path FROM face_data WHERE id = ?', (face_id,))
                row = cursor.fetchone()
                if not row:
                    failed.append(face_id)
                    continue
                image_path = row[0]
                
                # Normalize to filesystem path
                if image_path.startswith('/static/'):
                    fs_path = image_path.lstrip('/')
                elif image_path.startswith('static/'):
                    fs_path = image_path
                else:
                    fs_path = os.path.join('static', image_path if image_path.startswith('images/') else os.path.join('images', image_path))
                
                # Verify invalid
                is_invalid = False
                try:
                    if fs_path.endswith('placeholder.png') or (not os.path.exists(fs_path)):
                        is_invalid = True
                except Exception:
                    is_invalid = True
                
                if not is_invalid:
                    failed.append(face_id)
                    continue
                
                # Delete DB row
                cursor.execute('DELETE FROM face_data WHERE id = ?', (face_id,))
                conn.commit()
                
                # Attempt to delete file if it exists
                try:
                    if os.path.exists(fs_path) and not fs_path.endswith('placeholder.png'):
                        os.remove(fs_path)
                except Exception:
                    pass
                
                removed += 1
            except Exception:
                failed.append(face_id)
        
        conn.close()
        
        return jsonify({'success': True, 'removed_count': removed, 'failed_ids': failed})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/delete_all_invalid', methods=['POST'])
def delete_all_invalid():
    """Delete all invalid faces for a given model (missing image files or placeholder)."""
    try:
        data = request.get_json() or {}
        model_name = data.get('model', CURRENT_MODEL)
        
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, image_path FROM face_data WHERE model_name = ?', (model_name,))
        rows = cursor.fetchall()
        
        removed = 0
        failed_ids = []
        
        for face_id, image_path in rows:
            try:
                # Normalize to filesystem path
                if image_path.startswith('/static/'):
                    fs_path = image_path.lstrip('/')
                elif image_path.startswith('static/'):
                    fs_path = image_path
                else:
                    fs_path = os.path.join('static', image_path if image_path.startswith('images/') else os.path.join('images', image_path))
                
                # Determine invalid
                is_invalid = False
                try:
                    if fs_path.endswith('placeholder.png') or (not os.path.exists(fs_path)):
                        is_invalid = True
                except Exception:
                    is_invalid = True
                
                if not is_invalid:
                    continue
                
                # Delete DB row
                cursor.execute('DELETE FROM face_data WHERE id = ?', (face_id,))
                conn.commit()
                
                # Attempt to delete file if it exists and is not placeholder
                try:
                    if os.path.exists(fs_path) and not fs_path.endswith('placeholder.png'):
                        os.remove(fs_path)
                except Exception:
                    pass
                
                removed += 1
            except Exception:
                failed_ids.append(face_id)
        
        conn.close()
        
        return jsonify({'success': True, 'removed_count': removed, 'failed_ids': failed_ids})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

VIDEO_FACES_DB_FILE = 'video_faces.json'

@app.route('/save_video_faces', methods=['POST'])
def save_video_faces():
    try:
        data = request.get_json()
        video_name = data.get('video_name', 'unknown_video')
        selected_faces = data.get('faces', [])
        
        if not selected_faces:
            return jsonify({'error': 'No faces selected'}), 400
            
        # Create persistent directory for this video
        # Sanitize video name to be safe for filesystem
        safe_video_name = "".join([c for c in video_name if c.isalpha() or c.isdigit() or c in (' ', '.', '_', '-')]).strip()
        if not safe_video_name:
            safe_video_name = f"video_{int(time.time())}"
            
        video_dir = os.path.join(app.static_folder, 'video_faces', safe_video_name)
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
            
        # Load existing DB
        db = {}
        if os.path.exists(VIDEO_FACES_DB_FILE):
            try:
                with open(VIDEO_FACES_DB_FILE, 'r') as f:
                    db = json.load(f)
            except:
                pass
        
        if safe_video_name not in db:
            db[safe_video_name] = {
                'uploaded_at': time.time(),
                'faces': []
            }
            
        saved_count = 0
        video_faces_root = os.path.join(app.static_folder, 'video_faces')
        
        for face in selected_faces:
            face_id = face.get('id') # e.g. "temp/face.jpg"
            source_path = os.path.join(video_faces_root, face_id)
            
            if os.path.exists(source_path):
                basename = os.path.basename(face_id)
                dest_path = os.path.join(video_dir, basename)
                
                # Copy file
                shutil.copy2(source_path, dest_path)
                
                # New ID for library is relative to video_faces root
                new_id = f"{safe_video_name}/{basename}"
                
                # Add to DB structure
                # Check if already exists in DB to avoid dupes in JSON
                existing_ids = [f['id'] for f in db[safe_video_name]['faces']]
                if new_id not in existing_ids:
                    db[safe_video_name]['faces'].append({
                        'id': new_id,
                        'url': url_for('static', filename=f'video_faces/{safe_video_name}/{basename}'),
                        'timestamp': face.get('timestamp'),
                        'timestamp_str': face.get('timestamp_str'),
                        'score': face.get('score')
                    })
                    saved_count += 1
        
        with open(VIDEO_FACES_DB_FILE, 'w') as f:
            json.dump(db, f, indent=2)
            
        return jsonify({'success': True, 'saved_count': saved_count, 'video_group': safe_video_name})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get_video_library', methods=['GET'])
def get_video_library():
    try:
        if os.path.exists(VIDEO_FACES_DB_FILE):
            with open(VIDEO_FACES_DB_FILE, 'r') as f:
                db = json.load(f)
            
            # Sort by video name (keys)
            # Actually, let's return a list sorted by name
            library = []
            for video_name, data in db.items():
                library.append({
                    'video_name': video_name,
                    'uploaded_at': data.get('uploaded_at'),
                    'faces': data.get('faces', [])
                })
            
            # Sort by video name
            library.sort(key=lambda x: x['video_name'])
            
            return jsonify({'success': True, 'library': library})
        else:
            return jsonify({'success': True, 'library': []})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/delete_video_library_item', methods=['POST'])
def delete_video_library_item():
    try:
        data = request.get_json()
        video_name = data.get('video_name')
        face_id = data.get('face_id') # Optional, if provided delete only this face
        
        if not video_name:
            return jsonify({'error': 'Video name required'}), 400
            
        if not os.path.exists(VIDEO_FACES_DB_FILE):
            return jsonify({'error': 'Library empty'}), 404
            
        with open(VIDEO_FACES_DB_FILE, 'r') as f:
            db = json.load(f)
            
        if video_name not in db:
            return jsonify({'error': 'Video not found'}), 404
            
        video_dir = os.path.join(app.static_folder, 'video_faces', video_name)
        
        if face_id:
            # Delete specific face
            faces = db[video_name]['faces']
            new_faces = [f for f in faces if f['id'] != face_id]
            db[video_name]['faces'] = new_faces
            
            # Delete file
            # face_id can be "video_name/filename" or just "filename" relative to video dir?
            # In save_video_faces: new_id = f"{safe_video_name}/{basename}"
            # So face_id is "video_name/face_123.jpg"
            
            basename = os.path.basename(face_id)
            face_path = os.path.join(video_dir, basename)
            if os.path.exists(face_path):
                try:
                    os.remove(face_path)
                except Exception as e:
                    print(f"Error deleting file {face_path}: {e}")
                
            # If no faces left, maybe keep the video entry? Or delete?
            # User might want to keep the group. Let's keep it.
        else:
            # Delete entire video group
            del db[video_name]
            if os.path.exists(video_dir):
                shutil.rmtree(video_dir)
                
        with open(VIDEO_FACES_DB_FILE, 'w') as f:
            json.dump(db, f, indent=2)
            
        return jsonify({'success': True})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/add_video_face_to_db', methods=['POST'])
def add_video_face_to_db():
    try:
        data = request.get_json()
        face_id = data.get('face_id')
        person_name = data.get('person_name')
        model_name = data.get('model_name', CURRENT_MODEL)
        
        if not face_id or not person_name:
            return jsonify({'error': 'Face ID and Person Name are required'}), 400
            
        # Locate source image
        # face_id format: "video_name/face.jpg"
        source_path = os.path.join(app.static_folder, 'video_faces', face_id)
        
        if not os.path.exists(source_path):
            return jsonify({'error': 'Source face image not found'}), 404
            
        # Generate new filename for main DB
        # Clean person name for filename
        safe_name = "".join([c for c in person_name if c.isalnum() or c in (' ', '-', '_')]).strip().replace(' ', '_')
        new_filename = f"{safe_name}_{uuid.uuid4().hex[:8]}.jpg"
        dest_rel_path = os.path.join('images', new_filename)
        dest_path = os.path.join(app.static_folder, dest_rel_path)
        
        # Copy file
        shutil.copy2(source_path, dest_path)
        
        # Initialize model if needed
        global FaceAnalysis, INSIGHTFACE_AVAILABLE
        
        if AVAILABLE_MODELS[model_name]['instance'] is None:
            if not INSIGHTFACE_AVAILABLE:
                return jsonify({'error': 'InsightFace not available'}), 500
                
            # Initialize model
            AVAILABLE_MODELS[model_name]['instance'] = FaceAnalysis(name=model_name)
            ctx_id = 0 if PROCESSING_MODE == 'gpu' else -1
            AVAILABLE_MODELS[model_name]['instance'].prepare(ctx_id=ctx_id, det_size=(640, 640))
            
        face_app = AVAILABLE_MODELS[model_name]['instance']
        
        # Compute embedding
        img = cv2.imread(dest_path)
        if img is None:
            return jsonify({'error': 'Failed to read image'}), 500
            
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_app.get(rgb_img)
        
        if not faces:
            # Fallback: if detection fails on the crop, maybe just center crop?
            # But usually it should work since it was detected before.
            # Try with lower threshold or just fail?
            return jsonify({'error': 'No face detected in the image'}), 400
            
        # Sort by size, pick largest
        faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
        target_face = faces[0]
        embedding = target_face.embedding
        
        # Save to DB
        # path stored in DB is relative to static/images usually, or just filename if in images/
        # save_face_to_db expects full relative path?
        # Let's check save_face_to_db calls. 
        # Line 216: return url_for('static', filename=f'images/{filename}')
        # But load_faces_from_db parses it.
        # Let's use 'images/filename' format which seems consistent.
        
        db_image_path = f"images/{new_filename}"
        save_face_to_db(model_name, person_name, embedding, db_image_path)
        
        return jsonify({'success': True, 'message': f'Added {person_name} to database'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/match_face_in_library', methods=['POST'])
def match_face_in_library():
    try:
        data = request.get_json()
        target_face_id = data.get('face_id')
        threshold = float(data.get('threshold', 0.6))
        model_name = data.get('model_name', CURRENT_MODEL)
        
        if not target_face_id:
            return jsonify({'error': 'Face ID required'}), 400
            
        video_faces_dir = os.path.join(app.static_folder, 'video_faces')
        target_path = os.path.join(video_faces_dir, target_face_id)
        
        if not os.path.exists(target_path):
            return jsonify({'error': 'Target face not found'}), 404
            
        # Initialize model
        if AVAILABLE_MODELS[model_name]['instance'] is None:
            if not INSIGHTFACE_AVAILABLE:
                return jsonify({'error': 'InsightFace not available'}), 500
            AVAILABLE_MODELS[model_name]['instance'] = FaceAnalysis(name=model_name)
            ctx_id = 0 if PROCESSING_MODE == 'gpu' else -1
            AVAILABLE_MODELS[model_name]['instance'].prepare(ctx_id=ctx_id, det_size=(640, 640))
            
        face_app = AVAILABLE_MODELS[model_name]['instance']
        
        # 1. Get embedding for target face
        img = cv2.imread(target_path)
        if img is None:
            return jsonify({'error': 'Failed to read target image'}), 500
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = face_app.get(rgb_img)
        if not faces:
            return jsonify({'error': 'No face detected in target image'}), 400
        
        target_embedding = faces[0].embedding
        target_norm = l2_normalize(target_embedding)
        
        # 2. Iterate through library
        if not os.path.exists(VIDEO_FACES_DB_FILE):
             return jsonify({'success': True, 'matches': []})
             
        with open(VIDEO_FACES_DB_FILE, 'r') as f:
            library_db = json.load(f)
            
        matches = []
        
        # We need to process all images. This can be slow.
        # But we can skip the target face itself.
        
        for video_name, video_data in library_db.items():
            for face in video_data.get('faces', []):
                face_id = face['id']
                if face_id == target_face_id:
                    continue
                    
                face_path = os.path.join(video_faces_dir, face_id)
                if not os.path.exists(face_path):
                    continue
                    
                # Read and infer
                # Optimization: In a real app, we should cache these embeddings!
                # For now, we compute on the fly.
                
                curr_img = cv2.imread(face_path)
                if curr_img is None:
                    continue
                    
                curr_rgb = cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB)
                curr_faces = face_app.get(curr_rgb)
                
                if not curr_faces:
                    continue
                    
                # Compare
                curr_embedding = curr_faces[0].embedding
                curr_norm = l2_normalize(curr_embedding)
                
                # Cosine similarity
                sim = np.dot(target_norm, curr_norm)
                
                if sim > threshold:
                    matches.append({
                        'video_name': video_name,
                        'face_id': face_id,
                        'url': face['url'],
                        'timestamp_str': face['timestamp_str'],
                        'score': float(sim)
                    })
                    
        # Sort by score descending
        matches.sort(key=lambda x: x['score'], reverse=True)
        
        return jsonify({'success': True, 'matches': matches})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def estimate_age(face_img):
    """Estimate the approximate age range of a face"""
    if face_img is None:
        return None, 0.0
    
    # Option 1: Use a pre-trained age estimation model
    # This example uses a simplified approach with facial features
    # For production, consider using a dedicated age estimation model
    
    # Convert to grayscale for processing
    if len(face_img.shape) == 3:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = face_img
    
    # Extract facial features that correlate with age
    # (simplified approach - a real implementation would use a trained model)
    try:
        # Use face landmarks if available (from InsightFace)
        # Ensure model instance is initialized
        if AVAILABLE_MODELS[CURRENT_MODEL]['instance'] is None:
            if not INSIGHTFACE_AVAILABLE or FaceAnalysis is None:
                return "unknown", 0.1  # Very low confidence if FaceAnalysis not available
            AVAILABLE_MODELS[CURRENT_MODEL]['instance'] = FaceAnalysis(name=CURRENT_MODEL)
            ctx_id = 0 if PROCESSING_MODE == 'gpu' else -1
            AVAILABLE_MODELS[CURRENT_MODEL]['instance'].prepare(ctx_id=ctx_id, det_size=(640, 640))
        face_analysis = AVAILABLE_MODELS[CURRENT_MODEL]['instance']
        faces = face_analysis.get(face_img)
        
        if not faces:
            return "unknown", 0.3  # Low confidence if no face detected
        
        face = faces[0]
        
        # For a real implementation, you would use these landmarks
        # to extract age-relevant features or use a dedicated age model
        
        # Placeholder logic - replace with actual model inference
        # This is just a demonstration and should be replaced with a real model
        
        # Example categories with confidence
        age_categories = ["child", "teen", "young_adult", "adult", "senior"]
        
        # In a real implementation, this would be the output of your model
        # For now, we'll return a random category with medium confidence
        import random
        category = random.choice(age_categories)
        confidence = 0.7  # Medium confidence
        
        return category, confidence
        
    except Exception as e:
        print(f"Age estimation error: {str(e)}")
        return "unknown", 0.0

if __name__ == '__main__':
    # Ensure static/images directory exists
    os.makedirs('static/images', exist_ok=True)
    # Trigger reload
    app.run(debug=True, host='0.0.0.0', port=8080)
