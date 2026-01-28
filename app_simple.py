from flask import Flask, request, jsonify, render_template, url_for, send_from_directory
import os
import uuid
from werkzeug.utils import secure_filename
import cv2
import numpy as np

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = os.path.join('static', 'images')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

@app.route('/simple')
def simple():
    # Get list of images in upload folder
    images = []
    for filename in os.listdir(UPLOAD_FOLDER):
        if allowed_file(filename):
            images.append(url_for('static', filename=f'images/{filename}'))
    return render_template('simple_index.html', images=images)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    # Get list of images in upload folder
    images = []
    for filename in os.listdir(UPLOAD_FOLDER):
        if allowed_file(filename):
            images.append(url_for('static', filename=f'images/{filename}'))
    return render_template('index.html', images=images)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Create a unique filename
        filename = secure_filename(str(uuid.uuid4()) + os.path.splitext(file.filename)[1])
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Ensure upload folder exists
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        # Save the file
        file.save(filepath)
        
        # Return the URL of the saved file
        return jsonify({
            'success': True,
            'image_url': url_for('static', filename=f'images/{filename}')
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/images')
def list_images():
    images = []
    for filename in os.listdir(UPLOAD_FOLDER):
        if allowed_file(filename):
            images.append({
                'filename': filename,
                'url': url_for('static', filename=f'images/{filename}')
            })
    return jsonify({'images': images})

@app.route('/delete/<filename>', methods=['DELETE'])
def delete_image(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        return jsonify({'success': True})
    return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    # Create upload folder if it doesn't exist
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)