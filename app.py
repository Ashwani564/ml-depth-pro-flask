# app.py
import os
from flask import Flask, request, render_template, redirect, url_for, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import threading
import uuid
import time

# Import our processing functions
from processing import process_image, process_video

app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = os.path.join('static', 'uploads')
PROCESSED_FOLDER = os.path.join('static', 'processed')
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

tasks = {}

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        proc_type = request.form.get('type')
        allowed_extensions = ALLOWED_IMAGE_EXTENSIONS if proc_type == 'image' else ALLOWED_VIDEO_EXTENSIONS
        
        if file and allowed_file(file.filename, allowed_extensions):
            
            # --- START: MODIFIED FILENAME LOGIC ---

            # 1. Get the original, secure filename from the upload
            original_filename = secure_filename(file.filename)
            
            # 2. Split it into the base name and the extension
            #    e.g., "my_photo.jpg" -> ("my_photo", ".jpg")
            base_name, extension = os.path.splitext(original_filename)
            
            # 3. Create the desired output filename by adding "_processed"
            #    e.g., "my_photo_processed.jpg"
            output_filename = f"{base_name}_processed{extension}"

            # 4. Create a unique task ID and use it to make the *input* filename unique
            #    This prevents two users uploading "image.jpg" from overwriting each other.
            task_id = str(uuid.uuid4())
            input_filename = f"{task_id}_{original_filename}"
            
            # --- END: MODIFIED FILENAME LOGIC ---

            input_path = os.path.join(app.config['UPLOAD_FOLDER'], input_filename)
            file.save(input_path)
            
            # The output path is now based on our newly constructed filename
            output_path = os.path.join(app.config['PROCESSED_FOLDER'], output_filename)
            
            # Store the correct filenames in the task dictionary
            tasks[task_id] = {
                'status': 'queued', 
                'progress': 0, 
                'input_file': input_filename,       # The unique input file
                'output_file': output_filename,     # The desired output file
                'type': proc_type,
                'start_time': time.time()
            }

            # Start the background processing thread
            if proc_type == 'image':
                thread = threading.Thread(target=process_image, args=(input_path, output_path, task_id, tasks))
            else: # video
                thread = threading.Thread(target=process_video, args=(input_path, output_path, task_id, tasks))

            thread.start()
            
            return redirect(url_for('processing_status', task_id=task_id))

    upload_type = request.args.get('type', 'image')
    return render_template('upload.html', type=upload_type)

@app.route('/processing/<task_id>')
def processing_status(task_id):
    if task_id not in tasks:
        return "Task not found", 404
    return render_template('result.html', task_id=task_id)

@app.route('/status/<task_id>')
def status(task_id):
    task = tasks.get(task_id)
    if not task:
        return jsonify({'status': 'not_found'}), 404
    return jsonify(task)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    # It's good practice to use threaded=True for development with background tasks
    app.run(debug=True, threaded=True)