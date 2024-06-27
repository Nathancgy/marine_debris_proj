from flask import Flask, request, render_template, redirect, url_for, jsonify
from ultralytics import YOLO
import cv2
import os
import subprocess
import shutil

app = Flask(__name__)
model = YOLO("weights.pt")

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'static'
FRAMES_FOLDER = 'frames'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['FRAMES_FOLDER'] = FRAMES_FOLDER

ffmpeg_progress = 0
yolo_progress = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global ffmpeg_progress, yolo_progress
    ffmpeg_progress = 0
    yolo_progress = 0
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        process_video(file_path)
        return redirect(url_for('processed_file', filename=file.filename))
    return redirect(request.url)

def extract_frames(input_video, output_folder, frame_pattern):
    global ffmpeg_progress
    os.makedirs(output_folder, exist_ok=True)
    extract_frames_command = [
        'ffmpeg', '-i', input_video, '-vf', 'select=not(mod(n\\,1))', '-vsync', 'vfr', f'{output_folder}/{frame_pattern}'
    ]
    subprocess.run(extract_frames_command, check=True)
    ffmpeg_progress = 100

def combine_frames(input_pattern, output_video, framerate=30):
    combine_frames_command = [
        'ffmpeg', '-framerate', str(framerate), '-i', input_pattern, '-c:v', 'libx264', '-r', str(framerate), '-pix_fmt', 'yuv420p', output_video
    ]
    subprocess.run(combine_frames_command, check=True)

def delete_frames(folder):
    shutil.rmtree(folder)

def process_video(file_path):
    global yolo_progress
    frames_folder = app.config['FRAMES_FOLDER']
    frame_pattern = 'frame%03d.png'
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], 'processed_' + os.path.basename(file_path))

    # Extract frames from the video
    extract_frames(file_path, frames_folder, frame_pattern)

    # Process each frame using YOLO
    frames = sorted(os.listdir(frames_folder))
    total_frames = len(frames)
    
    for i, frame in enumerate(frames):
        frame_path = os.path.join(frames_folder, frame)
        img = cv2.imread(frame_path)
        results = model.predict(source=img)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                label = result.names[int(box.cls)]
                confidence = box.conf.item()  # Convert tensor to float
                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(img, f'{label} {confidence:.2f}', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.imwrite(frame_path, img)
        yolo_progress = int((i / total_frames) * 100)
        print(f'Processing frame {i+1}/{total_frames} ({yolo_progress}%)')

    # Combine frames back into a video
    combine_frames(f'{frames_folder}/{frame_pattern}', output_path)

    # Delete the frames
    delete_frames(frames_folder)

@app.route('/ffmpeg_progress')
def get_ffmpeg_progress():
    global ffmpeg_progress
    return jsonify(progress=ffmpeg_progress)

@app.route('/yolo_progress')
def get_yolo_progress():
    global yolo_progress
    return jsonify(progress=yolo_progress)

@app.route('/processed/<filename>')
def processed_file(filename):
    return render_template('processed.html', filename='processed_' + filename)

if __name__ == "__main__":
    app.run(debug=True)
