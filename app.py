from flask import Flask, request, render_template, redirect, url_for, jsonify
from ultralytics import YOLO
import cv2
import os

app = Flask(__name__)
model = YOLO("weights.pt")

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

progress = 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global progress
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

def process_video(file_path):
    global progress
    output_path = os.path.join(app.config['PROCESSED_FOLDER'], 'processed_' + os.path.basename(file_path))
    cap = cv2.VideoCapture(file_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            results = model.predict(source=frame)
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    label = result.names[int(box.cls)]
                    confidence = box.conf.item()  # Convert tensor to float
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                    cv2.putText(frame, f'{label} {confidence:.2f}', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            out.write(frame)
            frame_number += 1
            progress = int((frame_number / total_frames) * 100)
            print(f'Processing frame {frame_number}/{total_frames} ({progress}%)')
        else:
            break
    cap.release()
    out.release()

@app.route('/<filename>')
def processed_file(filename):
    return render_template('processed.html', filename='processed_' + filename)

@app.route('/progress')
def get_progress():
    global progress
    return jsonify(progress=progress)

if __name__ == "__main__":
    app.run(debug=True)

