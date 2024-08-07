# app.py
from flask import Flask, request, render_template, redirect, url_for, flash
import os
import subprocess

app = Flask(__name__)
app.secret_key = 'supersecretkey'

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'static'
COMBINED_VIDEO_FOLDER = 'combined'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER
app.config['COMBINED_VIDEO_FOLDER'] = COMBINED_VIDEO_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        processed_video_path = os.path.join(app.config['PROCESSED_FOLDER'], 'processed_' + file.filename)
        try:
            subprocess.run(['python3', 'src/ObjectCount.py', file_path, processed_video_path], check=True)
        except subprocess.CalledProcessError as e:
            flash(f'An error occurred while processing the video: {e}')
            return redirect(request.url)    

        return redirect(url_for('processed_file', filename='processed_' + file.filename))
    return redirect(request.url)

@app.route('/processed/<filename>')
def processed_file(filename):
    return render_template('processed.html', filename=filename)

@app.route('/contribute', methods=['GET', 'POST'])
def contribute():
    if request.method == 'POST':
        name = request.form.get('name', 'you')

        video_file = request.files.get('video')
        image_files = request.files.getlist('images')
        label_file = request.files.get('labels')

        if not video_file and not image_files and not label_file:
            return render_template('contribute.html', error='Please upload at least one file.')

        if video_file and video_file.filename != '':
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'contributed_videos', video_file.filename)
            os.makedirs(os.path.dirname(video_path), exist_ok=True)
            video_file.save(video_path)

        if image_files and label_file and label_file.filename != '':
            for image_file in image_files:
                if image_file.filename != '':
                    image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'contributed_images', image_file.filename)
                    os.makedirs(os.path.dirname(image_path), exist_ok=True)
                    image_file.save(image_path)

            label_path = os.path.join(app.config['UPLOAD_FOLDER'], 'contributed_labels', label_file.filename)
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            label_file.save(label_path)

        with open('contributors.txt', 'a') as file:
            file.write(f'{name}\n')

        return render_template('contribute_success.html', name=name)
    
    return render_template('contribute.html')

if __name__ == "__main__":
    app.run(debug=True)
