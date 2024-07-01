import cv2
from ultralytics import RTDETR
import matplotlib.pyplot as plt
import subprocess
import os
import sys
import shutil
from barchart import drawchart

model = RTDETR("rt-detr/weights.pt")

class_names = ['can', 'carton', 'p-bag', 'p-bottle', 'p-con', 'styrofoam', 'tire']

def extract_frames(input_video, output_folder, frame_pattern):
    os.makedirs(output_folder, exist_ok=True)
    extract_frames_command = [
        'ffmpeg', '-i', input_video, '-vf', 'select=not(mod(n\\,1))', '-vsync', 'vfr', f'{output_folder}/{frame_pattern}'
    ]
    subprocess.run(extract_frames_command, check=True)

def combine_frames(input_pattern, output_video, framerate=30):
    combine_frames_command = [
        'ffmpeg', '-framerate', str(framerate), '-i', input_pattern, '-c:v', 'libx264', '-r', str(framerate), '-pix_fmt', 'yuv420p', output_video
    ]
    subprocess.run(combine_frames_command, check=True)

input_video = sys.argv[1]
output_video = sys.argv[2]
frames_folder = 'frames'
frame_pattern = 'frame%03d.png'
cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)

extract_frames(input_video, frames_folder, frame_pattern)

tracking_data = []
frame_files = sorted(os.listdir(frames_folder))
for framecount, frame_file in enumerate(frame_files):
    frame_path = os.path.join(frames_folder, frame_file)
    frame = cv2.imread(frame_path)

    if frame is not None:
        print(f"Processing frame {framecount}...")

        results = model.track(frame, persist=True)
        
        frame_data = []
        
        if hasattr(results[0], 'boxes'):
            for obj in results[0].boxes:
                class_id = int(obj.cls)
                class_name = class_names[class_id]
                track_id = int(obj.id)
                confidence = float(obj.conf)
                bbox = obj.xyxy.tolist()
                frame_data.append({
                    'frame_idx': framecount,
                    'class_id': class_id,
                    'track_id': track_id,
                    'confidence': confidence,
                    'bbox': bbox
                })
                
                x1, y1, x2, y2 = map(int, bbox[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"ID: {track_id} Class: {class_name} Conf: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        tracking_data.append(frame_data)
        
        with open("tracking_info.txt", "a") as f:
            for obj in frame_data:
                f.write(f"Frame: {obj['frame_idx']}, Class ID: {obj['class_id']}, Track ID: {obj['track_id']}, Confidence: {obj['confidence']}, BBox: {obj['bbox']}\n")

        print(f"Tracking information for frame {framecount} saved to tracking_info.txt")
        
        cv2.imwrite(frame_path, frame)

    else:
        print(f"Failed to read frame {framecount}")
        break

combine_frames(f'{frames_folder}/{frame_pattern}', output_video)

drawchart(fps, "tracking_info.txt", input_video)

print("Processing complete.")

shutil.rmtree(frames_folder)

os.remove('tracking_info.txt')