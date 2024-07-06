import cv2
from ultralytics import RTDETR
import matplotlib.pyplot as plt
import subprocess
import os
import sys
import random
import shutil
import numpy as np
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

def create_blank_video_with_objects(input_video, objects_list, output_video, fps, frame_size):
    width, height = frame_size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    
    overlay_images = {}
    for class_name in class_names:
        img_path = f'../img/debris/{class_name}.png'
        if os.path.exists(img_path):
            overlay_images[class_name] = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    
    blank_frame = np.zeros((height, width, 3), dtype=np.uint8)  # Create a blank black frame
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    seen_ids = set()
    object_positions = []
    
    for i in range(frame_count):
        frame = blank_frame.copy()
        
        # Overlay all previous objects
        for pos in object_positions:
            frame = overlay_object(frame, pos['overlay_img'], pos['position'])
        
        if i in objects_list:
            for obj in objects_list[i]:
                track_id = obj['track_id']
                class_name = obj['class_name']
                if track_id not in seen_ids and class_name in overlay_images:
                    seen_ids.add(track_id)
                    overlay_img = overlay_images[class_name]
                    h, w = overlay_img.shape[:2]
                    if h > height // 2 or w > width // 2:
                        scaling_factor = min((height // 2) / h, (width // 2) / w)
                        overlay_img = cv2.resize(overlay_img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
                        h, w = overlay_img.shape[:2]
                    x = random.randint(0, width - w)
                    y = random.randint(0, height - h)
                    object_positions.append({'overlay_img': overlay_img, 'position': (x, y)})
                    frame = overlay_object(frame, overlay_img, (x, y))
        
        out.write(frame)
    
    out.release()

def overlay_object(frame, overlay_img, position):
    x, y = position
    h, w = overlay_img.shape[:2]
    
    if overlay_img.shape[2] == 4:
        alpha_overlay = overlay_img[:, :, 3] / 255.0
        alpha_background = 1.0 - alpha_overlay
    
        for c in range(0, 3):
            frame[y:y+h, x:x+w, c] = (alpha_overlay * overlay_img[:, :, c] + alpha_background * frame[y:y+h, x:x+w, c])
    else: 
        frame[y:y+h, x:x+w] = overlay_img
    
    return frame

input_video = sys.argv[1]
output_video = sys.argv[2]
frames_folder = 'frames'
frame_pattern = 'frame%03d.png'
cap = cv2.VideoCapture(input_video)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

extract_frames(input_video, frames_folder, frame_pattern)

tracking_data = []
objects_list = {}
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
                    'class_name': class_name,
                    'confidence': confidence,
                    'bbox': bbox
                })
                
                x1, y1, x2, y2 = map(int, bbox[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"ID: {track_id} Class: {class_name} Conf: {confidence:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        tracking_data.append(frame_data)

        if frame_data:
            objects_list[framecount] = frame_data
        
        with open("tracking_info.txt", "a") as f:
            for obj in frame_data:
                f.write(f"Frame: {obj['frame_idx']}, Class ID: {obj['class_id']}, Track ID: {obj['track_id']}, Confidence: {obj['confidence']}, BBox: {obj['bbox']}\n")

        print(f"Tracking information for frame {framecount} saved to tracking_info.txt")
        
        cv2.imwrite(frame_path, frame)

    else:
        print(f"Failed to read frame {framecount}")
        break

drawchart(fps, "tracking_info.txt", input_video)

create_blank_video_with_objects(input_video, objects_list, "debris.mp4", fps, frame_size)

print("Processing complete.")

shutil.rmtree(frames_folder)

os.remove('tracking_info.txt')