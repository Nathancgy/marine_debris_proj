import cv2
from ultralytics import RTDETR
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from barchart import drawchart
# Load the RTDETR model
model = RTDETR("../rt-detr/weights.pt")

# Open the video file
video_path = "k.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Frames per second: {fps}")

framecount = 0
tracking_data = []

while cap.isOpened():
    success, frame = cap.read()

    if success:
        print(f"Processing frame {framecount}...")
        
        results = model.track(frame, persist=True)
        
        frame_data = []
        
        if hasattr(results[0], 'boxes'):
            for obj in results[0].boxes:
                class_id = int(obj.cls)
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
        
        tracking_data.append(frame_data)
        
        with open("tracking_info.txt", "a") as f:
            for obj in frame_data:
                f.write(f"Frame: {obj['frame_idx']}, Class ID: {obj['class_id']}, Track ID: {obj['track_id']}, Confidence: {obj['confidence']}, BBox: {obj['bbox']}\n")

        print(f"Tracking information for frame {framecount} saved to tracking_info.txt")
        framecount += 1

    else:
        print('End of video')
        break

drawchart(fps, "tracking_info.txt", video_path)