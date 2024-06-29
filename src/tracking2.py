from ultralytics import RTDETR

model = RTDETR("../rt-detr/weights.pt")

results = model.track(source="k.mp4", show=True, tracker="botsort.yaml", save=True, conf=0.1)

tracking_data = []

for frame_idx, result in enumerate(results):
    frame_data = []
    if hasattr(result, 'boxes'):
        for obj in result.boxes:
            class_id = int(obj.cls)  
            track_id = int(obj.id)
            confidence = float(obj.conf) 
            bbox = obj.xyxy.tolist() 
            frame_data.append({
                'frame_idx': frame_idx,
                'class_id': class_id,
                'track_id': track_id,
                'confidence': confidence,
                'bbox': bbox
            })
    tracking_data.append(frame_data)

with open("tracking_info.txt", "w") as f:
    for frame_data in tracking_data:
        for obj in frame_data:
            f.write(f"Frame: {obj['frame_idx']}, Class ID: {obj['class_id']}, Track ID: {obj['track_id']}, Confidence: {obj['confidence']}, BBox: {obj['bbox']}\n")

print("Tracking information saved to tracking_info.txt")
