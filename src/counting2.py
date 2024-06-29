import cv2
from ultralytics import RTDETR
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Load the RTDETR model
model = RTDETR("../weights/rtdetr.pt")

# Open the video file
video_path = "../k.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Frames per second: {fps}")

framecount = 0
tracking_data = []

class_names = ['can', 'carton', 'p-bag', 'p-bottle', 'p-con', 'styrofoam', 'tire']
class_counts = [0] * 7
seen_track_ids = set()
unique_frame_ids = set()
frame_data = []
current_time = 0

fig, ax = plt.subplots()
bars = ax.bar(class_names, class_counts, color=['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink'])

ax.set_xlabel('Classes')
ax.set_ylabel('Counts')
ax.set_title('Dynamic Class Counts Over Time')
ax.set_ylim(0, 20)
ax.yaxis.get_major_locator().set_params(integer=True)

time_text = ax.text(0.95, 0.95, '', transform=ax.transAxes, ha='right', va='top', fontsize=12)

def update_bars(frame_data):
    counts, second = frame_data
    for bar, count in zip(bars, counts):
        bar.set_height(count)
    time_text.set_text(f'Time: {second} s')

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        print(f"Processing frame {framecount}...")
        
        # Run RTDETR tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)
        
        frame_data = []
        
        # Extract tracking information from results
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
        
        # Write tracking information to file for this frame
        with open("tracking_info.txt", "a") as f:
            for obj in frame_data:
                f.write(f"Frame: {obj['frame_idx']}, Class ID: {obj['class_id']}, Track ID: {obj['track_id']}, Confidence: {obj['confidence']}, BBox: {obj['bbox']}\n")

        print(f"Tracking information for frame {framecount} saved to tracking_info.txt")
        framecount += 1

        # Update class counts and frame data for animation
        with open("tracking_info.txt", "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.split(", ")
            frame_idx = int(parts[0].split(": ")[1])
            class_id = int(parts[1].split(": ")[1])
            track_id = int(parts[2].split(": ")[1])

            if frame_idx not in unique_frame_ids:
                unique_frame_ids.add(frame_idx)
                if len(unique_frame_ids) % fps == 0:
                    current_time += 1
                    frame_data.append((class_counts[:], current_time))

            if track_id not in seen_track_ids:
                seen_track_ids.add(track_id)
                class_counts[class_id] += 1

        # If the current frame count modulo fps is not zero, append the current data
        if len(unique_frame_ids) % fps != 0:
            frame_data.append((class_counts[:], current_time))

        # Update animation with the latest frame data
        ani = animation.FuncAnimation(fig, update_bars, frames=frame_data, repeat=False)

    else:
        print('End of video')
        break

# Save the animation
ani.save('class_counts_video.mp4', writer='ffmpeg', fps=1)

# Release the video capture object
cap.release()
