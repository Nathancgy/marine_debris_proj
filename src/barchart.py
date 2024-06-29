import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2

video_path = "k.mp4"

video_capture = cv2.VideoCapture(video_path)

if not video_capture.isOpened():
    print("Error: Could not open video.")
    exit()

fps = video_capture.get(cv2.CAP_PROP_FPS)
print(f"Frames per second: {fps}")

video_capture.release()

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

if len(unique_frame_ids) % fps != 0:
    frame_data.append((class_counts[:], current_time))

ani = animation.FuncAnimation(fig, update_bars, frames=frame_data, repeat=False)

ani.save('class_counts_video.mp4', writer='ffmpeg', fps=1)
