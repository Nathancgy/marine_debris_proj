import subprocess 
from ultralytics import RTDETR 
import matplotlib.pyplot as plt 
import numpy as np 
import cv2 
import imageio_ffmpeg as ffmpeg

model = RTDETR("../weights/rtdetr.pt")

video_path = 'C:/Users/vikas/OneDrive/Documents/Programs/Python/Machine Learning/ObjectDetection/Yolov10/Drone Videos/DJI_0033.MP4'
cap = cv2.VideoCapture(video_path)

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
fps = int(cap.get(cv2.CAP_PROP_FPS)) 
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

annotated_video_path = 'tracked_video.mp4' 
ffmpeg_command_annotated = [ ffmpeg.get_ffmpeg_exe(), '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', f"{width}x{height}", '-pix_fmt', 'bgr24', '-r', str(fps), '-i', '-', '-an', '-vcodec', 'mpeg4', annotated_video_path ]

out2_video_path = 'barchart_video.mp4' 
ffmpeg_command_barchart = [ ffmpeg.get_ffmpeg_exe(), '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-s', '640x480', '-pix_fmt', 'bgr24', '-r', '1', '-i', '-', '-an', '-vcodec', 'mpeg4', out2_video_path ]

pipe_annotated = subprocess.Popen(ffmpeg_command_annotated, stdin=subprocess.PIPE, stderr=subprocess.PIPE) 
pipe_barchart = subprocess.Popen(ffmpeg_command_barchart, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

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

print(f"Frames per second: {fps}")


framecount = 1 
while cap.isOpened(): 
    success, frame = cap.read()

    if success:
        print(f"processing frame {framecount}/{length}")
        results = model.track(frame, persist=True)
        annotated_frame = frame.copy()

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                label = result.names[int(box.cls)]
                confidence = box.conf.item()
                id = box.id.numpy()
                id = int(id[-1])

                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(annotated_frame, f'{label} {confidence:.2f}, id:{id}', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        frame_data = []
        
        if hasattr(results[0], 'boxes'):
            for obj in results[0].boxes:
                class_id = int(obj.cls)
                track_id = int(obj.id)
                if track_id not in seen_track_ids:
                    seen_track_ids.add(track_id)
                    class_counts[class_id] += 1
                    if class_counts[class_id] > 20:
                        ax.set_ylim(0, class_counts[class_id])
        
        if framecount % fps == 0 or framecount == length:
            current_time += 1
            frame_data.append((class_counts[:], current_time))
            update_bars(frame_data[0])
            fig.canvas.draw()
            fig.savefig('latest_chart.png')
            plt_img = cv2.imread('latest_chart.png')
            pipe_barchart.stdin.write(plt_img.tobytes())
        
        pipe_annotated.stdin.write(annotated_frame.tobytes())
        framecount += 1

    else:
        print('End of video')
        break
cap.release() 
pipe_annotated.stdin.close() 
pipe_annotated.stderr.close() 
pipe_annotated.wait()

pipe_barchart.stdin.close() 
pipe_barchart.stderr.close() 
pipe_barchart.wait()

print(f"Annotated video saved to {annotated_video_path}") 
print(f"Bar chart video saved to {out2_video_path}")
