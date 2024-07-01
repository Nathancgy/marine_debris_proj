import cv2
from ultralytics import RTDETR
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Load the RTDETR model
model = RTDETR("../weights/rtdetr.pt")

# Open the video file
video_path = "C:/Users/vikas/OneDrive/Documents/Programs/Python/Machine Learning/ObjectDetection/Yolov10/Drone Videos/DJI_0033.MP4"
cap = cv2.VideoCapture(video_path)

length = int(cap. get(cv2. CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc=cv2.VideoWriter.fourcc("M", "P", "4", "V")
out_video_path = 'tracked_video.mp4'
out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))


class_names = ['can', 'carton', 'p-bag', 'p-bottle', 'p-con', 'styrofoam', 'tire']
class_counts = [0] * 7
seen_track_ids = set()
unique_frame_ids = set()
frame_data = []
current_time = 0

fig, ax = plt.subplots()
bars = ax.bar(class_names, class_counts, color=['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink'])


out2_video_path = 'barchart_video.mp4'
out2 = cv2.VideoWriter(out2_video_path, fourcc, 1, (640,480))

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


# Loop through the video frames
framecount = 1
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        print(f"processing frame{framecount}/{length}")
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)
        annotated_frame=frame.copy()

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                label = result.names[int(box.cls)]
                confidence = box.conf.item()
                id=box.id.numpy()
                id=int(id[-1])

                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                cv2.putText(annotated_frame, f'{label} {confidence:.2f}, id:{id}', (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the annotated frame
        cv2.imshow("RT-DETR Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break      

        frame_data = []
        
        # Extract tracking information from results
        if hasattr(results[0], 'boxes'):
            for obj in results[0].boxes:
                class_id = int(obj.cls)
                track_id = int(obj.id)
                if track_id not in seen_track_ids:
                    seen_track_ids.add(track_id)
                    class_counts[class_id] += 1
                    if class_counts[class_id] > 20:
                        ax.set_ylim(0, class_counts[class_id])
                

        
        if framecount%fps ==0 or framecount==length:
            current_time += 1
            frame_data.append((class_counts[:], current_time))
            update_bars(frame_data[0])
            fig.canvas.draw()
            fig.savefig('latest_chart.png')
            plt_img = cv2.imread('latest_chart.png')
            out2.write(plt_img)
        
        # Write the annotated frame to the video file
        out.write(annotated_frame)


        framecount += 1

    

    else:
        print('End of video')
        break

# Release the video capture object and the video writer object
cap.release()
