from ultralytics import RTDETR 
import matplotlib.pyplot as plt 
import cv2 
import numpy as np

model = RTDETR("../weights/rtdetr.pt")

video_path = 'k2.mp4'
cap = cv2.VideoCapture(video_path)

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
fps = int(cap.get(cv2.CAP_PROP_FPS)) 
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

annotated_video_path = 'tracked_video.mp4' 
out1=cv2.VideoWriter(annotated_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

out2_video_path = 'barchart_video.mp4' 
out2=cv2.VideoWriter(out2_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (640, 480))

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


framecount = 0 
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
                id = box.id.numpy()
                id = int(id[-1])

                # Calculate area based on ratios
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                bbox_area = bbox_width * bbox_height

                if label == 'can':
                    area_ratio = 0.7
                elif label == 'carton':
                    area_ratio = 0.8
                elif label == 'p-bag':
                    area_ratio = 0.6
                elif label == 'p-bottle':
                    area_ratio = 0.75
                elif label == 'p-con':
                    area_ratio = 0.7
                elif label == 'styrofoam':
                    area_ratio = 0.65
                elif label == 'tire':
                    area_ratio = 0.9
                else:
                    area_ratio = 1  # Default ratio if class not found

                predicted_area = bbox_area * area_ratio

                # Draw bounding box
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)

                # Display label, confidence, id, and predicted area above the bounding box
                cv2.putText(annotated_frame, f'{label} {confidence:.2f}, id:{id} Area: {predicted_area:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
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
            out2.write(plt_img)
        
        combined_image = np.zeros((height, width, 3), dtype="uint8")
        annotated_frame = cv2.resize(annotated_frame, (width // 2, height // 2))
        frame= cv2.resize(frame, (width //2, height // 2))  
        plt_img= cv2.resize(plt_img, (width // 2, height // 2))
        cv2.imshow("plt",plt_img)
        cv2.waitKey(1)
        combined_image[0:height // 2, 0:width // 2] = frame
        combined_image[0:height // 2, width // 2:width] = annotated_frame
        combined_image[height // 2:height, 0:width] = plt_img

        cv2.imshow("combined image",combined_image)
        cv2.waitKey(1)
        out1.write(annotated_frame)
        framecount += 1

    else:
        print('End of video')
        break
cap.release() 


print(f"Annotated video saved to {annotated_video_path}") 
print(f"Bar chart video saved to {out2_video_path}")
