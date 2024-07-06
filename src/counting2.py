from ultralytics import RTDETR 
import matplotlib.pyplot as plt 
import cv2 
import numpy as np

model = RTDETR("../weights/rtdetr.pt")

video_path = 'k.mp4'
cap = cv2.VideoCapture(video_path)

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) 
fps = int(cap.get(cv2.CAP_PROP_FPS)) 
width = 2000
height = 1000

print(height,width)

annotated_video_path = 'combined_video.mp4' 
out1=cv2.VideoWriter(annotated_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))


class_names = ['can', 'carton', 'p-bag', 'p-bottle', 'p-con', 'styrofoam', 'tire'] 
class_counts = [0] * 7 
class_areas = [0] * 7
seen_track_ids = set() 
unique_frame_ids = set() 
frame_data = [] 
current_time = 0

colors=['blue', 'dodgerblue', 'royalblue', 'steelblue', 'darkblue', 'deepskyblue', 'aquamarine']
fig, ax = plt.subplots() 
bars = ax.bar(class_names, class_counts, color=colors)

ax.set_xlabel('Classes') 
ax.set_ylabel('Counts') 
ax.set_title('Dynamic Class Counts Over Time') 
ax.set_ylim(0, 20) 
ax.yaxis.get_major_locator().set_params(integer=True)

time_text = ax.text(0.95, 0.95, '', transform=ax.transAxes, ha='right', va='top', fontsize=12)

def removezeros(class_names,class_areas):
    new_class_areas=[]
    new_class_names=[]
    for i in range(len(class_areas)):
        if class_areas[i]!=0:
            new_class_areas.append(class_areas[i])
            new_class_names.append(class_names[i])
    return new_class_names,new_class_areas

def add_text_with_background(img, text, pos, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.7, color=(139, 0, 0), thickness=2, bg_color=(255, 255, 255)):
    # Get the size of the text
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Calculate the position for the background rectangle
    bg_top_left = (pos[0], pos[1] - text_size[1] - 5)
    bg_bottom_right = (pos[0] + text_size[0], pos[1])
    
    # Draw the filled rectangle behind the text
    cv2.rectangle(img, bg_top_left, bg_bottom_right, bg_color, -1)  # -1 thickness for filling
    
    # Write the text on the image
    cv2.putText(img, text, pos, font, font_scale, color, thickness)


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
        frame=cv2.resize(frame,(width,height))
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
                    area_ratio = 0.85
                elif label == 'carton':
                    area_ratio = 0.85
                elif label == 'p-bag':
                    area_ratio = 0.7
                elif label == 'p-bottle':
                    area_ratio = 0.75
                elif label == 'p-con':
                    area_ratio = 0.9
                elif label == 'styrofoam':
                    area_ratio = 0.80
                elif label == 'tire':
                    area_ratio = 0.9
                else:
                    area_ratio = 1  # Default ratio if class not found

                predicted_area = int(bbox_area * area_ratio)

                # Draw bounding box
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (139, 0, 0), 4)

                # Display label, confidence, id, and predicted area above the bounding box
                add_text_with_background(annotated_frame, f'{label} {confidence:.2f}, id:{id}, Area: {predicted_area:.2f}', (int(x1), int(y1)-10), bg_color=(255, 255, 255))
            
        frame_data = []
        
        if hasattr(results[0], 'boxes'):
            for obj in results[0].boxes:
                class_id = int(obj.cls)
                track_id = int(obj.id)
                if track_id not in seen_track_ids:
                    seen_track_ids.add(track_id)
                    class_counts[class_id] += 1
                    class_areas[class_id] += int(predicted_area)
                    if class_counts[class_id] > 20:
                        ax.set_ylim(0, class_counts[class_id])
            

        if framecount % fps == 0 or framecount == length:
            current_time += 1
            frame_data.append((class_counts[:], current_time))
            update_bars(frame_data[0])
            fig.canvas.draw()
            fig.savefig('latest_chart.png')
            plt_img = cv2.imread('latest_chart.png')
            fig2, ax2 = plt.subplots()
            ax2.set_title('Dynamic Class Areas Over Time (%)')
            new_class_names,new_class_areas = removezeros(class_names,class_areas)
            print (new_class_names,new_class_areas)
            pie = ax2.pie(new_class_areas, labels=new_class_names, autopct='%1.1f%%', startangle=140,colors=colors)
            ax2.axis('equal')
            fig2.canvas.draw()
            fig2.savefig('latest_pie_chart.png')
            plt.close(fig2)
            pie_img = cv2.imread('latest_pie_chart.png')
        
        combined_image = np.zeros((height, width, 3), dtype="uint8")
        annotated_frame = cv2.resize(annotated_frame, (width // 2, height // 2))
        frame= cv2.resize(frame, (width //2, height // 2))  
        plt_img= cv2.resize(plt_img, (width // 2, height // 2))
        pie_img= cv2.resize(pie_img, (width // 2, height // 2))
        key=cv2.waitKey(1)
        if key== ord('q'):
            break
        combined_image[:height // 2, :width // 2] = frame
        combined_image[:height // 2, width // 2:width] = annotated_frame
        combined_image[height // 2:height, :width//2] = plt_img
        combined_image[height // 2:height, width // 2:width] = pie_img

        cv2.imshow("combined image",combined_image)
        cv2.waitKey(1)
        out1.write(combined_image)
        framecount += 1

    else:
        print('End of video')
        break
cap.release() 


print(f"Annotated video saved to {annotated_video_path}") 
