from ultralytics import YOLO
import cv2

# Load a model
model = YOLO("best.pt")

predictframe=model.predict(source="Drone Videos/DJI_0019.MP4",show=True)

# vid=cv2.VideoCapture("Drone Videos/DJI_0019.MP4")
# ret,frame=vid.read()
# while True:
#     predictframe=model.predict(source=frame,show=True)
#     ret, frame = vid.read()