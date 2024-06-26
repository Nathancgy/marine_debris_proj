from ultralytics import YOLO

model = YOLO("yolov10n.pt")

predictframe=model.predict(source=0,show=True)
