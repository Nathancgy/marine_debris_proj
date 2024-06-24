from ultralytics import YOLO

model = YOLO("runs/detect/train3/weights/best.pt")

predictframe=model.predict(source=0,show=True)
