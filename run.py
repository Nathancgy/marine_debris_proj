from ultralytics import YOLO

model = YOLO("weights.pt")

predictframe=model.predict(source="",show=True)
