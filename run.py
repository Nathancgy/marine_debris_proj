from ultralytics import YOLO

# Change with your weights file
model = YOLO("weights.pt")

# Change with your source file
predictframe=model.predict(source="",show=True)
