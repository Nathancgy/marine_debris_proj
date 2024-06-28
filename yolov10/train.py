from ultralytics import YOLO

model = YOLO("yolov10n.pt")

model.train(data="your/path.yaml", epochs=1)