from ultralytics import YOLO

model = YOLO("yolov10n.pt")

model.train(data="Marine-Debris-1/data.yaml", epochs=1)