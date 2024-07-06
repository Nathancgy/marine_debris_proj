from ultralytics import YOLO
if __name__ == "__main__":
  model = YOLO("yolov10n.pt")

  model.train(data="your/path.yaml", epochs=1)
