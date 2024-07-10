from ultralytics import RTDETR

# Change with your weights file
model = RTDETR("../rt-detr/weights.pt")

# Change with your source file
predictframe=model.track(source="../src/k.mp4",save=True,tracker="bytetrack.yaml")