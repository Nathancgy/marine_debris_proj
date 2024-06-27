from ultralytics import YOLO

# Change with your weights file
model = YOLO("weights.pt")

# Change with your source file
predictframe=model.predict(source="/Users/nathanchen/Downloads/11.mp4",show=True,save=True)