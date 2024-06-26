from ultralytics import YOLO

# Change with your weights file
model = YOLO("weights.pt")

# Change with your source file
predictframe=model.predict(source="/Users/nathanchen/Desktop/2.jpg",show=True,save=True)