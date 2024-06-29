from ultralytics import YOLO

# Change with your weights file
model = YOLO("weights2.pt")

# Change with your source file
predictframe=model.predict(source="/Users/nathanchen/Desktop/Competitions/Avant/sea_debris_proj/src/test.png",show=True,save=True)