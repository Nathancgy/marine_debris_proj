from ultralytics import YOLO

# Change with your weights file
model = YOLO("Trained_Weights.pt")

# Change with your source file
predictframe=model.predict(source="Drone Videos/Pictures/Test.jpg",show=True,save=True)