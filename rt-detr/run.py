from ultralytics import RTDETR

# Change with your weights file
model = RTDETR("weights.pt")

# Change with your source file
predictframe=model.predict(source="/Users/nathanchen/Desktop/Competitions/Avant/sea_debris_proj/src/k.mp4",show=True,save=True)