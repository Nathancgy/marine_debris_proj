from ultralytics import RTDETR

# Change with your weights file
model = RTDETR("weights2.pt")

# Change with your source file
predictframe=model.predict(source="/Users/nathanchen/Desktop/Competitions/Avant/sea_debris_proj/src/test.png",show=True,save=True)