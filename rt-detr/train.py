from ultralytics import RTDETR

model = RTDETR("rtdetr-l.pt")

model.train(data="your/path.yaml", epochs=1)