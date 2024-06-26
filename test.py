from roboflow import Roboflow
rf = Roboflow(api_key="pz0GGVyqdzrz268Xf7Gb")
project = rf.workspace("nathan-chen").project("marine-debris-qcfuj")
version = project.version(1)
dataset = version.download("yolov9")