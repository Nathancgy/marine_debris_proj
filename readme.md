# Marine Debris Detection Project

This repository contains a Real-Time DEtection TRansformer (RT-DETR)-based project for detecting marine debris on sandy beaches. The dataset (made by the owners of this repository: [Nathancgy](https://github.com/Nathancgy) and [KrishManan](https://github.com/KrishManan)) includes approximately 400 images categorized into seven types of debris: plastic bottle, styrofoam, plastic container, plastic bag, can, and tire. Each image is annotated with bounding boxes suitable for YOLOv10 training.

## Features
- **Data**: 400 images with annotations for seven categories of marine debris.
- **Model**: Fine-tuned YOLOv8l, YOLOv10l, and RT-DETR model for accurate prediction of plastic waste.

### Validation Label
<img src = 'https://github.com/Nathancgy/marine_debris_proj/blob/main/img/val_label.jpg?raw=true' width = '600'>

### Validation Prediction
<img src = 'https://github.com/Nathancgy/marine_debris_proj/blob/main/img/val_pred.jpg?raw=true' width = '600'>

### Sample Result on Beach. Photo taken on 06/25/2024.
<img src = 'https://github.com/Nathancgy/marine_debris_proj/blob/main/img/beach.png?raw=true' width = '600'>

### Sample Result in real time detection video.
<video width="640" height="480" controls>
  <source src="img/demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## Installation
Clone the repository and install the required packages:
```bash
git clone https://github.com/Nathancgy/marine_debris_proj.git
cd marine_debris_proj
pip install -r requirements.txt
```

## Training
Configure the config.yaml file to have the paths to your training images.
```bash
config.yaml
```
Run the train script.
```bash
train.py
```
or use the CLI command
```bash
yolo task=detect mode=train epochs=[num of epochs] batch=[num of batches] plots=True model=[your weights] data=config.yaml imgsz=1000
```


## Usage
- **Weights**: Pretrained weights are not yet released. They are being verified and improved for further usages.
Run the detection script with customed weights:
```bash
python run.py
```
or use the CLI command
```bash
yolo detect predict model=[Your Weights] source=[source].jpg
```

## Future Work
This project is in the development phase and aims to be implemented in real-world scenarios.

Feel free to modify or expand upon this as needed for your project.
