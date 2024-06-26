# Marine Debris Detection Project

This repository contains a YOLOv10-based project for detecting marine debris on sandy beaches. The dataset includes approximately 400 images categorized into seven types of debris: plastic bottle, styrofoam, plastic container, plastic bag, can, and tire. Each image is annotated with bounding boxes suitable for YOLOv10 training.

## Features
- **Data**: 400 images with annotations for seven categories of marine debris.
- **Model**: Fine-tuned YOLOv10 model for accurate prediction of plastic waste.

### Validation Label
<img src = 'https://github.com/Nathancgy/marine_debris_proj/blob/main/img/val_label.jpg?raw=true' width = '600'>

### Validation Prediction
<img src = 'https://github.com/Nathancgy/marine_debris_proj/blob/main/img/val_pred.jpg?raw=true' width = '600'>

### Sample Result on Beach. Photo taken on 06/25/2024.
<img src = 'https://github.com/Nathancgy/marine_debris_proj/blob/main/img/beach.png?raw=true' width = '600'>

## Installation
Clone the repository and install the required packages:
```bash
git clone https://github.com/Nathancgy/marine_debris_proj.git
cd marine_debris_proj
pip install -r requirements.txt
```

## Usage
- **Weights**: Pretrained weights are not yet released. They are being verified and improved for further usages.
Run the detection script with customed weights:
```bash
python run.py
```
## Future Work
This project is in the development phase and aims to be implemented in real-world scenarios.

Feel free to modify or expand upon this as needed for your project.