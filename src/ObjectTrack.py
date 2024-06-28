import sys
import os

from sort import Sort
from lib import VisTrack, show_video, create_video

import numpy as np
import PIL.Image
import cv2
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import shutil
from ultralytics import YOLO
import ultralytics
"""
results = model.predict(source='test.png')
results_dict = results[0].__dict__

p_img = PIL.Image.open('test.png')
boxes = results_dict['boxes'].xyxy 
classes = results_dict['boxes'].cls
scores = results_dict['boxes'].conf 
vt.draw_bounding_boxes(p_img, boxes.numpy(), classes.numpy(), scores.numpy(), 20)
"""
vidcap = cv2.VideoCapture("k.mp4")
fps = vidcap.get(cv2.CAP_PROP_FPS)
length = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

vt = VisTrack()
model = YOLO("../weights2.pt")

folder_out = "Track"
if os.path.exists(folder_out):
    shutil.rmtree(folder_out)
os.makedirs(folder_out)

draw_imgs = []

sort = Sort(max_age=45, min_hits=3, iou_threshold=0.2)

pbar = tqdm(total=length)
i = 0

while True:
    ret, frame = vidcap.read()
    if not ret:
        break

    results = model.predict(frame)
    results_dict = results[0].__dict__
    boxes = results_dict['boxes'].xyxy 
    boxes = boxes.numpy()
    classes = results_dict['boxes'].cls
    classes = classes.numpy()
    scores = results_dict['boxes'].conf 
    scores = scores.numpy()
    detections_in_frame = len(boxes)
    print(boxes, classes, scores)

    if not detections_in_frame:
        boxes = np.empty((0, 5))

    dets = np.hstack((boxes, scores[:,np.newaxis]))
    res = sort.update(dets)

    boxes_track = res[:,:-1]
    boces_ids = res[:,-1].astype(int)

    p_frame = PIL.Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if detections_in_frame:
        p_frame = vt.draw_bounding_boxes(p_frame, boxes_track, boces_ids, scores, 100, 30, 23)
    p_frame.save(os.path.join(folder_out, f"{i:03d}.png"))

    i+=1
    pbar.update(1)


track_video_file = 'tracking.mp4'
create_video(frames_patten='Track/%03d.png', video_file = track_video_file, framerate=fps)

show_video(track_video_file)

