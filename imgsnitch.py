import cv2
import numpy as np
import os

def stitch_images( images):
    stitcher = cv2.Stitcher_create( )
    status, stitched = stitcher.stitch(images)
    return stitched

images = []
folder_path = "data/snitchtest/"
for filename in os.listdir(folder_path):
    if filename.endswith(".png"):
        images.append(cv2.imread(os.path.join(folder_path, filename)))
result = stitch_images(images)

cv2.imshow("Stitched Image", result)
cv2.waitKey(0)
cv2.destroyAlWindows()