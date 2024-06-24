import cv2
import os

def stitch_images(images):
    stitcher = cv2.Stitcher_create()
    status, stitched = stitcher.stitch(images)
    if status == cv2.Stitcher_OK:
        return stitched
    else:
        print(f"Error during stitching. Status code: {status}")
        return None

folder_path = "/Users/nathanchen/Desktop"

images = []

for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        file_path = os.path.join(folder_path, filename)
        image = cv2.imread(file_path)
        if image is not None:
            images.append(image)
        else:
            print(f"Error reading image: {file_path}")

if len(images) < 2:
    print("Not enough images to perform stitching. At least 2 images are required.")
else:
    stitched_image = stitch_images(images)
    if stitched_image is not None:
        cv2.imshow("Stitched Image", stitched_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Stitching failed. No image to display.")
