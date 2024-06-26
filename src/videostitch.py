import cv2

def stitch_images(images):
    stitcher = cv2.Stitcher_create()
    status, stitched = stitcher.stitch(images)
    if status == cv2.Stitcher_OK:
        return stitched
    else:
        print("Error during stitching.")
        return None

video_path = "/Users/nathanchen/Desktop/1.MP4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error: Could not open video {video_path}")
    exit()

images = []
frame_count = 0
recorded_count = 0
interval = 5

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    print("Error: Could not retrieve FPS from video.")
    exit()

frame_interval = int(interval * fps)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        images.append(frame)
        recorded_count += 1

    frame_count += 1

cap.release()
print(f"Number of frames recorded: {len(images)}")

if recorded_count > 1:
    stitched_image = stitch_images(images)
    if stitched_image is not None:
        cv2.imshow("Stitched Image", stitched_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
else:
    print("Not enough frames recorded to perform stitching.")
