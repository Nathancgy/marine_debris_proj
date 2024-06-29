import cv2
from ultralytics import RTDETR
from ultralytics.solutions import object_counter as oc

input_video_path = "k.mp4"
output_video_path = "count/k.mp4"

video_capture = cv2.VideoCapture(input_video_path)
assert video_capture.isOpened(), "Illegal or non-existing video file"

video_width, video_height, video_fps = (
    int(video_capture.get(p))
    for p in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
)

video_writer = cv2.VideoWriter(
    output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), video_fps, (video_width, video_height)
)

model = RTDETR("weights/rtdetr.pt")

object_counter = oc.ObjectCounter(view_img=True,
    reg_pts=[(0, 540), (1280, 540)],
    classes_names=model.names,
    draw_tracks=True
)
# object_counter.set_args(
#     view_img=True,
#     reg_pts=[(0, 540), (1280, 540)],
#     classes_names=model.names,
#     draw_tracks=True
# )

while video_capture.isOpened():
    success, frame = video_capture.read()
    if not success:
        break

    tracks = model.track(frame, persist=True, show=False, classes=[0, 2])
    frame = object_counter.start_counting(frame, tracks)
    video_writer.write(frame)

video_capture.release()
video_writer.release()
cv2.destroyAllWindows()