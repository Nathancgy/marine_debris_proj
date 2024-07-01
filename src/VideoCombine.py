import cv2
import sys

def combine_videos(video_paths, output_path):
    videos = [cv2.VideoCapture(path) for path in video_paths]

    if not all(video.isOpened() for video in videos):
        print("Error: One or more videos failed to open.")
        for video in videos:
            video.release()
        exit()

    frame_width = int(videos[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(videos[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(videos[0].get(cv2.CAP_PROP_FPS))

    output_size = (frame_width * 2, frame_height * 2)  # Adjusted for 2x2 grid
    output_fps = fps

    output_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), output_fps, output_size)

    while True:
        frames = []
        for video in videos:
            ret, frame = video.read()
            if not ret:
                break
            small_frame = cv2.resize(frame, (frame_width, frame_height))
            frames.append(small_frame)
        
        if len(frames) < 4:
            break

        top_row = cv2.hconcat([frames[0], frames[1]])
        bottom_row = cv2.hconcat([frames[2], frames[3]])
        combined_frame = cv2.vconcat([top_row, bottom_row])

        output_video.write(combined_frame)

    for video in videos:
        video.release()
    output_video.release()

if __name__ == "__main__":
    video_paths = sys.argv[1:5]
    output_path = sys.argv[5]
    combine_videos(video_paths, output_path)
