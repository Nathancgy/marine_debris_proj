import cv2

output_image=cv2.haveImageWriter

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

    output_size = (frame_width, frame_height)
    output_fps = fps

    output_video = cv2.VideoWriter(output_path, cv2.VideoWriter.fourcc(*'mp4v'), output_fps, output_size)
    framecount=0
    while True:
        frames = []
        for video in videos:
            ret, frame = video.read()
            if not ret:
                break
            small_frame = cv2.resize(frame, (frame_width // 2, frame_height // 2))
            frames.append(small_frame)

        if len(frames) < 4:
            break

        combined_frame = cv2.hconcat([cv2.vconcat([frames[0], frames[1]]), cv2.vconcat([frames[2], frames[3]])])

        cv2.imshow('Combined Frame', combined_frame)
        cv2.imwrite(f'frames/combined_frame.jpg{framecount}', combined_frame)
        framecount+=1

        output_video.write(combined_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for video in videos:
        video.release()
    output_video.release()
    cv2.destroyAllWindows()

video_paths = ["styro.mp4", "styro.mp4", "styro.mp4", "styro.mp4"]

combine_videos(video_paths, "combined.mp4")
