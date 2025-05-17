import cv2
import os

def extract_frames(video_path, output_dir, frame_rate=1):
    """
    Extracts frames from a video at the given frame_rate (frames per second)
    and saves them to output_dir.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps // frame_rate)  # Capture every nth frame

    count = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{saved:03d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved += 1

        count += 1

    cap.release()
    return saved  # number of frames saved
