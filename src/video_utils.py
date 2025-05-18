import cv2
import os
import numpy as np

def extract_video_clips(video_path, output_dir, clip_length=2, frame_size=(112, 112), frame_rate=1, max_clips=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    frame_interval = max(1, int(fps // frame_rate))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {video_path}, FPS: {fps}, Total Frames: {total_frames}")
    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            if frame is None or frame.size == 0:
                print(f"Warning: Invalid frame at index {frame_count}")
                continue
            try:
                frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
                frame_path = os.path.join(output_dir, f"frame_{len(frames):04d}.jpg")
                if cv2.imwrite(frame_path, frame):
                    frames.append(frame_path)
                else:
                    print(f"Warning: Failed to save frame at {frame_path}")
            except Exception as e:
                print(f"Warning: Error processing frame at index {frame_count}: {e}")
                continue
        frame_count += 1
        if len(frames) >= clip_length * max_clips:
            break
    cap.release()
    if not frames:
        raise ValueError("No valid frames extracted from video")
    clip_paths = []
    for i in range(0, len(frames), clip_length):
        clip = frames[i:i + clip_length]
        if len(clip) == clip_length:
            clip_paths.append(clip)
    if not clip_paths:
        raise ValueError("No valid clips extracted from video")
    print(f"Extracted {len(clip_paths)} clips with {clip_length} frames each")
    return clip_paths

def preprocess_clip(clip_paths, frame_size=(112, 112)):
    frames = []
    for path in clip_paths:
        frame = cv2.imread(path)
        if frame is None or frame.size == 0:
            print(f"Warning: Failed to read frame: {path}")
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
        frame = frame / 255.0
        frames.append(frame)
    if not frames:
        raise ValueError("No valid frames in clip")
    clip_array = np.array(frames).transpose(3, 0, 1, 2)
    clip_tensor = np.expand_dims(clip_array, axis=0).astype(np.float32)
    return clip_tensor