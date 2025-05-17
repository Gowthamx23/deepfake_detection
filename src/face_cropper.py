import cv2
import os
import numpy as np
from PIL import Image

def crop_faces_from_frames(frame_dir, output_dir):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    os.makedirs(output_dir, exist_ok=True)

    cropped_faces = []

    for filename in os.listdir(frame_dir):
        frame_path = os.path.join(frame_dir, filename)
        frame = cv2.imread(frame_path)

        if frame is None:
            print(f"[SKIP] Couldn't read frame: {frame_path}")
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for i, (x, y, w, h) in enumerate(faces):
            try:
                face_img = frame[y:y+h, x:x+w]

                if face_img is None or face_img.shape[0] < 10 or face_img.shape[1] < 10:
                    print(f"[SKIP] Invalid face crop size: {face_img.shape if face_img is not None else 'None'}")
                    continue

                face_img = cv2.resize(face_img, (224, 224))

                if face_img.dtype != np.uint8:
                    face_img = face_img.astype(np.uint8)

                if face_img.shape != (224, 224, 3):
                    print(f"[SKIP] Invalid shape after resize: {face_img.shape}")
                    continue

                print(f"[INFO] Saving face with shape {face_img.shape} and dtype {face_img.dtype}")  # Debug log

                out_path = os.path.join(output_dir, f"{filename[:-4]}_face{i}.jpg")
                Image.fromarray(face_img).save(out_path)
                cropped_faces.append(out_path)

            except Exception as e:
                print(f"[ERROR] While processing face from {filename}: {e}")
                continue

    return cropped_faces
