import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import streamlit as st
import os
import shutil
from src.video_utils import extract_frames
from src.face_cropper import crop_faces_from_frames
from src.inference import DeepfakeDetector

# Create temp folders
FRAME_DIR = "temp/frames"
FACE_DIR = "temp/faces"

def clear_temp_dirs():
    for d in [FRAME_DIR, FACE_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

st.title("Deepfake Video Detection System")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi"])

if uploaded_file:
    clear_temp_dirs()
    
    video_path = os.path.join("temp", uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.text("Extracting frames...")
    extract_frames(video_path, FRAME_DIR, frame_rate=1)

    st.text("Detecting and cropping faces...")
    cropped_faces = crop_faces_from_frames(FRAME_DIR, FACE_DIR)

    if not cropped_faces:
        st.warning("No faces detected in the video.")
    else:
        st.text(f"Cropped {len(cropped_faces)} face(s).")
        detector = DeepfakeDetector()

        results = []
        for face_path in cropped_faces:
            label, confidence = detector.predict(face_path)
            results.append((face_path, label, confidence))

        # Majority vote for video prediction
        fake_count = sum(1 for _, label, _ in results if label == "Fake")
        real_count = len(results) - fake_count

        video_label = "Fake" if fake_count > real_count else "Real"
        st.subheader(f"Overall Video Prediction: {video_label}")

        st.write("Detailed Frame Predictions:")
        for face_path, label, conf in results:
            st.write(f"{os.path.basename(face_path)}: **{label}** ({conf:.2f})")
