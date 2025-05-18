import streamlit as st
import os
import shutil
import numpy as np
from src.video_utils import extract_video_clips
from src.inference import AIDetector
from src.utils import get_ui_styles, plot_confidence_scores

FRAME_DIR = "temp/frames"

def clear_temp_dirs():
    for d in [FRAME_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

st.set_page_config(page_title="AI Video Detection", layout="wide")

# Apply dark theme CSS
st.markdown(get_ui_styles(), unsafe_allow_html=True)

st.markdown('<div class="title">AI Video Detection</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a video (mp4, mov, avi)", type=["mp4", "mov", "avi"])

if uploaded_file:
    try:
        clear_temp_dirs()
        video_path = os.path.join("temp", uploaded_file.name)
        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Extracting video clips..."):
            clip_paths = extract_video_clips(video_path, FRAME_DIR, clip_length=4, frame_size=(112, 112), frame_rate=1)

        if not clip_paths:
            st.error("No valid clips extracted from the video.")
        else:
            with st.spinner("Loading AI detection model..."):
                detector = AIDetector(pretrained=True, device='cpu')

            with st.spinner("Analyzing video..."):
                video_label, video_confidence, clip_results = detector.predict_video(clip_paths)

            st.markdown(f'<div class="prediction-box">Video Prediction: <b>{video_label}</b> (Confidence: {video_confidence:.2f})</div>', unsafe_allow_html=True)

            if clip_results:
                predictions, confidences = zip(*[(p, c) for p, c in clip_results if c >= 0.3])
                if predictions:
                    fig = plot_confidence_scores(predictions, confidences)
                    st.pyplot(fig)

                    st.subheader("Clip-Level Results")
                    for idx, (label, confidence) in enumerate(clip_results):
                        st.markdown(f"Clip {idx+1}: {label} (Confidence: {confidence:.2f})")
                else:
                    st.warning("No reliable clip predictions (all confidences < 0.3).")
            else:
                st.warning("No clip-level results available.")

    except Exception as e:
        st.error(f"Error processing video: {str(e)}")

st.markdown("---")
st.markdown("Built with ❤️ for AI video detection")