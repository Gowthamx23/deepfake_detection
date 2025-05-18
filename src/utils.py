import numpy as np
import matplotlib.pyplot as plt

def get_ui_styles():
    """
    Custom CSS for dark theme.
    """
    return """
    <style>
        .stApp {
            background-color: #1a1a1a;
            font-family: Arial, sans-serif;
            color: #ffffff;
        }
        .title {
            color: #00ff00;
            font-size: 2em;
            text-align: center;
            margin-bottom: 20px;
        }
        .prediction-box {
            background-color: #333333;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            color: #ffffff;
        }
        .prediction-box b {
            color: #00ff00;
        }
        .stSpinner, .stText, .stError, .stSuccess, .stWarning {
            color: #ffffff !important;
        }
        .stError {
            background-color: #660000 !important;
        }
        .stWarning {
            background-color: #663300 !important;
        }
        .stSuccess {
            background-color: #006600 !important;
        }
        .stButton>button {
            background-color: #00ff00;
            color: #000000;
            border-radius: 5px;
            padding: 8px 16px;
        }
        .stButton>button:hover {
            background-color: #33ff33;
        }
    </style>
    """

def plot_confidence_scores(predictions, confidences):
    """
    Plot frame-level confidence scores for dark theme.
    """
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ['#00ff00' if p == 'Real' else '#ff0000' for p in predictions]
    ax.bar(range(len(confidences)), confidences, color=colors)
    ax.set_xlabel('Frame Index')
    ax.set_ylabel('Confidence')
    ax.set_ylim(0, 1)
    ax.set_facecolor('#333333')
    fig.patch.set_facecolor('#1a1a1a')
    ax.xaxis.label.set_color('#ffffff')
    ax.yaxis.label.set_color('#ffffff')
    ax.tick_params(colors='#ffffff')
    plt.tight_layout()
    return fig