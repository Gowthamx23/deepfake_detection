import torch
import numpy as np
from src.model import VideoAIDetector
from src.video_utils import preprocess_clip

class AIDetector:
    def __init__(self, pretrained=True, device='cpu'):
        self.device = torch.device(device)
        self.model = VideoAIDetector(pretrained=pretrained).to_device(self.device)

    def predict_clip(self, clip_tensor):
        clip_tensor = torch.from_numpy(clip_tensor).float().to(self.device)
        try:
            with torch.no_grad():
                outputs = self.model(clip_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, pred = torch.max(probs, dim=1)
            label = 'AI-generated' if pred.item() == 1 else 'Real'
            return label, confidence.item()
        except Exception as e:
            print(f"Prediction failed: {e}")
            return 'Unknown', 0.0

    def predict_video(self, clip_paths):
        if not clip_paths:
            print("No clips provided for prediction")
            return 'Unknown', 0.0, []

        predictions = []
        confidences = []
        clip_results = []

        batch_size = 2
        for i in range(0, len(clip_paths), batch_size):
            batch_clips = clip_paths[i:i + batch_size]
            try:
                for clip in batch_clips:
                    clip_tensor = preprocess_clip(clip)
                    label, confidence = self.predict_clip(clip_tensor)
                    print(f"Clip {len(predictions)+1}: {label}, Confidence: {confidence:.2f}")
                    if confidence >= 0.3:  # Stricter filter for reliability
                        predictions.append(label)
                        confidences.append(confidence)
                    clip_results.append((label, confidence))
            except Exception as e:
                print(f"Error processing clip batch {i//batch_size + 1}: {e}")
                continue

        if not predictions:
            print("No reliable clip predictions (all confidences < 0.3)")
            return 'Unknown', 0.0, clip_results

        # Weighted voting: sum confidence-weighted scores
        ai_score = sum(c for p, c in zip(predictions, confidences) if p == 'AI-generated')
        real_score = sum(c for p, c in zip(predictions, confidences) if p == 'Real')
        total_score = ai_score + real_score
        if total_score == 0:
            print("No valid scores after weighting")
            return 'Unknown', 0.0, clip_results

        video_confidence = ai_score / total_score
        video_label = 'AI-generated' if video_confidence < 0.5 else 'Real'

        print(f"Video prediction: {video_label}, Confidence: {video_confidence:.2f}")
        return video_label, video_confidence, clip_results  