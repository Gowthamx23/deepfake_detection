import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# Define image transformations (XceptionNet expects 299x299 RGB images)
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Adjust if needed
])

class DeepfakeDetector:
    def __init__(self, model_path=None, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Load pretrained XceptionNet model - placeholder architecture
        # You can replace this with an actual pretrained model for deepfake detection
        self.model = self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

    def load_model(self, model_path):
        # For simplicity, use torchvision's xception or import from a repo
        # Here, let's assume a dummy model (replace with actual weights/model)

        from src.xception import Xception


        model = Xception()
        
        # Replace final layer for binary classification
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)

        if model_path and os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            print("Loaded model weights from", model_path)
        else:
            print("Model weights not found, using pretrained ImageNet weights")

        return model

    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        input_tensor = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_tensor)
            _, preds = torch.max(outputs, 1)
            probs = torch.softmax(outputs, dim=1)

        label = 'Fake' if preds.item() == 1 else 'Real'
        confidence = probs[0][preds.item()].item()

        return label, confidence
