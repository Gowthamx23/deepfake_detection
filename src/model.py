import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18
import os

class VideoAIDetector(nn.Module):
    """
    R2Plus1D-18 model for detecting AI-generated videos.
    """
    def __init__(self, pretrained=True, num_classes=2):
        """
        Initialize R2Plus1D-18 with pretrained weights.
        Args:
            pretrained: If True, load pretrained weights (Kinetics-400).
            num_classes: Number of output classes (2 for Real vs. AI-generated).
        """
        super(VideoAIDetector, self).__init__()
        
        # Load R2Plus1D-18 model
        try:
            self.model = r2plus1d_18(pretrained=pretrained)
        except Exception as e:
            raise ValueError(f"Failed to load R2Plus1D-18 model: {e}")

        # Modify the final layer for binary classification
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

        # Load synthetic fine-tuned weights if available
        checkpoint_path = 'models/r2plus1d_18_synthetic.pth'
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                self.model.load_state_dict(checkpoint)
                print(f"Loaded synthetic fine-tuned weights from {checkpoint_path}")
            except Exception as e:
                print(f"Warning: Failed to load synthetic weights: {e}")

        self.model.eval()

    def forward(self, x):
        """
        Forward pass through the model.
        Args:
            x: Input tensor (B, C, T, H, W).
        Returns:
            Output tensor with logits (B, num_classes).
        """
        return self.model(x)

    def to_device(self, device):
        """
        Move model to specified device (CPU or GPU).
        Args:
            device: torch.device (e.g., 'cpu' or 'cuda').
        """
        self.to(device)
        return self