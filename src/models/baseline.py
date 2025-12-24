import torch
from torch import nn
from torchvision import models


class BaselineModel(nn.Module):
    """
    Model based on ResNet-18 backbone
    """
    def __init__(self, num_classes: int):
        """
        Initializes a model
        
        Args:
            num_classes (int): Number of tags
        """
        super().__init__()

        # Setup pretrained ResNet-18
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Capture the number of input features for the FC layer
        num_features = self.backbone.fc.in_features
        
        # Replace the original FC layer with Identity to use it as a feature extractor
        self.backbone.fc = nn.Identity()

        # Custom classification head
        self.head = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        features = self.backbone(X)
        logits = self.head(features)
        return logits
