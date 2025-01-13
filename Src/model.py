import torch
import torch.nn as nn
from torchvision import models

def create_model(num_classes=3):
    """
    Create the MobileNet V3 model with a custom classification head.
    Args:
        num_classes (int): Number of output classes (default: 3).
    Returns:
        torch.nn.Module: The modified MobileNet V3 model.
    """
    # Load pre-trained MobileNet V3 model
    model = models.mobilenet_v3_large(pretrained=True)

    # Replace the classifier with a custom one
    model.classifier = nn.Sequential(
        nn.Linear(model.last_channel, 128),  # Add a dense layer
        nn.ReLU(),
        nn.Dropout(0.4),                    # Add dropout for regularization
        nn.Linear(128, num_classes)         # Output layer for classification
    )
    return model
