"""
Model definitions for acoustic vehicle distance detection.

Contains:
    - AudioClassifier: PyTorch MLP model definition (must match the trained model)
    - Helper functions for loading saved models (XGBoost, SVM, PyTorch)

Author: Hananeh Shabestani, Setayesh Zaheri Azam
License: MIT
"""

import joblib
import torch
import torch.nn as nn
import numpy as np


class AudioClassifier(nn.Module):
    """
    PyTorch MLP classifier for acoustic features.

    Architecture:
        Input (132) -> Linear(128) -> BatchNorm -> ReLU -> Dropout(0.3)
                    -> Linear(64)  -> BatchNorm -> ReLU -> Dropout(0.3)
                    -> Linear(32)  -> ReLU
                    -> Linear(num_classes)

    This must match the architecture used during training for the saved
    state_dict to load correctly.

    Parameters
    ----------
    input_dim : int
        Number of input features (default 132).
    num_classes : int
        Number of output classes (default 5).
    """

    def __init__(self, input_dim=132, num_classes=5):
        super(AudioClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)

        self.layer2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)

        self.layer3 = nn.Linear(64, 32)

        self.output = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)

        x = self.layer2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)

        x = self.layer3(x)
        x = self.relu(x)

        x = self.output(x)
        return x


# Class names for human-readable output
CLASS_NAMES = {
    0: 'Far-Approaching (~50m -> 20m)',
    1: 'DANGER (~20m -> 0m)',
    2: 'Near-Receding (0m -> ~20m)',
    3: 'Far-Receding (20m -> 50m)',
    4: 'Safe (No vehicle)',
}


def load_xgboost_model(path):
    """Load a saved XGBoost model (.pkl)."""
    return joblib.load(path)


def load_svm_model(path):
    """Load a saved SVM model (.pkl)."""
    return joblib.load(path)


def load_pytorch_model(path, input_dim=132, num_classes=5, device='cpu'):
    """
    Load a saved PyTorch model state_dict (.pth).

    Parameters
    ----------
    path : str
        Path to the .pth file.
    input_dim : int
        Must match the input dimension used during training.
    num_classes : int
        Must match the number of classes.
    device : str
        'cpu' or 'cuda'.

    Returns
    -------
    AudioClassifier
        The loaded model in eval mode.
    """
    model = AudioClassifier(input_dim=input_dim, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def load_scaler(path):
    """Load the StandardScaler used during training."""
    return joblib.load(path)


if __name__ == '__main__':
    # Quick architecture test
    model = AudioClassifier(input_dim=132, num_classes=5)
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test forward pass
    dummy_input = torch.randn(1, 132)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    print(f"Class names: {CLASS_NAMES}")
