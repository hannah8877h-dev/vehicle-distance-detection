"""
src package - Acoustic Vehicle Distance Detection

Modules:
    - feature_extraction: Extract 132 acoustic features from audio files
    - model: PyTorch MLP model definition and model loading utilities
    - predict: Inference functions and CLI

Usage:
    from src.feature_extraction import extract_features_from_file
    from src.model import load_xgboost_model, load_scaler
    from src.predict import predict_file
"""

__version__ = "1.0.0"
__authors__ = "Hananeh Shabestani, Setayesh Zaheri Azam"
