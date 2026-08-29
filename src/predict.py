"""
Inference script for acoustic vehicle distance detection.

Provides a unified `predict()` function that works with XGBoost, SVM, or
PyTorch models, plus a command-line interface for predicting a single file.

Author: Hananeh Shabestani, Setayesh Zaheri Azam
License: MIT
"""

import argparse
import numpy as np
import torch

from feature_extraction import extract_features_from_file
from model import (
    AudioClassifier,
    CLASS_NAMES,
    load_xgboost_model,
    load_svm_model,
    load_pytorch_model,
    load_scaler,
)


def predict(model, features, scaler=None, model_type='xgboost',
            device='cpu', threshold_c1=None):
    """
    Predict the vehicle distance class for a single feature vector.

    Parameters
    ----------
    model : model object
        A trained XGBoost, SVM, or PyTorch model.
    features : np.ndarray of shape (132,)
        Raw (unscaled) feature vector.
    scaler : StandardScaler or None
        The scaler fit on training data. If None, features must already be scaled.
    model_type : str
        'xgboost', 'svm', or 'pytorch'.
    device : str
        'cpu' or 'cuda' (PyTorch only).
    threshold_c1 : float or None
        If provided, apply safety-oriented threshold tuning for Class 1.
        If P(Class 1) > threshold_c1, force prediction to Class 1.

    Returns
    -------
    prediction : int
        Predicted class label (0-4).
    probabilities : np.ndarray
        Class probabilities.
    class_name : str
        Human-readable class name.
    """
    # Scale features if scaler is provided
    if scaler is not None:
        features = features.reshape(1, -1)
        features_scaled = scaler.transform(features)[0]
    else:
        features_scaled = features

    # Get probabilities based on model type
    if model_type in ('xgboost', 'svm'):
        probs = model.predict_proba(features_scaled.reshape(1, -1))[0]
        prediction = int(np.argmax(probs))

    elif model_type == 'pytorch':
        tensor = torch.FloatTensor(features_scaled).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        prediction = int(np.argmax(probs))

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'xgboost', 'svm', or 'pytorch'.")

    # Apply safety-oriented threshold tuning for Class 1
    if threshold_c1 is not None and probs[1] > threshold_c1:
        prediction = 1

    class_name = CLASS_NAMES.get(prediction, f'Class {prediction}')
    return prediction, probs, class_name


def predict_file(audio_path, model, scaler=None, model_type='xgboost',
                 device='cpu', threshold_c1=None):
    """
    Convenience wrapper: extract features from a file, then predict.

    Parameters
    ----------
    audio_path : str
        Path to the .wav file.
    (see predict() for other parameters)

    Returns
    -------
    (prediction, probabilities, class_name)  — same as predict()
    """
    features = extract_features_from_file(audio_path)
    if features is None:
        raise ValueError(f"Could not extract features from: {audio_path}")
    return predict(model, features, scaler=scaler, model_type=model_type,
                   device=device, threshold_c1=threshold_c1)


# ============================================================
# Command-line interface
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Predict vehicle distance class from an audio file.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict with XGBoost (default)
  python predict.py --audio vehicle.wav --model-type xgboost \\
      --model-path models/xgboost_audio_model.pkl \\
      --scaler-path models/audio_scaler.pkl

  # Predict with safety threshold tuning for Class 1
  python predict.py --audio vehicle.wav --model-type xgboost \\
      --model-path models/xgboost_audio_model.pkl \\
      --scaler-path models/audio_scaler.pkl \\
      --threshold-c1 0.25

  # Predict with PyTorch
  python predict.py --audio vehicle.wav --model-type pytorch \\
      --model-path models/dl_audio_model.pth \\
      --scaler-path models/audio_scaler.pkl
        """)
    parser.add_argument('--audio', required=True, help='Path to the .wav audio file')
    parser.add_argument('--model-type', default='xgboost',
                        choices=['xgboost', 'svm', 'pytorch'],
                        help='Type of model to use (default: xgboost)')
    parser.add_argument('--model-path', required=True, help='Path to the saved model file')
    parser.add_argument('--scaler-path', default=None,
                        help='Path to the saved StandardScaler (.pkl)')
    parser.add_argument('--threshold-c1', type=float, default=None,
                        help='Safety threshold for Class 1 (e.g., 0.25 for XGBoost)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print full probability distribution')

    args = parser.parse_args()

    # Load scaler
    scaler = None
    if args.scaler_path:
        print(f"Loading scaler from: {args.scaler_path}")
        scaler = load_scaler(args.scaler_path)

    # Load model
    print(f"Loading {args.model_type} model from: {args.model_path}")
    if args.model_type == 'xgboost':
        model = load_xgboost_model(args.model_path)
    elif args.model_type == 'svm':
        model = load_svm_model(args.model_path)
    else:
        model = load_pytorch_model(args.model_path)

    # Predict
    print(f"\nExtracting features from: {args.audio}")
    prediction, probs, class_name = predict_file(
        args.audio, model, scaler=scaler, model_type=args.model_type,
        threshold_c1=args.threshold_c1)

    print(f"\n{'='*50}")
    print(f"PREDICTION: Class {prediction}")
    print(f"MEANING:    {class_name}")
    print(f"{'='*50}")

    if args.verbose:
        print(f"\nFull probability distribution:")
        for i, p in enumerate(probs):
            name = CLASS_NAMES.get(i, f'Class {i}')
            bar = '#' * int(p * 40)
            print(f"  Class {i} ({name:35s}): {p:.4f} |{bar}")

    # Safety warning
    if prediction == 1:
        print(f"\n[WARNING] DANGER ZONE DETECTED — Vehicle approaching within ~20m!")


if __name__ == '__main__':
    main()
