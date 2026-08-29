# Trained Models

## Status

The trained model files (`.pkl`, `.pth`) are **not included** in this repository because they are binary artifacts that GitHub is not optimized to host. The `.gitignore` file excludes these formats by default.

## How to Reproduce the Models

Run the main training notebook to regenerate all models from scratch:

```bash
jupyter notebook notebooks/vehicle_distance_detection.ipynb
```

After training, the following files will be generated:

| File | Description | Approximate Size |
|------|-------------|:----------------:|
| `xgboost_audio_model.pkl` | Default XGBoost model | ~500 KB |
| `xgboost_tuned_model.pkl` | XGBoost model + threshold (0.25) | ~500 KB |
| `svm_audio_model.pkl` | Default SVM model | ~5 MB |
| `svm_tuned_model.pkl` | SVM model + threshold (0.10) | ~5 MB |
| `dl_audio_model.pth` | PyTorch MLP model state_dict | ~50 KB |
| `audio_scaler.pkl` | StandardScaler used for preprocessing | ~5 KB |
| `feature_metadata.pkl` | Feature extraction metadata | ~5 KB |

## Model Performance Summary

### Validation Set

| Model | Accuracy | Class 1 Recall | Notes |
|-------|----------|----------------|-------|
| XGBoost (Tuned) | 82.8% | 90.4% | Best overall balance |
| SVM (Tuned) | 74.5% | 97.8% | Highest safety recall |
| PyTorch MLP (Tuned) | 81.8% | 94.0% | Strong DL baseline |

### Unseen Test Set (New Environment)

| Model | Accuracy | Class 1 Recall |
|-------|----------|----------------|
| XGBoost | 73.7% | 78.4% |
| SVM | 69.8% | 77.3% |
| PyTorch MLP | N/A (version mismatch) | — |

## Loading Models for Inference

```python
from src.model import load_xgboost_model, load_scaler
from src.predict import predict_file

model = load_xgboost_model('models/xgboost_audio_model.pkl')
scaler = load_scaler('models/audio_scaler.pkl')

prediction, probs, class_name = predict_file(
    'path/to/audio.wav',
    model,
    scaler=scaler,
    model_type='xgboost',
    threshold_c1=0.25  # safety-oriented threshold
)
```

## Alternative: Download Pre-trained Models

If you only need the trained models for inference (without retraining), contact the authors to receive a download link. The models will also be published on Hugging Face Hub alongside the dataset release.
