# Results Summary

This file consolidates all experimental results from the project for quick reference.

## 1. Validation Set Performance (20% hold-out, stratified, random_state=42)

### Default Models (argmax decision rule)

| Model | Accuracy | Macro Recall | Class 1 Recall | Class 1 Precision | Class 1 F1 |
|-------|----------|--------------|----------------|-------------------|------------|
| XGBoost | 83.4% | 82.1% | 85.7% | 77.3% | 0.813 |
| SVM | 81.2% | 79.4% | 82.8% | 78.3% | 0.805 |
| PyTorch MLP | 82.1% | 80.9% | 83.1% | 80.8% | 0.819 |

### Tuned Models (safety-oriented threshold for Class 1)

| Model | Threshold | Accuracy | Class 1 Recall | Class 1 Precision | Class 1 F1 |
|-------|:---------:|----------|----------------|-------------------|------------|
| XGBoost | 0.25 | 82.8% | **90.4%** | 71.7% | 0.800 |
| SVM | 0.10 | 74.5% | **97.8%** | 54.1% | 0.697 |
| PyTorch MLP | 0.10 | 81.8% | **94.0%** | 70.7% | 0.807 |

## 2. ROC / AUC Analysis (Class 1 — Danger, XGBoost)

| Metric | Value |
|--------|-------|
| AUC | **0.9686** |
| Default FPR | 0.0731 |
| Default TPR (Recall) | 0.8567 |
| Tuned FPR | 0.1037 |
| Tuned TPR (Recall) | 0.9045 |

## 3. Ablation Study (Doppler Features)

| Model Variant | Accuracy | Class 1 Recall | Class 1 Precision |
|---------------|----------|----------------|-------------------|
| Full Model (132 features) | 83.4% | 85.7% | 77.3% |
| Reduced Model (130 features, no Doppler) | 83.4% | 86.6% | — |
| **Change** | 0.0% | **+0.96%** | — |

**Conclusion:** Doppler-inspired features (spectral centroid slope + centroid diff std) have **no significant impact** on performance. The MFCC and standard spectral features already capture the necessary information.

## 4. Learning Curve (XGBoost)

| Data % | Samples | Accuracy | Class 1 Recall |
|:------:|:-------:|:--------:|:--------------:|
| 10% | 557 | 78.3% | 79.3% |
| 20% | 1,114 | 79.8% | 82.5% |
| 30% | 1,671 | 82.1% | 84.4% |
| 40% | 2,229 | 81.2% | 85.0% |
| 50% | 2,786 | 82.0% | 86.9% |
| 60% | 3,343 | 82.4% | 85.4% |
| 70% | 3,901 | 83.8% | 86.9% |
| 80% | 4,458 | 82.9% | 87.9% |
| 90% | 5,015 | 83.4% | 86.0% |
| 100% | 5,573 | 83.4% | 85.7% |

**Conclusion:** Moderate improvement from 50% to 100% data (+1.43% accuracy). Model converges reasonably by 50% but full data adds robustness.

## 5. File-Level Split (No Data Leakage)

| Metric | Value |
|--------|-------|
| Train Windows | 44,494 |
| Val Windows | 11,356 |
| Accuracy | **86.2%** |

**Conclusion:** The windowed file-level split approach achieves higher accuracy (86.2%) by using more data through overlapping windows, while avoiding leakage between train and validation sets.

## 6. Event-Level Safety Evaluation (Hybrid Logic)

Parameters: `c1_threshold=0.25, risk_threshold=0.4, consecutive=1`

| Class | Precision | Recall | F1 | Support |
|-------|----------|--------|----|---------|
| Safe | 0.888 | 0.655 | 0.754 | 1,080 |
| Danger | 0.376 | 0.717 | 0.493 | 314 |

Confusion Matrix:
- TN (Safe correct): 707
- FP (False alarm): 373
- FN (Missed danger): 89
- TP (Danger detected): 225

## 7. Unseen Test Set (New Environment)

**Test data:** 384 valid samples from a different recording environment

| Model | Accuracy | Class 1 Recall | Class 1 Precision | TP | FN |
|-------|----------|----------------|-------------------|----|----|
| XGBoost | 73.7% | 78.4% | 69.7% | 69 | 19 |
| SVM | 69.8% | 77.3% | 67.3% | 68 | 20 |
| PyTorch MLP | N/A | — | — | — | — |

**Per-class breakdown (XGBoost):**

| Class | Precision | Recall | F1 | Support |
|-------|----------|--------|----|---------|
| 0 | 0.671 | 0.750 | 0.708 | 68 |
| 1 (Danger) | 0.697 | 0.784 | 0.738 | 88 |
| 2 | 0.780 | 0.568 | 0.657 | 81 |
| 3 | 0.750 | 0.716 | 0.733 | 67 |
| 4 | 0.802 | 0.863 | 0.831 | 80 |

## 8. Final Recommendations

| Use Case | Recommended Model | Reason |
|----------|-------------------|--------|
| **General deployment** | XGBoost (Tuned, t=0.25) | Best accuracy-recall balance (82.8% / 90.4%) |
| **Maximum safety** | SVM (Tuned, t=0.10) | Highest recall (97.8%), but more false alarms |
| **Future end-to-end** | PyTorch MLP | Strong baseline for deep learning extension |

## 9. Key Insights

1. **Safety-oriented threshold tuning works:** All three models exceeded the 90% recall target for Class 1 after threshold adjustment, at the cost of increased false positives — a worthwhile trade-off for pedestrian safety applications.

2. **XGBoost is the best all-rounder:** It maintains the best accuracy while achieving the 90% recall target, making it the recommended choice for general deployment.

3. **Generalization gap is real:** The ~9% accuracy drop on unseen data highlights the importance of environment diversity in the training set. Future work should focus on domain adaptation.

4. **Doppler features are redundant:** The ablation study showed that the two Doppler-inspired features do not contribute meaningfully. The standard MFCC + spectral feature set is sufficient.

5. **Data efficiency is good:** The model converges by 50% of the data (~2,800 samples), but the full dataset provides stability benefits.
