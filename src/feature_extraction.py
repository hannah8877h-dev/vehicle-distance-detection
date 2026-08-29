"""
Feature extraction module for acoustic vehicle distance detection.

This module provides reusable functions to extract the same 132 acoustic
features used in the training notebook, so the same preprocessing can be
applied consistently during inference.

Features extracted (132 total):
    - MFCC (mean + std):        20 * 2 = 40
    - Delta MFCC (mean + std):  20 * 2 = 40
    - Delta2 MFCC (mean + std): 20 * 2 = 40
    - Spectral Centroid (m+s):  2
    - Spectral Bandwidth (m+s): 2
    - Spectral Rolloff (m+s):   2
    - Zero Crossing Rate (m+s): 2
    - RMS Energy (m+s):         2
    - Doppler slope:            1
    - Centroid diff std:        1
    -----------------------------------
    Total:                      132

Author: Hananeh Shabestani, Setayesh Zaheri Azam
License: MIT
"""

import os
import numpy as np
import librosa

# Default audio parameters (must match training)
DEFAULT_SR = 22050
DEFAULT_N_MFCC = 20
DEFAULT_TRIM_TOP_DB = 20


def extract_features(signal, sr=DEFAULT_SR, n_mfcc=DEFAULT_N_MFCC, trim_top_db=DEFAULT_TRIM_TOP_DB):
    """
    Extract 132 acoustic features from a 1-D audio signal (numpy array).

    Parameters
    ----------
    signal : np.ndarray
        1-D audio time series.
    sr : int
        Sample rate of the signal (default 22050).
    n_mfcc : int
        Number of MFCC coefficients (default 20).
    trim_top_db : float
        Silence trimming threshold in dB (default 20).

    Returns
    -------
    np.ndarray of shape (132,) or None
        Feature vector, or None if the signal is empty.
    """
    if signal is None or len(signal) == 0:
        return None

    # Peak normalization
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val

    # Silence trimming
    signal, _ = librosa.effects.trim(signal, top_db=trim_top_db)

    if len(signal) == 0:
        return None

    features = []

    # ---- MFCC + Delta + Delta2 ----
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    for feat in (mfcc, delta, delta2):
        features.extend(np.mean(feat, axis=1))
        features.extend(np.std(feat, axis=1))

    # ---- Spectral Features ----
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sr)[0]
    zcr = librosa.feature.zero_crossing_rate(signal)[0]
    rms = librosa.feature.rms(y=signal)[0]

    for feat in (centroid, bandwidth, rolloff, zcr, rms):
        features.append(np.mean(feat))
        features.append(np.std(feat))

    # ---- Doppler-inspired features ----
    slope = np.polyfit(range(len(centroid)), centroid, 1)[0]
    centroid_diff_std = np.std(np.diff(centroid))

    features.append(slope)
    features.append(centroid_diff_std)

    return np.array(features)


def extract_features_from_file(file_path, sr=DEFAULT_SR, n_mfcc=DEFAULT_N_MFCC):
    """
    Load an audio file and extract features.

    Parameters
    ----------
    file_path : str
        Path to a .wav (or other librosa-supported) audio file.
    sr : int
        Target sample rate (audio will be resampled).
    n_mfcc : int
        Number of MFCC coefficients.

    Returns
    -------
    np.ndarray of shape (132,) or None
    """
    y, sr = librosa.load(file_path, sr=sr)
    return extract_features(y, sr=sr, n_mfcc=n_mfcc)


def extract_label_from_filename(filename, fmt='suffix'):
    """
    Extract the class label from a filename.

    Two formats are supported:
        - 'suffix': number_class.wav      (e.g., 12_3.wav -> 3)   [training set]
        - 'prefix': ClassNumber_x.wav     (e.g., 23_1.wav -> 23)  [test set]

    Parameters
    ----------
    filename : str
        The audio filename.
    fmt : str
        'suffix' or 'prefix'.

    Returns
    -------
    int or None
    """
    import re
    name = os.path.splitext(filename)[0]
    if fmt == 'suffix':
        match = re.search(r'_(\d)', name)
        return int(match.group(1)) if match else None
    elif fmt == 'prefix':
        try:
            return int(name.split('_')[0])
        except (ValueError, IndexError):
            return None
    else:
        raise ValueError(f"Unknown format: {fmt}. Use 'suffix' or 'prefix'.")


def process_folder(folder_path, label_format='suffix', sr=DEFAULT_SR, n_mfcc=DEFAULT_N_MFCC):
    """
    Process all .wav files in a folder and return features + labels.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing .wav files.
    label_format : str
        'suffix' or 'prefix' (see extract_label_from_filename).
    sr : int
        Sample rate.
    n_mfcc : int
        Number of MFCC coefficients.

    Returns
    -------
    X : np.ndarray of shape (n_samples, 132)
    y : np.ndarray of shape (n_samples,)
    filenames : list of str
    """
    from tqdm import tqdm

    X, y, filenames = [], [], []
    files = sorted([f for f in os.listdir(folder_path) if f.endswith('.wav')])

    for file in tqdm(files, desc="Extracting features"):
        label = extract_label_from_filename(file, fmt=label_format)
        if label is None:
            continue

        feats = extract_features_from_file(os.path.join(folder_path, file), sr=sr, n_mfcc=n_mfcc)
        if feats is None:
            continue

        X.append(feats)
        y.append(label)
        filenames.append(file)

    return np.array(X), np.array(y), filenames


if __name__ == '__main__':
    # Quick self-test
    print(f"Feature Extraction Module")
    print(f"Expected feature count: 132")
    print(f"Default sample rate: {DEFAULT_SR}")
    print(f"Default n_mfcc: {DEFAULT_N_MFCC}")
    print(f"\nUsage:")
    print(f"  from feature_extraction import extract_features_from_file")
    print(f"  features = extract_features_from_file('audio.wav')")
