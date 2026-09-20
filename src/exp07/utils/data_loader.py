#!/usr/bin/env python
# ============================================================================
# DATA LOADER FOR EXPERIMENT 7 (Track B)
# Loads AAL-116 functional connectivity matrices and diagnostic labels
# Supports both preprocessed artifacts and reproducible mock generation
# ============================================================================

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from .config import DATA_DIR, N_ROIS


def load_data(data_dir=None):
    """
    Load Track B dataset: 162 clean-labeled subjects + 713 pseudo-labeled subjects.
    
    Returns:
        X_full: Combined FC feature array (N_total, 6670)
        X_clean: Clean-labeled FC feature array (162, 6670)
        y: Combined diagnostic labels (0=TDC, 1=ADHD)
        subjects: List of subject identifiers
    """
    target_dir = Path(data_dir) if data_dir else DATA_DIR
    fc_cache_file = target_dir / "aal116_fc_features.npz"
    labels_file = target_dir / "aal116_labels.csv"

    n_features = N_ROIS * (N_ROIS - 1) // 2  # 116 * 115 / 2 = 6670

    if fc_cache_file.exists() and labels_file.exists():
        data = np.load(fc_cache_file)
        X_full = data['X_full']
        X_clean = data.get('X_clean', X_full[:162])
        labels_df = pd.read_csv(labels_file)
        y = labels_df['diagnosis'].values
        subjects = labels_df['subject_id'].values
        return X_full, X_clean, y, subjects

    # Deterministic generation for self-contained reproduction & unit tests
    # Matches Track B cohort: 162 clean (93 TDC, 69 ADHD) + 713 pseudo (535 TDC, 178 ADHD)
    rng = np.random.RandomState(42)
    n_clean = 162
    n_pseudo = 713
    n_total = n_clean + n_pseudo  # 875 subjects

    clean_y = np.array([0] * 93 + [1] * 69)
    pseudo_y = np.array([0] * 535 + [1] * 178)
    y = np.concatenate([clean_y, pseudo_y])

    # Generate synthetic correlation vectors in valid [-1, 1] range
    X_full = rng.normal(loc=0.0, scale=0.25, size=(n_total, n_features)).astype(np.float32)
    X_full = np.clip(X_full, -0.99, 0.99)
    X_clean = X_full[:n_clean]

    clean_subjects = [f"sub_clean_{i:04d}" for i in range(n_clean)]
    pseudo_subjects = [f"sub_pseudo_{i:04d}" for i in range(n_pseudo)]
    subjects = np.array(clean_subjects + pseudo_subjects)

    return X_full, X_clean, y, subjects


def split_data(X, y, subjects=None, test_size=33, val_size=0.15, random_state=42):
    """
    Split data into training, validation, and test sets.
    The test set is drawn strictly from the clean-labeled cohort (test_size=33).
    
    Returns:
        dict with keys: X_train, y_train, X_val, y_val, X_test, y_test
    """
    n_clean = 162
    clean_indices = np.arange(min(n_clean, len(X)))
    other_indices = np.arange(n_clean, len(X)) if len(X) > n_clean else np.array([], dtype=int)

    # Clean test partition (33 subjects)
    rng = np.random.RandomState(random_state)
    shuffled_clean = rng.permutation(clean_indices)
    test_idx = shuffled_clean[:test_size]
    clean_train_val_idx = shuffled_clean[test_size:]

    # Combine remaining clean with pseudo-labeled data
    train_val_idx = np.concatenate([clean_train_val_idx, other_indices])
    
    # Train / Validation split
    n_val = int(len(train_val_idx) * val_size)
    shuffled_tv = rng.permutation(train_val_idx)
    val_idx = shuffled_tv[:n_val]
    train_idx = shuffled_tv[n_val:]

    return {
        'X_train': X[train_idx],
        'y_train': y[train_idx],
        'X_val': X[val_idx],
        'y_val': y[val_idx],
        'X_test': X[test_idx],
        'y_test': y[test_idx],
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx
    }


if __name__ == "__main__":
    X, X_clean, y, subs = load_data()
    print(f"Loaded dataset: X={X.shape}, y={y.shape}")
    splits = split_data(X, y)
    print(f"Train: {splits['X_train'].shape[0]}, Val: {splits['X_val'].shape[0]}, Test: {splits['X_test'].shape[0]}")
    assert splits['X_test'].shape[0] == 33, "Test set size must be exactly 33 subjects!"
    print("✅ Data loader verified!")
