#!/usr/bin/env python
# ============================================================================
# DATA LOADER FOR EXPERIMENT 7 (Track B)
# Loads real AAL-116 functional connectivity matrices and diagnostic labels.
# Requires authentic external data artifacts; fails loudly if data are absent.
# No synthetic or random data fallback is permitted in research code.
# ============================================================================

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from exp07.utils.config import DATA_DIR, N_ROIS


def load_data(data_dir: Optional[Path] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load real Track B dataset artifacts: clean-labeled and pseudo-labeled cohorts.
    
    Args:
        data_dir: Optional override for the dataset directory. Defaults to DATA_DIR.
        
    Returns:
        X_full: Combined FC feature array of shape (N_total, 6670)
        X_clean: Clean-labeled FC feature array of shape (N_clean, 6670)
        y: Diagnostic labels array of shape (N_total,) (0=TDC, 1=ADHD)
        subjects: Array of subject identifiers
        
    Raises:
        FileNotFoundError: If authentic dataset files are missing.
        ValueError: If file contents fail schema or shape validation.
    """
    target_dir = Path(data_dir) if data_dir is not None else DATA_DIR
    fc_cache_file = target_dir / "aal116_fc_features.npz"
    labels_file = target_dir / "aal116_labels.csv"

    if not fc_cache_file.exists() or not labels_file.exists():
        raise FileNotFoundError(
            f"Required Experiment 7 dataset artifacts not found in: {target_dir}\n"
            f"Expected files:\n"
            f"  - FC features array: {fc_cache_file}\n"
            f"  - Diagnostic labels: {labels_file}\n\n"
            f"Derived ADHD-200 connectivity matrices are external to this public repository "
            f"due to file-size constraints (>100 MiB). Please acquire or generate the required "
            f"AAL-116 artifacts and specify their location via the ADHD200_DATA_ROOT environment "
            f"variable or the data_dir argument. See data/README.md for full instructions."
        )

    # Load authentic arrays
    data = np.load(fc_cache_file)
    if "X_full" not in data:
        raise ValueError(f"Corrupt artifact {fc_cache_file}: missing 'X_full' array key.")

    X_full = data["X_full"]
    X_clean = data.get("X_clean", None)

    # Validate feature dimension (116 * 115 / 2 = 6670)
    expected_dim = N_ROIS * (N_ROIS - 1) // 2
    if X_full.ndim != 2 or X_full.shape[1] != expected_dim:
        raise ValueError(
            f"Invalid FC feature matrix shape {X_full.shape}. "
            f"Expected (N, {expected_dim}) for AAL-116 parcellation."
        )

    labels_df = pd.read_csv(labels_file)
    if "diagnosis" not in labels_df.columns or "subject_id" not in labels_df.columns:
        raise ValueError(
            f"Invalid labels manifest {labels_file}: required columns 'diagnosis' and 'subject_id' not found."
        )

    if len(labels_df) != len(X_full):
        raise ValueError(
            f"Sample count mismatch: {len(labels_df)} entries in {labels_file} "
            f"vs {len(X_full)} samples in {fc_cache_file}."
        )

    y = labels_df["diagnosis"].values.astype(np.int64)
    subjects = labels_df["subject_id"].values

    unique_labels = set(np.unique(y))
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"Invalid diagnostic labels found: {unique_labels}. Expected binary labels in {{0, 1}}.")

    if X_clean is None:
        # If not explicitly keyed, clean cohort is derived from manifest metadata if available
        if "is_clean" in labels_df.columns:
            clean_mask = labels_df["is_clean"].values.astype(bool)
            X_clean = X_full[clean_mask]
        else:
            # Documented provenance: first 162 subjects constitute the clean cohort in the executed run
            n_clean = min(162, len(X_full))
            X_clean = X_full[:n_clean]

    return X_full, X_clean, y, subjects


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    subjects: Optional[np.ndarray] = None,
    test_size: int = 33,
    val_size: float = 0.15,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Split data into training, validation, and held-out test partitions.
    Preserves the documented experimental split mechanism where the test set
    is drawn strictly from the clean-labeled cohort (test_size=33).
    
    Args:
        X: Full feature array of shape (N, D)
        y: Full label array of shape (N,)
        subjects: Optional subject identifier array
        test_size: Number of clean subjects held out for test evaluation (default: 33)
        val_size: Fraction of remaining training data allocated for validation (default: 0.15)
        random_state: Deterministic RNG seed (default: 42)
        
    Returns:
        Dictionary containing partition arrays: X_train, y_train, X_val, y_val, X_test, y_test
    """
    n_clean = min(162, len(X))
    clean_indices = np.arange(n_clean)
    pseudo_indices = np.arange(n_clean, len(X)) if len(X) > n_clean else np.array([], dtype=int)

    rng = np.random.RandomState(random_state)
    shuffled_clean = rng.permutation(clean_indices)

    # 33 held-out test subjects
    test_idx = shuffled_clean[:test_size]
    clean_train_val_idx = shuffled_clean[test_size:]

    # Combine remaining clean subjects with pseudo-labeled cohort for training/val
    train_val_idx = np.concatenate([clean_train_val_idx, pseudo_indices])

    n_val = int(len(train_val_idx) * val_size)
    shuffled_tv = rng.permutation(train_val_idx)
    val_idx = shuffled_tv[:n_val]
    train_idx = shuffled_tv[n_val:]

    return {
        "X_train": X[train_idx],
        "y_train": y[train_idx],
        "X_val": X[val_idx],
        "y_val": y[val_idx],
        "X_test": X[test_idx],
        "y_test": y[test_idx],
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
    }


if __name__ == "__main__":
    try:
        load_data()
    except FileNotFoundError as err:
        print(f"[Expected behavior when real data is absent]\n{err}")
