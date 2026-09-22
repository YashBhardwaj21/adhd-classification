#!/usr/bin/env python
# ============================================================================
# DATA LOADER FOR EXPERIMENT 7 (Track B)
# Historical Protocol: AAL-116 FC matrices and diagnostic labels
# Historical contract: X_combined_full.npy, y_combined.npy, subjects_combined.npy
# Cohort partition:
#   - 162 clean-labelled subjects: 103 train, 26 validation, 33 held-out test
#   - 713 pseudo-labelled subjects: allocated strictly to training
#
# Note: The original combined numpy arrays exceed distribution quotas and are
# not stored in this public repository. This loader preserves the exact
# historical execution contract and raises FileNotFoundError when called
# without the external arrays present.
# ============================================================================

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from exp07.utils.config import DATA_DIR, N_ROIS, RANDOM_SEED


def load_data(
    data_dir: Optional[Path] = None,
    use_reduced: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load historical Track B dataset arrays.

    Args:
        data_dir: Directory containing historical array artifacts. Defaults to DATA_DIR.
        use_reduced: If True, load X_combined_reduced.npy instead of X_combined_full.npy.

    Returns:
        X: Combined FC feature array of shape (875, 6670)
        y: Combined diagnostic labels array of shape (875,)
        subjects: Array of subject identifiers of shape (875,)

    Raises:
        FileNotFoundError: If authentic historical dataset files are missing.
        ValueError: If array dimensions or sample counts do not match the historical protocol.
    """
    target_dir = Path(data_dir) if data_dir is not None else DATA_DIR

    x_filename = "X_combined_reduced.npy" if use_reduced else "X_combined_full.npy"
    x_path = target_dir / x_filename
    y_path = target_dir / "y_combined.npy"
    sub_path = target_dir / "subjects_combined.npy"

    missing = [str(p) for p in [x_path, y_path, sub_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Required historical Experiment 7 dataset artifacts not found in: {target_dir}\n"
            f"Missing file(s): {', '.join(missing)}\n\n"
            f"Historical execution contract requires:\n"
            f"  - {x_filename} (FC correlation features)\n"
            f"  - y_combined.npy (diagnostic labels: 162 clean + 713 pseudo)\n"
            f"  - subjects_combined.npy (subject identifiers)\n\n"
            f"Provenance Status: The original large training arrays are no longer available in this "
            f"public repository. The reported test results (N=33) are archived in results/exp07/checkpoint_analysis.json. "
            f"This experiment cannot currently be rerun end-to-end from scratch without external restoration of "
            f"these arrays. See docs/provenance.md and docs/reproduction.md for full details."
        )

    X = np.load(x_path)
    y = np.load(y_path)
    subjects = np.load(sub_path)

    expected_dim = N_ROIS * (N_ROIS - 1) // 2
    if not use_reduced and (X.ndim != 2 or X.shape[1] != expected_dim):
        raise ValueError(
            f"Invalid FC feature matrix shape {X.shape}. "
            f"Expected (N, {expected_dim}) for AAL-116 parcellation."
        )

    if len(X) != len(y) or len(X) != len(subjects):
        raise ValueError(
            f"Sample count mismatch: X={len(X)}, y={len(y)}, subjects={len(subjects)}."
        )

    return X, y, subjects


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    subjects: Optional[np.ndarray] = None,
    n_clean: int = 162,
    n_train_clean: int = 103,
    n_val_clean: int = 26,
    n_test_clean: int = 33,
    random_state: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Partition dataset according to the historical Experiment 7 protocol:
      - 162 clean subjects: partitioned into train (103), validation (26), test (33)
      - 713 pseudo-labelled subjects (indices >= 162): allocated strictly to training

    Args:
        X: Combined feature array of shape (N, D)
        y: Combined label array of shape (N,)
        subjects: Optional array of subject identifiers
        n_clean: Total number of clean-labelled subjects (historical default: 162)
        n_train_clean: Number of clean subjects in training set (default: 103)
        n_val_clean: Number of clean subjects in validation set (default: 26)
        n_test_clean: Number of clean subjects in held-out test set (default: 33)
        random_state: Random seed for clean subject permutation (default: 42)

    Returns:
        Dictionary containing partitioned arrays and indices:
          X_train, y_train, X_val, y_val, X_test, y_test,
          train_idx, val_idx, test_idx
    """
    total_clean_split = n_train_clean + n_val_clean + n_test_clean
    if total_clean_split != n_clean:
        raise ValueError(
            f"Clean split counts ({n_train_clean} + {n_val_clean} + {n_test_clean} = {total_clean_split}) "
            f"must sum to n_clean ({n_clean})."
        )

    n_total = len(X)
    if n_total < n_clean:
        raise ValueError(
            f"Input dataset has {n_total} samples, fewer than the required {n_clean} clean subjects."
        )

    clean_indices = np.arange(n_clean)
    pseudo_indices = np.arange(n_clean, n_total)

    rng = np.random.RandomState(random_state)
    shuffled_clean = rng.permutation(clean_indices)

    # Clean cohort partitions
    clean_train_idx = shuffled_clean[:n_train_clean]
    val_idx = shuffled_clean[n_train_clean:n_train_clean + n_val_clean]
    test_idx = shuffled_clean[n_train_clean + n_val_clean:n_train_clean + n_val_clean + n_test_clean]

    # Pseudo-labelled subjects go strictly to training
    train_idx = np.concatenate([clean_train_idx, pseudo_indices]) if len(pseudo_indices) > 0 else clean_train_idx

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
        "n_clean": n_clean,
        "n_pseudo": len(pseudo_indices),
    }


if __name__ == "__main__":
    try:
        load_data()
    except FileNotFoundError as err:
        print(f"[Expected behavior when historical arrays are absent]\n{err}")
