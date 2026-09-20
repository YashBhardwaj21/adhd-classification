"""Portable filesystem path resolution utilities for ADHD-200 Connectomics research.

Provides deterministic resolution of repository directories, supporting optional
environment variable overrides for external data and compute configurations.
Never creates directories upon module import.
"""

import os
from pathlib import Path


def project_root() -> Path:
    """Return the absolute Path to the repository root directory."""
    env_root = os.environ.get("ADHD200_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    # Resolve relative to this file: src/common/paths.py -> parent x 2 = repo root
    return Path(__file__).resolve().parents[2]


def data_root() -> Path:
    """Return the Path to the dataset directory.
    
    Prefers ADHD200_DATA_DIR or ADHD200_DATA_ROOT if set;
    otherwise defaults to <project_root>/data.
    """
    env_data = os.environ.get("ADHD200_DATA_DIR") or os.environ.get("ADHD200_DATA_ROOT")
    if env_data:
        return Path(env_data).resolve()
    return project_root() / "data"


def results_root() -> Path:
    """Return the Path to the results directory.
    
    Prefers ADHD200_RESULTS_DIR or ADHD200_RESULTS_ROOT if set;
    otherwise defaults to <project_root>/results.
    """
    env_results = os.environ.get("ADHD200_RESULTS_DIR") or os.environ.get("ADHD200_RESULTS_ROOT")
    if env_results:
        return Path(env_results).resolve()
    return project_root() / "results"


def artifacts_root() -> Path:
    """Return the Path to the runtime artifacts directory.
    
    Prefers ADHD200_ARTIFACTS_DIR or ADHD200_ARTIFACTS_ROOT if set;
    otherwise defaults to <project_root>/artifacts.
    """
    env_artifacts = os.environ.get("ADHD200_ARTIFACTS_DIR") or os.environ.get("ADHD200_ARTIFACTS_ROOT")
    if env_artifacts:
        return Path(env_artifacts).resolve()
    return project_root() / "artifacts"


def checkpoints_root() -> Path:
    """Return the Path to the model checkpoints directory.
    
    Prefers ADHD200_CHECKPOINTS_DIR or ADHD200_CHECKPOINT_ROOT if set;
    otherwise defaults to <artifacts_root>/checkpoints.
    """
    env_ckpt = (
        os.environ.get("ADHD200_CHECKPOINTS_DIR")
        or os.environ.get("ADHD200_CHECKPOINT_ROOT")
        or os.environ.get("ADHD200_CHECKPOINTS_ROOT")
    )
    if env_ckpt:
        return Path(env_ckpt).resolve()
    return artifacts_root() / "checkpoints"


def validate_input_path(path: Path, description: str = "Required input") -> Path:
    """Verify that an input path exists, raising an informative FileNotFoundError if absent.
    
    Args:
        path: Path object to check.
        description: Human-readable description of the required resource.
        
    Returns:
        The validated Path object.
        
    Raises:
        FileNotFoundError: If the path does not exist, with instructions to consult data/README.md.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found at expected path: {path}\n"
            f"Please ensure required dataset files are installed. "
            f"Set the ADHD200_DATA_ROOT environment variable or see data/README.md for instructions."
        )
    return path
