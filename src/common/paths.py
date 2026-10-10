"""Portable filesystem path resolution utilities for ADHD-200 Connectomics research.

Provides deterministic resolution of repository directories across four artifact classes:
1. Repository artifacts: canonical results and configs tracked in Git (<results_root>, <configs_root>).
2. External source data: raw and preprocessed neuroimaging/phenotypic data (<data_root>).
3. Regenerable intermediates: generated parquet, npy arrays, and workflow stages (<intermediate_root>).
4. Model checkpoints: pretrained weights and deep model weights (<checkpoints_root>).

Never creates directories upon module import.
"""

import os
from pathlib import Path


def find_repo_root() -> Path:
    """Locate the repository root deterministically from current working directory or parents."""
    env_root = os.environ.get("ADHD200_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    current = Path.cwd().resolve()
    for candidate in (current, *current.parents):
        if (
            (candidate / "pyproject.toml").exists()
            and (candidate / "results").exists()
            and (candidate / "notebooks").exists()
        ):
            return candidate
    # Fallback to file-relative resolution
    return Path(__file__).resolve().parents[2]


def project_root() -> Path:
    """Return the absolute Path to the repository root directory."""
    return find_repo_root()


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


def intermediate_root() -> Path:
    """Return the Path to regenerable intermediate data directory.
    
    Prefers ADHD200_WORK_DIR, ADHD200_INTERMEDIATE_DIR, or ADHD200_WORK_ROOT if set;
    otherwise defaults to <data_root>/intermediate.
    """
    env_intermediate = (
        os.environ.get("ADHD200_WORK_DIR")
        or os.environ.get("ADHD200_INTERMEDIATE_DIR")
        or os.environ.get("ADHD200_WORK_ROOT")
    )
    if env_intermediate:
        return Path(env_intermediate).resolve()
    return data_root() / "intermediate"


def workflow2_root() -> Path:
    """Return the Path to intermediate workflow2 directory."""
    return intermediate_root() / "workflow2"


def v6_root() -> Path:
    """Return the Path to intermediate workflow2/v6 directory."""
    return workflow2_root() / "v6"


def v7_root() -> Path:
    """Return the Path to intermediate workflow2/v7 directory."""
    return workflow2_root() / "v7"


def v8_root() -> Path:
    """Return the Path to intermediate workflow2/v8 directory."""
    return workflow2_root() / "v8"


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
    
    Prefers ADHD200_CHECKPOINTS_DIR or ADHD200_CHECKPOINT_DIR if set;
    otherwise defaults to <data_root>/checkpoints (or <artifacts_root>/checkpoints).
    """
    env_ckpt = (
        os.environ.get("ADHD200_CHECKPOINTS_DIR")
        or os.environ.get("ADHD200_CHECKPOINT_DIR")
        or os.environ.get("ADHD200_CHECKPOINT_ROOT")
        or os.environ.get("ADHD200_CHECKPOINTS_ROOT")
    )
    if env_ckpt:
        return Path(env_ckpt).resolve()
    if (artifacts_root() / "checkpoints").exists() and not (data_root() / "checkpoints").exists():
        return artifacts_root() / "checkpoints"
    return data_root() / "checkpoints"


def validate_input_path(path: Path, description: str = "Required input") -> Path:
    """Verify that an input path exists, raising an informative FileNotFoundError if absent.
    
    Args:
        path: Path object to check.
        description: Human-readable description of the required resource.
        
    Returns:
        The validated Path object.
        
    Raises:
        FileNotFoundError: If the path does not exist, with instructions to consult docs/reproduction.md.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{description} not found at expected path: {path}\n"
            f"Please ensure required dataset files are installed. "
            f"Set the ADHD200_DATA_DIR environment variable or see docs/reproduction.md for instructions."
        )
    return path


# Canonical module-level path constants
REPO_ROOT = project_root()
DATA_ROOT = data_root()
RESULTS_ROOT = results_root()
INTERMEDIATE_ROOT = intermediate_root()
CHECKPOINT_ROOT = checkpoints_root()
WORKFLOW2_ROOT = workflow2_root()
V6_ROOT = v6_root()
V7_ROOT = v7_root()
V8_ROOT = v8_root()
