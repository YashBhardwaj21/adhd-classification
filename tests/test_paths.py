"""
Tests for common.paths module: path resolution, environment variable overrides, and path validation.
"""

import shutil
from pathlib import Path

import pytest

from common.paths import (
    artifacts_root,
    checkpoints_root,
    data_root,
    project_root,
    results_root,
    validate_input_path,
)


@pytest.fixture
def local_tmp_dir():
    """Provides a safe temporary directory within the workspace to avoid Windows temp permission issues."""
    tmp_path = project_root() / "tests" / ".tmp_test_paths"
    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)
    tmp_path.mkdir(parents=True, exist_ok=True)
    yield tmp_path
    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_project_root():
    root = project_root()
    assert isinstance(root, Path)
    assert (root / "pyproject.toml").exists()
    assert (root / "src").exists()


def test_default_paths():
    root = project_root()
    assert data_root() == root / "data"
    assert results_root() == root / "results"
    assert artifacts_root() == root / "artifacts"
    assert checkpoints_root() == root / "artifacts" / "checkpoints"


def test_env_var_override_data_root(monkeypatch, local_tmp_dir):
    custom_data = local_tmp_dir / "custom_data"
    custom_data.mkdir()
    monkeypatch.setenv("ADHD200_DATA_DIR", str(custom_data))
    assert data_root() == custom_data


def test_env_var_override_results_root(monkeypatch, local_tmp_dir):
    custom_results = local_tmp_dir / "custom_results"
    custom_results.mkdir()
    monkeypatch.setenv("ADHD200_RESULTS_DIR", str(custom_results))
    assert results_root() == custom_results


def test_validate_input_path_success(local_tmp_dir):
    existing_file = local_tmp_dir / "sample.csv"
    existing_file.write_text("a,b,c\n1,2,3\n", encoding="utf-8")
    validated = validate_input_path(existing_file, description="Sample test file")
    assert validated == existing_file


def test_validate_input_path_missing(local_tmp_dir):
    missing_file = local_tmp_dir / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError) as exc_info:
        validate_input_path(missing_file, description="Missing test file")
    assert "Missing test file not found" in str(exc_info.value)
