"""Unit tests asserting that core repository modules can be imported without missing dependencies."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def test_import_common_paths():
    """Verify common.paths module imports and exposes functions."""
    from common.paths import (
        artifacts_root,
        checkpoints_root,
        data_root,
        project_root,
        results_root,
        validate_input_path,
    )
    assert callable(project_root)
    assert callable(data_root)
    assert callable(results_root)
    assert callable(checkpoints_root)
    assert callable(artifacts_root)
    assert callable(validate_input_path)


def test_import_exp02_config():
    """Verify Experiment 2 configuration imports cleanly."""
    from exp02.config import DENSITY, N_ROIS
    assert N_ROIS == 190
    assert DENSITY == 0.20


def test_import_exp02_graph_utils():
    """Verify Experiment 2 graph utilities import when PyG is present."""
    pytest.importorskip("torch_geometric", reason="PyG required for graph_utils")
    from exp02.graph_utils import prepare_graphs
    assert callable(prepare_graphs)


def test_import_exp07_utils_config():
    """Verify Experiment 7 config and data loader import cleanly."""
    from exp07.utils.config import N_QUBITS, N_ROIS, NODE_FEATURE_DIM
    assert N_ROIS == 116
    assert NODE_FEATURE_DIM == 117
    assert N_QUBITS == 6

    from exp07.utils.data_loader import load_data, split_data
    assert callable(load_data)
    assert callable(split_data)


def test_import_exp07_graph_and_training_utils():
    """Verify Experiment 7 PyG utilities import when PyG is present."""
    pytest.importorskip("torch_geometric", reason="PyG required for exp07 graph utilities")
    from exp07.utils.graph_utils import prepare_graphs
    assert callable(prepare_graphs)

    from exp07.utils.training_utils import train_one_epoch, validate
    assert callable(train_one_epoch)
    assert callable(validate)


def test_import_exp07_classical_models():
    """Verify Classical GCN model imports and instantiates when PyG is present."""
    pytest.importorskip("torch_geometric", reason="PyG required for ClassicalGCN")
    from exp07.classical_models.model_classical_gcn import ClassicalGCN
    model = ClassicalGCN(input_dim=117, hidden_dim=32)
    assert model is not None
