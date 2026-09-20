"""Unit tests asserting that core repository modules can be imported without missing dependencies."""

import pytest
import sys
from pathlib import Path

# Add project root and src subdirectories to python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def test_import_exp02_modules():
    """Verify Experiment 2 graph and config modules import cleanly."""
    from src.exp02.config import N_ROIS, DENSITY
    assert N_ROIS == 190
    assert DENSITY == 0.20
    
    from src.exp02.graph_utils import prepare_graphs
    assert callable(prepare_graphs)


def test_import_exp07_utils():
    """Verify Experiment 7 utilities import cleanly."""
    from src.exp07.utils.config import N_ROIS, NODE_FEATURE_DIM, N_QUBITS
    assert N_ROIS == 116
    assert NODE_FEATURE_DIM == 117
    assert N_QUBITS == 6

    from src.exp07.utils.data_loader import load_data, split_data
    assert callable(load_data)
    assert callable(split_data)

    from src.exp07.utils.graph_utils import prepare_graphs
    assert callable(prepare_graphs)

    from src.exp07.utils.training_utils import train_one_epoch, validate
    assert callable(train_one_epoch)
    assert callable(validate)


def test_import_exp07_models():
    """Verify Classical GCN model imports and instantiates cleanly."""
    from src.exp07.classical_models.model_classical_gcn import ClassicalGCN
    model = ClassicalGCN(input_dim=117, hidden_dim=32)
    assert model is not None
