"""Unit tests for graph construction and node feature tensor generation."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Require PyTorch Geometric for graph builder tests
pyg = pytest.importorskip("torch_geometric", reason="PyTorch Geometric (torch_geometric) required for graph tensor tests")
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from common.paths import data_root
from exp07.utils.config import N_ROIS, NODE_FEATURE_DIM
from exp07.utils.graph_utils import prepare_graphs as prepare_exp07_graphs


def test_exp07_graph_shapes():
    """Verify that Exp07 graph builder produces exact (116, 117) node feature tensors."""
    n_samples = 4
    n_upper = N_ROIS * (N_ROIS - 1) // 2  # 6,670
    rng = np.random.default_rng(seed=42)
    test_X = rng.standard_normal((n_samples, n_upper), dtype=np.float32)
    test_y = np.array([0, 1, 0, 1])

    graphs = prepare_exp07_graphs(test_X, test_y, n_rois=N_ROIS, density=0.15)
    assert len(graphs) == n_samples

    for i, g in enumerate(graphs):
        assert g.x.shape == (N_ROIS, NODE_FEATURE_DIM), f"Graph {i} node feature shape mismatch: {g.x.shape}"
        assert g.edge_index.dim() == 2
        assert g.edge_index.shape[0] == 2
        assert g.y.item() == test_y[i]
        u, v = g.edge_index[0], g.edge_index[1]
        assert not torch.any(u == v), f"Self-loops found in graph {i}!"


def test_exp07_data_split_disjointness():
    """Verify that split_data maintains a 33-subject test set strictly disjoint from training when data is present."""
    data_file = data_root() / "exp07" / "X_combined_full.npy"
    if not data_file.exists():
        pytest.skip(f"Historical dataset file {data_file} not present; skipping data-dependent split test.")

    from exp07.utils.data_loader import load_data, split_data

    X, y, subjects = load_data()
    splits = split_data(X, y, subjects, n_test_clean=33)

    assert len(splits["X_test"]) == 33
    assert len(splits["y_test"]) == 33

    train_idx = set(splits["train_idx"])
    val_idx = set(splits["val_idx"])
    test_idx = set(splits["test_idx"])

    assert train_idx.isdisjoint(test_idx), "Train and test indices must be strictly disjoint!"
    assert val_idx.isdisjoint(test_idx), "Val and test indices must be strictly disjoint!"
    assert train_idx.isdisjoint(val_idx), "Train and val indices must be strictly disjoint!"
