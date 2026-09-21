"""Unit tests for Experiment 2 (Track A) CC200 graph construction protocol."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from exp02.config import DENSITY as EXP02_DENSITY, N_ROIS as EXP02_N_ROIS
from exp02.graph_utils import prepare_graphs as prepare_exp02_graphs


def test_exp02_config_constants():
    """Verify Experiment 02 configuration constants (CC200 atlas, 190 ROIs, density 0.20)."""
    assert EXP02_N_ROIS == 190, "Exp 02 CC200 atlas uses 190 active ROIs"
    assert EXP02_DENSITY == 0.20, "Exp 02 uses 0.20 density"


def test_exp02_graph_construction_shapes():
    """Verify that CC200 upper-triangular FC features produce graphs with 190 nodes and 191 features."""
    n_rois = EXP02_N_ROIS
    triu_dim = n_rois * (n_rois - 1) // 2  # 17,955 features
    n_samples = 2

    rng = np.random.RandomState(42)
    X_synthetic = rng.uniform(-0.9, 0.9, size=(n_samples, triu_dim)).astype(np.float32)
    y_synthetic = np.array([0, 1])

    graphs = prepare_exp02_graphs(X_synthetic, y_synthetic, n_rois=n_rois, density=EXP02_DENSITY)

    assert len(graphs) == n_samples

    for idx, g in enumerate(graphs):
        # Node features: 190 nodes, 191 features (190 FC + 1 normalized degree)
        assert g.x.shape == (190, 191), f"Sample {idx} node feature shape mismatch: {g.x.shape}"
        assert g.y.item() == y_synthetic[idx]

        # Edges
        num_edges = g.edge_index.shape[1]
        assert num_edges > 0, "Graph must have edges"
        assert g.edge_index.shape[0] == 2
        assert g.edge_attr.shape[0] == num_edges

        # No self-loops
        assert not torch.any(g.edge_index[0] == g.edge_index[1]), "Graph must not contain self-loops"


def test_exp02_proportional_threshold_edge_count():
    """Verify that edge density matches ~0.20 of total possible connections."""
    n_rois = 190
    triu_dim = n_rois * (n_rois - 1) // 2  # 17,955
    density = 0.20

    rng = np.random.RandomState(42)
    X = rng.randn(1, triu_dim).astype(np.float32)
    y = np.array([0])

    graphs = prepare_exp02_graphs(X, y, n_rois=n_rois, density=density)
    g = graphs[0]

    undirected_edges = g.edge_index.shape[1] // 2
    expected_undirected = int(np.round(triu_dim * density))  # ~3,591

    assert abs(undirected_edges - expected_undirected) <= 10, (
        f"Undirected edge count {undirected_edges} deviates from expected ~{expected_undirected}"
    )
