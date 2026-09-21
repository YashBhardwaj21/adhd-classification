"""Unit tests for Experiment 7 (Track B) graph construction protocol."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from exp07.utils.graph_utils import prepare_graphs


def test_exp07_graph_construction_dimensions():
    """Verify that AAL-116 FC upper-triangle features produce graphs with 116 nodes and 117 features."""
    n_rois = 116
    triu_dim = n_rois * (n_rois - 1) // 2  # 6670
    n_samples = 3

    # Generate synthetic upper-triangle correlation vectors with positive and negative weights
    rng = np.random.RandomState(42)
    X_synthetic = rng.uniform(-0.8, 0.8, size=(n_samples, triu_dim)).astype(np.float32)
    y_synthetic = np.array([0, 1, 0])

    graphs = prepare_graphs(X_synthetic, y_synthetic, n_rois=n_rois, density=0.15)

    assert len(graphs) == n_samples

    for idx, g in enumerate(graphs):
        # Node features: (116 nodes, 117 features)
        assert g.x.shape == (116, 117), f"Sample {idx} x shape mismatch: {g.x.shape}"

        # Target label
        assert g.y.item() == y_synthetic[idx]

        # Edges should be non-empty
        num_edges = g.edge_index.shape[1]
        assert num_edges > 0, "Graph must have non-zero edge count"

        # Edge index shape (2, E)
        assert g.edge_index.shape[0] == 2

        # Edge attributes must match edge count and contain signed values
        assert g.edge_attr.shape[0] == num_edges
        assert torch.any(g.edge_attr < 0), "Signed edge attributes must include negative weights"
        assert torch.any(g.edge_attr > 0), "Signed edge attributes must include positive weights"

        # No self-loops
        assert not torch.any(g.edge_index[0] == g.edge_index[1]), "Graph must not contain self-loops"


def test_exp07_density_proportional_thresholding():
    """Assert that edge count reflects the verified 15% density threshold."""
    n_rois = 116
    triu_dim = n_rois * (n_rois - 1) // 2  # 6670
    density = 0.15

    rng = np.random.RandomState(42)
    X = rng.randn(1, triu_dim).astype(np.float32)
    y = np.array([1])

    graphs = prepare_graphs(X, y, n_rois=n_rois, density=density)
    g = graphs[0]

    # In an undirected graph with symmetric edges, undirected count is num_edges / 2
    undirected_edges = g.edge_index.shape[1] // 2
    expected_undirected = int(np.round(triu_dim * density))

    # Allow minor boundary variation due to percentile thresholding
    assert abs(undirected_edges - expected_undirected) <= 5, (
        f"Undirected edge count {undirected_edges} deviates from expected ~{expected_undirected} for density {density}"
    )
