#!/usr/bin/env python
# Graph utilities to convert CC200 FC features to PyTorch Geometric graphs.
# Experiment 02: CC200 Atlas (190 ROIs, density 0.20).
# For topological analysis, see notebooks/exp02/02_graph_construction_and_validation.ipynb.

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from .config import DENSITY, N_ROIS


def _build_mst_pt_edge_mask(fc_matrix, n_rois, density):
    """
    Construct MST+PT edge mask.
    1. Distance metric d_ij = 1 - |r_ij| (Kruskal MST maximizing absolute correlation).
    2. Add strongest remaining absolute correlation edges until density target is reached.
    """
    total_possible = n_rois * (n_rois - 1) // 2
    target_edges = int(round(total_possible * density))
    triu_idx = np.triu_indices(n_rois, k=1)
    rows, cols = triu_idx
    abs_fc = np.abs(fc_matrix)
    abs_weights = abs_fc[rows, cols]

    # Deterministic descending sort by |r|: primary key -abs_weights, secondary keys rows, cols
    order = np.lexsort((cols, rows, -abs_weights))

    # Union-Find for Minimum Spanning Tree
    parent = list(range(n_rois))

    def find(i):
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
        return i

    def union(i, j):
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_i] = root_j
            return True
        return False

    edge_mask = np.zeros((n_rois, n_rois), dtype=bool)
    mst_edges_count = 0
    non_mst_indices = []

    for idx in order:
        r, c = rows[idx], cols[idx]
        if union(r, c):
            edge_mask[r, c] = True
            edge_mask[c, r] = True
            mst_edges_count += 1
        else:
            non_mst_indices.append(idx)

    # Add strongest remaining edges until target undirected edge count is reached
    remaining_needed = max(0, target_edges - mst_edges_count)
    for idx in non_mst_indices[:remaining_needed]:
        r, c = rows[idx], cols[idx]
        edge_mask[r, c] = True
        edge_mask[c, r] = True

    return edge_mask


def prepare_graphs(X_features, y_labels, n_rois=N_ROIS, density=DENSITY):
    """
    Convert upper-triangular FC features into PyTorch Geometric graphs
    using canonical MST + Proportional Thresholding (MST+PT).

    Args:
        X_features: (n_samples, n_rois * (n_rois - 1) / 2) features
        y_labels: (n_samples,) labels
        n_rois: Number of ROIs (nodes), default 190
        density: Target graph edge density threshold, default 0.20 (3,591 undirected edges)

    Returns:
        List of torch_geometric.data.Data objects with signed FC edge attributes
    """
    triu_idx = np.triu_indices(n_rois, k=1)
    graphs = []

    for i in tqdm(range(len(X_features)), desc="Building graphs"):
        # Reconstruct full correlation matrix with zero diagonal
        fc_matrix = np.zeros((n_rois, n_rois), dtype=np.float32)
        fc_matrix[triu_idx] = X_features[i]
        fc_matrix = fc_matrix + fc_matrix.T

        # Canonical MST+PT edge selection (distance d = 1 - |r|, target 20% density)
        edge_mask = _build_mst_pt_edge_mask(fc_matrix, n_rois, density)

        # Edge index (2 x num_edges, bidirectional / directed)
        edge_index = torch.tensor(
            np.array(np.where(edge_mask)),
            dtype=torch.long
        )

        # Retain signed FC values as edge attributes
        edge_attr = torch.tensor(
            fc_matrix[edge_mask],
            dtype=torch.float32
        )

        # Node features: FC matrix + normalized degree
        degree = (np.sum(edge_mask, axis=1, keepdims=True) / n_rois).astype(np.float32)
        node_features = np.concatenate([fc_matrix, degree], axis=1)
        node_features = torch.tensor(node_features, dtype=torch.float32)

        # Create PyG Data object
        graphs.append(Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor(y_labels[i], dtype=torch.long)
        ))

    return graphs


if __name__ == "__main__":
    # Test the function
    print("Testing graph_utils...")
    print("graph_utils.py loaded successfully")
