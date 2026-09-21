#!/usr/bin/env python
# ============================================================================
# GRAPH UTILITIES - Convert CC200 FC features to PyTorch Geometric graphs
# Experiment 02: CC200 Atlas (190 ROIs, density 0.20)
# Note: For topological analysis, notebooks/exp02/02_graph_construction_and_validation.ipynb
# implements mst_graph (MST + Proportional Thresholding to 3,591 edges).
# ============================================================================

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from .config import DENSITY, N_ROIS


def prepare_graphs(X_features, y_labels, n_rois=N_ROIS, density=DENSITY):
    """
    Convert upper-triangular FC features into PyTorch Geometric graphs.
    
    Args:
        X_features: (n_samples, n_rois * (n_rois - 1) / 2) features
        y_labels: (n_samples,) labels
        n_rois: Number of ROIs (nodes)
        density: Edge density threshold
    
    Returns:
        List of torch_geometric.data.Data objects
    """
    triu_idx = np.triu_indices(n_rois, k=1)
    graphs = []

    for i in tqdm(range(len(X_features)), desc="Building graphs"):
        # Reconstruct full correlation matrix
        fc_matrix = np.zeros((n_rois, n_rois), dtype=np.float32)
        fc_matrix[triu_idx] = X_features[i]
        fc_matrix = fc_matrix + fc_matrix.T

        # Edge selection based on density threshold
        abs_fc = np.abs(fc_matrix)
        threshold = np.percentile(abs_fc[triu_idx], 100 * (1 - density))
        edge_mask = abs_fc >= threshold

        # Edge index (2 x num_edges)
        edge_index = torch.tensor(
            np.array(np.where(edge_mask)),
            dtype=torch.long
        )

        # Edge attributes (correlation values)
        edge_attr = torch.tensor(
            fc_matrix[edge_mask],
            dtype=torch.float32
        )

        # Node features: FC matrix + degree info
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
    print("✅ graph_utils.py loaded successfully")
