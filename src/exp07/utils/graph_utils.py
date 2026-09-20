#!/usr/bin/env python
# ============================================================================
# GRAPH UTILITIES FOR EXPERIMENT 7 (Track B)
# Transforms functional connectivity matrices into PyTorch Geometric graph data
# Atlas: AAL-116 (116 nodes, 117 features: 116 FC correlations + 1 degree)
# ============================================================================

import numpy as np
import torch
from torch_geometric.data import Data

from exp07.utils.config import DENSITY, N_ROIS


def prepare_graphs(X_features, y_labels, n_rois=N_ROIS, density=DENSITY):
    """
    Convert functional connectivity matrices or upper-triangular vectors into PyG graphs.
    
    Args:
        X_features: Array of shape (N, n_rois * (n_rois - 1) // 2) or (N, n_rois, n_rois)
        y_labels: Array of shape (N,)
        n_rois: Number of regions (116 for AAL)
        density: Proportional threshold density (default: 0.20)
        
    Returns:
        List of torch_geometric.data.Data objects with:
            - x: Node feature matrix of shape (116, 117)
            - edge_index: Graph connectivity tensor of shape (2, E)
            - edge_attr: Edge correlation weights of shape (E,)
            - y: Subject diagnostic label tensor (scalar)
    """
    n_samples = len(X_features)
    triu_dim = n_rois * (n_rois - 1) // 2
    triu_idx = np.triu_indices(n_rois, k=1)
    graphs = []

    for i in range(n_samples):
        feat = X_features[i]
        if feat.ndim == 1 and feat.shape[0] == triu_dim:
            fc_matrix = np.zeros((n_rois, n_rois), dtype=np.float32)
            fc_matrix[triu_idx] = feat
            fc_matrix = fc_matrix + fc_matrix.T
        elif feat.ndim == 2 and feat.shape == (n_rois, n_rois):
            fc_matrix = np.array(feat, dtype=np.float32)
            np.fill_diagonal(fc_matrix, 0.0)
        else:
            raise ValueError(f"Unexpected feature shape {feat.shape} for n_rois={n_rois}")

        # Proportional thresholding based on absolute correlation magnitude
        abs_fc = np.abs(fc_matrix)
        upper_vals = abs_fc[triu_idx]
        if len(upper_vals) > 0 and not np.all(upper_vals == 0):
            threshold = np.percentile(upper_vals, 100 * (1 - density))
            edge_mask = abs_fc >= threshold
        else:
            edge_mask = np.zeros((n_rois, n_rois), dtype=bool)

        np.fill_diagonal(edge_mask, False)

        # Graph edges
        edge_coords = np.where(edge_mask)
        edge_index = torch.tensor(np.array(edge_coords), dtype=torch.long)
        edge_attr = torch.tensor(fc_matrix[edge_mask], dtype=torch.float32)

        # Node features: 116 FC correlation values + 1 normalized degree = 117
        degree = (np.sum(edge_mask, axis=1, keepdims=True) / max(n_rois - 1, 1)).astype(np.float32)
        node_feats = np.concatenate([fc_matrix, degree], axis=1)  # shape: (116, 117)
        node_feats_tensor = torch.tensor(node_feats, dtype=torch.float32)

        # Target label
        label_tensor = torch.tensor(int(y_labels[i]), dtype=torch.long)

        graphs.append(Data(
            x=node_feats_tensor,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=label_tensor
        ))

    return graphs
