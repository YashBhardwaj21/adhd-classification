#!/usr/bin/env python
# ============================================================================
# GRAPH UTILITIES FOR EXPERIMENT 7 (Track B)
# Historical Protocol: AAL-116 (116 nodes, 117 features: 116 FC + 1 degree)
# Edge selection: Strongest 15% absolute correlation magnitude, signed weights
# No Minimum Spanning Tree (MST is specific to Exp 02 / Track A)
# ============================================================================

from typing import List, Optional, Union
import numpy as np
import torch
from torch_geometric.data import Data

from exp07.utils.config import DENSITY, N_ROIS


def prepare_graphs(
    X_features: Union[np.ndarray, List[np.ndarray]],
    y_labels: Union[np.ndarray, List[int]],
    n_rois: int = N_ROIS,
    density: float = DENSITY,
) -> List[Data]:
    """
    Convert functional connectivity matrices or upper-triangular vectors into PyG graphs.

    Historical protocol specifications:
      - 116 nodes (AAL-116 parcellation)
      - Proportional thresholding on absolute FC magnitude (density=0.15, top 15%)
      - Signed correlation values retained as edge_attr
      - 117 node features: 116 full correlation row values + 1 normalized degree
        (Note: the historical code used normalized thresholded edge count, whereas
        the paper describes weighted node strength; see docs/provenance.md)
      - No MST applied (MST + 20% is specific to Exp 02)

    Args:
        X_features: Array of shape (N, 6670) or (N, 116, 116)
        y_labels: Array of binary diagnostic labels of shape (N,)
        n_rois: Number of atlas ROIs (default: 116)
        density: Proportional threshold density (default: 0.15)

    Returns:
        List of torch_geometric.data.Data objects with:
          - x: (116, 117) node feature tensor
          - edge_index: (2, E) graph connectivity tensor
          - edge_attr: (E,) signed correlation weight tensor
          - y: scalar target label tensor
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

        # Historical node features: 116 FC row values + 1 normalized degree = 117
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
