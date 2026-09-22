#!/usr/bin/env python
# Graph utilities for Experiment 7 (Track B).
# Converts AAL-116 FC features to PyG Data objects with 117 node features
# (116 FC + 1 normalized degree) and 15% nominal density thresholding.

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
      - Proportional thresholding on absolute FC magnitude (15% nominal density parameter)
      - Percentile threshold on upper-triangle absolute FC values: threshold = np.percentile(upper_vals, 100 * (1 - density))
      - Edge selection: edge_mask = abs_fc >= threshold (note: >= comparison; ties may produce more than 15% edges)
      - Signed correlation values retained as edge_attr
      - Stored edge_attr is NOT passed as edge weights to the historical GCNConv message-passing layers
      - 117 node features: 116 signed FC values + 1 normalized degree (np.sum(abs_fc >= threshold, axis=1, keepdims=True) / n_rois)
        (Note: the historical code used normalized thresholded degree, whereas early paper drafts describe weighted node strength)
      - No MST applied (MST + 20% is specific to Exp 02)

    Args:
        X_features: Array of shape (N, 6670) or (N, 116, 116)
        y_labels: Array of binary diagnostic labels of shape (N,)
        n_rois: Number of atlas ROIs (default: 116)
        density: Nominal proportional threshold density (default: 0.15)

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

        # Nominal 15% threshold based on percentile of upper-triangle absolute FC
        abs_fc = np.abs(fc_matrix)
        upper_vals = abs_fc[triu_idx]
        if len(upper_vals) > 0 and not np.all(upper_vals == 0):
            threshold = np.percentile(upper_vals, 100 * (1 - density))
            edge_mask = abs_fc >= threshold
        else:
            edge_mask = np.zeros((n_rois, n_rois), dtype=bool)

        np.fill_diagonal(edge_mask, False)

        # Graph edges (edge_attr stores original signed FC values)
        edge_coords = np.where(edge_mask)
        edge_index = torch.tensor(np.array(edge_coords), dtype=torch.long)
        edge_attr = torch.tensor(fc_matrix[edge_mask], dtype=torch.float32)

        # Historical node features: 116 signed FC row values + 1 normalized degree = 117
        degree = (np.sum(abs_fc >= threshold, axis=1, keepdims=True) / n_rois).astype(np.float32)
        node_feats = np.hstack([fc_matrix, degree])  # shape: (116, 117)
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
