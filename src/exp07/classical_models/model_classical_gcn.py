#!/usr/bin/env python
# ============================================================================
# CLASSICAL GCN MODEL - Baseline without quantum embedding
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

from utils.config import NODE_FEATURE_DIM

# Define hidden dimensions directly
HIDDEN_DIM = 32


class ClassicalGCN(nn.Module):
    """
    Classical Graph Convolutional Network baseline.
    No quantum components - runs entirely on GPU.
    """
    
    def __init__(self, input_dim=NODE_FEATURE_DIM, hidden_dim=32):
        super().__init__()
        
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.gcn3 = GCNConv(hidden_dim, 16)
        
        self.dropout = nn.Dropout(p=0.30)
        self.classifier = nn.Linear(16, 2)
    
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        
        # GCN Layer 1
        x = self.gcn1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # GCN Layer 2
        x = self.gcn2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # GCN Layer 3
        x = self.gcn3(x, edge_index)
        x = F.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        return self.classifier(x)


if __name__ == "__main__":
    print("✅ ClassicalGCN model defined")
    model = ClassicalGCN()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")