#!/usr/bin/env python
# ============================================================================
# EXPERIMENT 7 CONFIGURATION
# Track B: Graph Convolutional Networks (Classical GCN vs Quantum QGCNN)
# Atlas: AAL-116 parcellation (116 regions of interest)
# ============================================================================

import os
import torch
from pathlib import Path

# Compute device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Directories
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
DATA_DIR = PROJECT_ROOT / "results" / "exp07"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Architecture & Parcellation parameters
N_ROIS = 116             # AAL-116 anatomical atlas regions
NODE_FEATURE_DIM = 117   # 116 connectivity weights + 1 normalized degree feature
N_QUBITS = 6             # 6-qubit quantum variational circuit
N_LAYERS = 2             # 2 strongly entangling variational layers
HIDDEN_DIM = 32          # Classical GCN hidden dimension
DENSITY = 0.20           # Proportional thresholding density

# Training hyperparameters
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 15
CHECKPOINT_INTERVAL = 10

def print_config():
    """Print configuration summary to stdout."""
    print("=" * 60)
    print("EXPERIMENT 7 CONFIGURATION (Track B)")
    print("=" * 60)
    print(f"Device:              {DEVICE}")
    print(f"Project Root:        {PROJECT_ROOT}")
    print(f"Results Directory:   {DATA_DIR}")
    print(f"Atlas / ROIs:        AAL-116 ({N_ROIS} nodes)")
    print(f"Node Feature Dim:    {NODE_FEATURE_DIM} (116 FC + 1 degree)")
    print(f"Quantum Qubits:      {N_QUBITS}")
    print(f"Quantum Layers:      {N_LAYERS}")
    print(f"Batch Size:          {BATCH_SIZE}")
    print(f"Learning Rate:       {LEARNING_RATE}")
    print(f"Weight Decay:        {WEIGHT_DECAY}")
    print(f"Max Epochs:          {EPOCHS}")
    print(f"Early Stop Patience: {PATIENCE}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
