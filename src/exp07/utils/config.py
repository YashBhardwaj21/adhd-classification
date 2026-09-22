#!/usr/bin/env python
# Experiment 7 configuration (Track B: Classical GCN vs Quantum QGCNN).
# Atlas: AAL-116 parcellation (116 regions of interest).
# Loads verified historical parameters from configs/exp07/reported_run.json.

import json
from pathlib import Path
import torch

from common.paths import checkpoints_root, data_root, project_root, results_root

# Compute device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Directories
DATA_DIR = data_root() / "exp07"
RESULTS_DIR = results_root() / "exp07"
CHECKPOINTS_DIR = checkpoints_root() / "exp07"

# Load reported run configuration if available
CONFIG_FILE = project_root() / "configs" / "exp07" / "reported_run.json"
_reported_config = {}
if CONFIG_FILE.exists():
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            _reported_config = json.load(f)
    except Exception:
        _reported_config = {}

_graph_cfg = _reported_config.get("graph", {})
_training_cfg = _reported_config.get("training", {})
_quantum_cfg = _reported_config.get("quantum", {})

# Verified Architecture & Parcellation parameters
N_ROIS = int(_graph_cfg.get("n_rois", 116))
NODE_FEATURE_DIM = N_ROIS + 1  # 116 FC correlations + 1 normalized degree = 117
N_QUBITS = int(_quantum_cfg.get("n_qubits", 6))
N_LAYERS = int(_quantum_cfg.get("n_layers", 1))
HIDDEN_DIM = 32
DENSITY = float(_graph_cfg.get("density", 0.15))

# Verified Training hyperparameters
BATCH_SIZE = int(_training_cfg.get("batch_size", 8))
EPOCHS = int(_training_cfg.get("epochs", 20))
LEARNING_RATE = float(_training_cfg.get("learning_rate", 1e-3))
WEIGHT_DECAY = float(_training_cfg.get("weight_decay", 1e-5))
PATIENCE = int(_training_cfg.get("patience", 10))
CHECKPOINT_INTERVAL = int(_training_cfg.get("checkpoint_interval", 1))

# Deterministic random seed
SEED = int(_training_cfg.get("seed", 42))
RANDOM_SEED = SEED


def print_config():
    """Print configuration summary to stdout."""
    print("Experiment 7 Configuration (Track B - Verified Historical)")
    print(f"Device:              {DEVICE}")
    print(f"Data Directory:      {DATA_DIR}")
    print(f"Results Directory:   {RESULTS_DIR}")
    print(f"Checkpoints Dir:     {CHECKPOINTS_DIR}")
    print(f"Atlas / ROIs:        AAL-116 ({N_ROIS} nodes)")
    print(f"Node Feature Dim:    {NODE_FEATURE_DIM} (116 FC + 1 degree)")
    print(f"Density:             {DENSITY}")
    print(f"Quantum Qubits:      {N_QUBITS}")
    print(f"Quantum Layers:      {N_LAYERS}")
    print(f"Batch Size:          {BATCH_SIZE}")
    print(f"Learning Rate:       {LEARNING_RATE}")
    print(f"Weight Decay:        {WEIGHT_DECAY}")
    print(f"Max Epochs:          {EPOCHS}")
    print(f"Early Stop Patience: {PATIENCE}")


if __name__ == "__main__":
    print_config()
