"""Utility functions and configuration for Experiment 7 (Classical vs Quantum GCNN)."""

from exp07.utils.config import (
    BATCH_SIZE,
    CHECKPOINT_INTERVAL,
    CHECKPOINTS_DIR,
    DATA_DIR,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    N_LAYERS,
    N_QUBITS,
    N_ROIS,
    NODE_FEATURE_DIM,
    PATIENCE,
    RANDOM_SEED,
    RESULTS_DIR,
    WEIGHT_DECAY,
    print_config,
)
from exp07.utils.data_loader import load_data, split_data

try:
    from exp07.utils.graph_utils import prepare_graphs
except ImportError:
    prepare_graphs = None

try:
    from exp07.utils.training_utils import (
        save_checkpoint,
        save_timed_checkpoint,
        train_one_epoch,
        validate,
    )
except ImportError:
    train_one_epoch = validate = save_checkpoint = save_timed_checkpoint = None

__all__ = [
    "DEVICE",
    "DATA_DIR",
    "RESULTS_DIR",
    "CHECKPOINTS_DIR",
    "N_QUBITS",
    "N_LAYERS",
    "NODE_FEATURE_DIM",
    "N_ROIS",
    "BATCH_SIZE",
    "EPOCHS",
    "LEARNING_RATE",
    "WEIGHT_DECAY",
    "PATIENCE",
    "CHECKPOINT_INTERVAL",
    "RANDOM_SEED",
    "print_config",
    "load_data",
    "split_data",
    "prepare_graphs",
    "train_one_epoch",
    "validate",
    "save_checkpoint",
    "save_timed_checkpoint",
]
