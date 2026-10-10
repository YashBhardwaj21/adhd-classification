"""Classical Graph Convolutional Network (GCN) models and runners."""

try:
    from .gcn_model import ClassicalGCN
    from .gcn_training import run_experiment
except ImportError:
    ClassicalGCN = None
    run_experiment = None

__all__ = ["ClassicalGCN", "run_experiment"]

