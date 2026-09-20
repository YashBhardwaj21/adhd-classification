"""Classical Graph Convolutional Network (GCN) models and runners."""

try:
    from .model_classical_gcn import ClassicalGCN
    from .train_classical_gcn import run_experiment
except ImportError:
    ClassicalGCN = None
    run_experiment = None

__all__ = ["ClassicalGCN", "run_experiment"]
