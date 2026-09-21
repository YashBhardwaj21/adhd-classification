"""Quantum Graph Convolutional Neural Network (QGCNN) models and runners."""

try:
    from .quantum_embedding_broadcast import QuantumEmbeddingGPU_Broadcast
    from .train_qgcnn_vectorized import HybridQGCNN_Vectorized, run_experiment
except ImportError:
    QuantumEmbeddingGPU_Broadcast = None
    HybridQGCNN_Vectorized = None
    run_experiment = None

__all__ = ["QuantumEmbeddingGPU_Broadcast", "HybridQGCNN_Vectorized", "run_experiment"]
