"""Unit tests for Experiment 7 model architectures, tensor shapes, and parameter counts."""

import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("torch_geometric", reason="PyTorch Geometric required for Exp07 model tests")
qml = pytest.importorskip("pennylane", reason="PennyLane required for Exp07 quantum model shape tests")

from torch_geometric.data import Batch, Data

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from exp07.classical_models.model_classical_gcn import ClassicalGCN
from exp07.quantum_models.quantum_embedding_broadcast import QuantumEmbeddingGPU_Broadcast
from exp07.quantum_models.train_qgcnn_vectorized import HybridQGCNN_Vectorized
from exp07.utils.config import (
    BATCH_SIZE,
    DENSITY,
    LEARNING_RATE,
    N_LAYERS,
    N_QUBITS,
    NODE_FEATURE_DIM,
    WEIGHT_DECAY,
)


def count_parameters(model: torch.nn.Module) -> int:
    """Return total number of trainable and non-trainable parameters."""
    return sum(p.numel() for p in model.parameters())


def create_dummy_batch(batch_size: int = 2, n_nodes: int = 116, n_features: int = 117):
    """Helper creating a synthetic PyG Batch object for testing."""
    data_list = []
    for i in range(batch_size):
        x = torch.randn(n_nodes, n_features, dtype=torch.float32)
        # Create a simple cycle edge index
        src = torch.arange(n_nodes)
        dst = (src + 1) % n_nodes
        edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
        edge_attr = torch.randn(edge_index.shape[1], dtype=torch.float32)
        y = torch.tensor(i % 2, dtype=torch.long)
        data_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))
    return Batch.from_data_list(data_list)


def test_exp07_config_constants():
    """Verify verified historical configuration constants for Experiment 7."""
    assert N_QUBITS == 6
    assert N_LAYERS == 1
    assert BATCH_SIZE == 8
    assert LEARNING_RATE == 1e-3
    assert WEIGHT_DECAY == 1e-5
    assert DENSITY == 0.15


def test_classical_gcn_architecture_and_parameters():
    """
    Assert Classical GCN architecture:
      117 -> 32 -> 32 -> 16 -> 2
      Exact parameter count must equal 5,522.
    """
    model = ClassicalGCN(input_dim=NODE_FEATURE_DIM, hidden_dim=32)
    n_params = count_parameters(model)

    assert n_params == 5522, f"Classical GCN parameter count must be 5522, got {n_params}"

    batch = create_dummy_batch(batch_size=2)
    out = model(batch)
    assert out.shape == (2, 2), f"Expected logits shape (2, 2), got {out.shape}"


def test_quantum_qgcnn_architecture_and_parameters():
    """
    Assert Hybrid QGCNN architecture:
      117 -> 12 (classical projection)
      6-qubit quantum embedding (1 layer, 18 weights)
      GCN: 6 -> 16 -> 16 -> 16 -> 2
      Exact parameter count must equal 2,188.
    """
    # Use CPU default.qubit for deterministic parameter verification
    dev = qml.device("default.qubit", wires=N_QUBITS)

    model = HybridQGCNN_Vectorized(
        dev=dev,
        input_dim=NODE_FEATURE_DIM,
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
        hidden_dim=16,
    )
    n_params = count_parameters(model)

    assert n_params == 2188, f"QGCNN parameter count must be 2188, got {n_params}"


def test_quantum_embedding_module_parameters():
    """Verify standalone quantum embedding module projection and weights."""
    dev = qml.device("default.qubit", wires=6)
    embedding = QuantumEmbeddingGPU_Broadcast(dev=dev, input_dim=117, n_qubits=6, n_layers=1)

    # Classical proj: 117 * 12 + 12 = 1416
    # Quantum weights: 1 * 6 * 3 = 18
    # Total = 1434
    total_embedding_params = count_parameters(embedding)
    assert total_embedding_params == 1434, f"Embedding params expected 1434, got {total_embedding_params}"
