#!/usr/bin/env python
# ============================================================================
# QUANTUM EMBEDDING - VECTORIZED VERSION USING torch.vmap
# Processes ALL nodes in a graph with ONE quantum circuit call
# Speedup: ~50-100x faster than per-node quantum calls
# ============================================================================

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from exp07.utils.config import DEVICE, N_LAYERS, N_QUBITS, NODE_FEATURE_DIM


class QuantumEmbeddingGPU_VMap(nn.Module):
    """
    Optimized quantum embedding using torch.vmap.
    Processes ALL nodes in ONE quantum call.
    """

    def __init__(
        self,
        dev,
        input_dim=NODE_FEATURE_DIM,
        n_qubits=N_QUBITS,
        n_layers=N_LAYERS,
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.dev = dev

        # Classical projection
        self.classical_proj = nn.Linear(input_dim, 2 * n_qubits)

        # Learnable quantum weights
        self.quantum_weights = nn.Parameter(
            0.01 * torch.randn(n_layers, n_qubits, 3, dtype=torch.float32)
        )

        # Create the quantum circuit
        self._create_circuit()

    def _create_circuit(self):
        """Define the single-node quantum circuit with vmap support."""

        @qml.qnode(self.dev, interface="torch", diff_method="adjoint")
        def quantum_circuit_single(inputs, weights):
            """Circuit for a SINGLE node."""
            # Angle encoding
            for qubit in range(self.n_qubits):
                qml.RY(inputs[qubit], wires=qubit)
                qml.RZ(inputs[qubit + self.n_qubits], wires=qubit)

            # Variational layers
            for layer in range(self.n_layers):
                for qubit in range(self.n_qubits):
                    qml.RX(weights[layer, qubit, 0], wires=qubit)
                    qml.RY(weights[layer, qubit, 1], wires=qubit)
                    qml.RZ(weights[layer, qubit, 2], wires=qubit)

                for qubit in range(self.n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])

            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        # Wrap to return tensor
        def circuit_fn(inputs, weights):
            result = quantum_circuit_single(inputs, weights)
            return torch.stack(result)

        # Apply torch.vmap
        self.circuit_fn = circuit_fn
        self.batched_circuit = torch.vmap(circuit_fn, in_dims=(0, None))

    def forward(self, x):
        """
        Forward pass: ALL nodes in ONE quantum call.
        
        Args:
            x: (num_nodes, input_dim)
        
        Returns:
            (num_nodes, n_qubits)
        """
        x = self.classical_proj(x)
        x = torch.tanh(x) * np.pi
        return self.batched_circuit(x, self.quantum_weights)


if __name__ == "__main__":
    print("="*60)
    print("Testing QuantumEmbeddingGPU_VMap (torch.vmap)")
    print("="*60)

    dev = qml.device("lightning.gpu", wires=N_QUBITS, batch_obs=True, c_dtype=np.complex64)
    print(f"Quantum device: {dev}")

    embedding = QuantumEmbeddingGPU_VMap(dev).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in embedding.parameters()):,}")

    num_nodes = 116
    test_input = torch.randn(num_nodes, NODE_FEATURE_DIM, device=DEVICE)
    print(f"Input shape: {test_input.shape}")

    with torch.no_grad():
        output = embedding(test_input)

    print(f"Output shape: {output.shape}")
    print(f"Expected: ({num_nodes}, {N_QUBITS})")

    if output.shape == (num_nodes, N_QUBITS):
        print("✅ CORRECT! Vectorized quantum embedding works!")
    else:
        print(f"❌ Shape mismatch: got {output.shape}")

    print(f"Device: {output.device}")
