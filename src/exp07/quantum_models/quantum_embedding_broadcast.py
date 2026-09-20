#!/usr/bin/env python
# ============================================================================
# QUANTUM EMBEDDING - VECTORIZED USING PENNYLANE BROADCASTING
# Canonical implementation variant for Experiment 7
# Processes all 116 nodes in a graph with a single broadcasted quantum circuit call
# ============================================================================

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from exp07.utils.config import N_LAYERS, N_QUBITS, NODE_FEATURE_DIM


class QuantumEmbeddingGPU_Broadcast(nn.Module):
    """
    Optimized quantum embedding using PennyLane's native broadcasting.
    Evaluates all 116 node feature vectors in a single batched quantum circuit call.
    """

    def __init__(
        self,
        dev,
        input_dim: int = NODE_FEATURE_DIM,
        n_qubits: int = N_QUBITS,
        n_layers: int = N_LAYERS,
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

        # Create the broadcasted quantum circuit
        self._create_broadcasted_circuit()

    def _create_broadcasted_circuit(self):
        """Create a quantum circuit that uses broadcasting."""

        @qml.qnode(self.dev, interface="torch", diff_method="adjoint")
        def quantum_circuit_broadcast(inputs, weights):
            """
            Quantum circuit with broadcasting support.
            inputs: (num_nodes, 2 * n_qubits) - batched inputs
            weights: (n_layers, n_qubits, 3) - quantum weights
            """
            # Angle encoding for all nodes simultaneously
            for q in range(self.n_qubits):
                qml.RY(inputs[..., 2 * q], wires=q)
                qml.RZ(inputs[..., 2 * q + 1], wires=q)

            # Variational layers
            for l in range(self.n_layers):
                # Entanglement ring
                for q in range(self.n_qubits):
                    qml.CNOT(wires=[q, (q + 1) % self.n_qubits])

                # Parameterized single-qubit rotations
                for q in range(self.n_qubits):
                    qml.Rot(
                        weights[l, q, 0],
                        weights[l, q, 1],
                        weights[l, q, 2],
                        wires=q,
                    )

            # Measure expectation values for all qubits
            return [qml.expval(qml.PauliZ(q)) for q in range(self.n_qubits)]

        self.broadcasted_circuit = quantum_circuit_broadcast

    def forward(self, x):
        """
        Forward pass for a batch of nodes.
        
        Args:
            x: Node features tensor of shape (num_nodes, input_dim)
            
        Returns:
            Quantum embeddings tensor of shape (num_nodes, n_qubits)
        """
        # Classical projection: (num_nodes, 117) -> (num_nodes, 2 * n_qubits)
        x = self.classical_proj(x)
        x = torch.tanh(x) * np.pi   # Scale to [-pi, pi]

        # Batched quantum call across all nodes
        result = self.broadcasted_circuit(x, self.quantum_weights)

        # Stack expectations to return shape (num_nodes, n_qubits)
        return torch.stack(result, dim=1)
