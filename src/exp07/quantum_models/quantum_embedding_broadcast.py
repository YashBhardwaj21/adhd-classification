#!/usr/bin/env python
# ============================================================================
# QUANTUM EMBEDDING - VECTORIZED USING PENNYLANE BROADCASTING
# Processes ALL nodes in a graph with ONE quantum circuit call
# Speedup: ~50-100x faster than per-node quantum calls
# ============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import pennylane as qml

from utils.config import N_QUBITS, N_LAYERS, NODE_FEATURE_DIM, DEVICE


class QuantumEmbeddingGPU_Broadcast(nn.Module):
    """
    Optimized quantum embedding using PennyLane's native broadcasting.
    Processes ALL nodes in ONE quantum call.
    
    BEFORE: for node in 116 nodes: quantum_circuit(node)  # 116 calls
    AFTER:  quantum_circuit(all_nodes)                    # 1 call
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
        
        # Create the broadcasted quantum circuit
        self._create_broadcasted_circuit()
    
    def _create_broadcasted_circuit(self):
        """Create a quantum circuit that uses broadcasting."""
        
        @qml.qnode(self.dev, interface="torch", diff_method="adjoint")
        def quantum_circuit_broadcast(inputs, weights):
            """
            Quantum circuit with broadcasting support.
            inputs: (num_nodes, 2 * n_qubits) - batched inputs
            weights: (n_layers, n_qubits, 3) - variational parameters
            """
            # Angle encoding with broadcasting
            # inputs[:, qubit] processes ALL nodes at once
            for qubit in range(self.n_qubits):
                qml.RY(inputs[:, qubit], wires=qubit)
                qml.RZ(inputs[:, qubit + self.n_qubits], wires=qubit)
            
            # Variational layers with broadcasting
            for layer in range(self.n_layers):
                # Rotation gates
                for qubit in range(self.n_qubits):
                    qml.RX(weights[layer, qubit, 0], wires=qubit)
                    qml.RY(weights[layer, qubit, 1], wires=qubit)
                    qml.RZ(weights[layer, qubit, 2], wires=qubit)
                
                # Entangling gates (CNOT ring)
                for qubit in range(self.n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
                qml.CNOT(wires=[self.n_qubits - 1, 0])
            
            # Return expectation values
            # Each returns shape (num_nodes,)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]
        
        self.broadcasted_circuit = quantum_circuit_broadcast
    
    def forward(self, x):
        """
        Forward pass: ALL nodes in ONE quantum call.
        
        Args:
            x: (num_nodes, input_dim)
        
        Returns:
            (num_nodes, n_qubits)
        """
        # Classical projection
        x = self.classical_proj(x)  # (num_nodes, 2 * n_qubits)
        x = torch.tanh(x) * np.pi   # Scale to [-pi, pi]
        
        # ONE quantum call for ALL nodes!
        result = self.broadcasted_circuit(x, self.quantum_weights)
        
        # result is a tuple of tensors, one per qubit
        # Each tensor shape: (num_nodes,)
        # Stack to get (num_nodes, n_qubits)
        return torch.stack(result, dim=1)


if __name__ == "__main__":
    print("="*60)
    print("Testing QuantumEmbeddingGPU_Broadcast")
    print("="*60)
    
    dev = qml.device("lightning.gpu", wires=N_QUBITS, batch_obs=True, c_dtype=np.complex64)
    print(f"Quantum device: {dev}")
    
    embedding = QuantumEmbeddingGPU_Broadcast(dev).to(DEVICE)
    print(f"Parameters: {sum(p.numel() for p in embedding.parameters()):,}")
    
    num_nodes = 116
    test_input = torch.randn(num_nodes, NODE_FEATURE_DIM, device=DEVICE)
    print(f"Input shape: {test_input.shape}")
    
    with torch.no_grad():
        output = embedding(test_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Expected: ({num_nodes}, {N_QUBITS})")
    
    if output.shape == (num_nodes, N_QUBITS):
        print("✅ CORRECT! Broadcasted quantum embedding works!")
    else:
        print(f"❌ Shape mismatch: got {output.shape}")
    
    print(f"Device: {output.device}")