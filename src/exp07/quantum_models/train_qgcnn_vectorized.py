#!/usr/bin/env python
"""
Hybrid Quantum-Classical Graph Convolutional Neural Network (QGCNN).
Uses PennyLane broadcasting with the lightning.gpu simulator device.

Audited Architecture (Exp 07 Track B):
- Quantum embedding: 6 qubits, 1 variational layer, angle encoding (RY, RZ), CNOT entanglement ring.
- GCN backbone: 3 GCNConv layers (6 -> 16 -> 16 -> 16) with BatchNorm1d and Dropout(0.30).
- Global pooling: global_mean_pool -> Linear(16, 2).
"""

import gc
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from common.paths import checkpoints_root, data_root, results_root
from exp07.quantum_models.quantum_embedding_broadcast import QuantumEmbeddingGPU_Broadcast
from exp07.utils.config import (
    BATCH_SIZE,
    CHECKPOINT_INTERVAL,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    N_LAYERS,
    N_QUBITS,
    NODE_FEATURE_DIM,
    PATIENCE,
    RANDOM_SEED,
    WEIGHT_DECAY,
    print_config,
)
from exp07.utils.data_loader import load_data, split_data
from exp07.utils.graph_utils import prepare_graphs
from exp07.utils.training_utils import save_checkpoint, save_timed_checkpoint, train_one_epoch, validate

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


class HybridQGCNN_Vectorized(nn.Module):
    """
    Hybrid Quantum-Classical GCNN with broadcast quantum embedding.
    """

    def __init__(
        self,
        dev,
        input_dim: int = NODE_FEATURE_DIM,
        n_qubits: int = N_QUBITS,
        n_layers: int = N_LAYERS,
        hidden_dim: int = 16,
    ):
        super().__init__()
        self.quantum_embedding = QuantumEmbeddingGPU_Broadcast(dev, input_dim, n_qubits, n_layers)
        self.gcn1 = GCNConv(n_qubits, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.gcn3 = GCNConv(hidden_dim, 16)
        self.dropout = nn.Dropout(p=0.30)
        self.classifier = nn.Linear(16, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.quantum_embedding(x)
        x = self.gcn1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.gcn2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.gcn3(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        return self.classifier(x)


def create_quantum_device(device_name: str = "lightning.gpu", n_qubits: int = N_QUBITS):
    """
    Initialize the PennyLane quantum device with clear diagnostics on failure.
    """
    try:
        if device_name == "lightning.gpu":
            return qml.device("lightning.gpu", wires=n_qubits, batch_obs=True, c_dtype=np.complex64)
        else:
            return qml.device(device_name, wires=n_qubits)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize quantum device '{device_name}': {exc}\n"
            "Track B requires PennyLane Lightning GPU (pennylane-lightning-gpu==0.44.0) with CUDA.\n"
            "If running on a CPU-only environment or running smoke checks, specify device_name='default.qubit'."
        ) from exc


def run_experiment(
    data_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    checkpoints_dir: Optional[Path] = None,
    seed: int = RANDOM_SEED,
    quantum_device: str = "lightning.gpu",
) -> Tuple[nn.Module, Dict]:
    """
    Execute the hybrid quantum GCNN training protocol.
    """
    data_dir = Path(data_dir) if data_dir is not None else data_root()
    results_dir = Path(results_dir) if results_dir is not None else results_root() / "exp07"
    checkpoints_dir = Path(checkpoints_dir) if checkpoints_dir is not None else checkpoints_root() / "exp07"

    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print_config()
    print("\n" + "=" * 60)
    print("VECTORIZED QUANTUM GCNN (BROADCASTING)")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Data directory: {data_dir}")
    print(f"Results directory: {results_dir}")
    print(f"Checkpoints directory: {checkpoints_dir}")
    print("=" * 60)

    dev = create_quantum_device(quantum_device, N_QUBITS)
    print(f"Quantum device: {dev}")

    print("\nLoading data...")
    X, y, subjects = load_data(data_dir=data_dir)
    data = split_data(X, y, subjects, random_state=seed)

    print("\nBuilding graphs...")
    train_graphs = prepare_graphs(data["X_train"], data["y_train"])
    val_graphs = prepare_graphs(data["X_val"], data["y_val"])
    test_graphs = prepare_graphs(data["X_test"], data["y_test"])

    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    model = HybridQGCNN_Vectorized(dev).to(DEVICE)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    history = {"train_loss": [], "train_acc": [], "val_acc": [], "val_f1": [], "val_auc": []}
    best_auc = -1.0
    patience_counter = 0

    print("\n" + "=" * 60)
    print("STARTING TRAINING (BROADCASTING)")
    print("=" * 60)

    total_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
        val_acc, val_f1, val_auc = validate(model, val_loader)

        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_auc"].append(val_auc)

        epoch_time = time.time() - epoch_start
        mem_alloc = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0

        print(f"\nEpoch {epoch:03d}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"  Val Acc: {val_acc:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}")
        print(f"  GPU: {mem_alloc:.2f}GB | LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")

        if epoch % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = checkpoints_dir / f"quantum_checkpoint_epoch_{epoch}.pth"
            save_checkpoint(model, optimizer, epoch, history, best_auc, checkpoint_path)

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            best_path = checkpoints_dir / "quantum_best_model.pth"
            torch.save(model.state_dict(), best_path)
            print(f"  * NEW BEST! AUC: {val_auc:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

        save_timed_checkpoint(model, optimizer, epoch, history, best_auc, checkpoints_dir, interval_seconds=1800)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    total_time = time.time() - total_start
    print(f"\nCompleted in {total_time / 60:.2f} minutes!")
    print(f"Best AUC: {best_auc:.4f}")

    history_path = results_dir / "quantum_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return model, history


if __name__ == "__main__":
    run_experiment()
