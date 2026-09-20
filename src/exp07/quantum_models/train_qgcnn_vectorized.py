#!/usr/bin/env python
# ============================================================================
# TRAIN QGCNN - VECTORIZED VERSION (BROADCASTING)
# Uses PennyLane broadcasting for 50-100x speedup
# ============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.training_utils import save_timed_checkpoint
import numpy as np
import gc
import time
import json
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import pennylane as qml

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

from utils.config import (
    DEVICE, DATA_DIR, N_QUBITS, N_LAYERS, NODE_FEATURE_DIM,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE,
    CHECKPOINT_INTERVAL, print_config
)
from utils.data_loader import load_data, split_data
from utils.graph_utils import prepare_graphs
from utils.training_utils import train_one_epoch, validate, save_checkpoint, save_timed_checkpoint
#from utils.training_utils import train_one_epoch, validate, save_checkpoint
from quantum_models.quantum_embedding_broadcast import QuantumEmbeddingGPU_Broadcast


class HybridQGCNN_Vectorized(nn.Module):
    def __init__(self, dev, input_dim=NODE_FEATURE_DIM, n_qubits=N_QUBITS,
                 n_layers=N_LAYERS, hidden_dim=16):
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


def run_experiment():
    print_config()
    print("\n" + "="*60)
    print("VECTORIZED QUANTUM GCNN (BROADCASTING)")
    print("="*60)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*60)

    dev = qml.device("lightning.gpu", wires=N_QUBITS, batch_obs=True, c_dtype=np.complex64)
    print(f"Quantum device: {dev}")

    print("\nLoading data...")
    X_full, _, y, subjects = load_data()
    data = split_data(X_full, y, subjects)

    print("\nBuilding graphs...")
    train_graphs = prepare_graphs(data['X_train'], data['y_train'])
    val_graphs = prepare_graphs(data['X_val'], data['y_val'])
    test_graphs = prepare_graphs(data['X_test'], data['y_test'])

    print(f"Train: {len(train_graphs)}, Val: {len(val_graphs)}, Test: {len(test_graphs)}")

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    model = HybridQGCNN_Vectorized(dev).to(DEVICE)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    history = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'val_f1': [], 'val_auc': []}
    best_auc = -1.0
    patience_counter = 0

    print("\n" + "="*60)
    print("STARTING TRAINING (BROADCASTING)")
    print("="*60)

    total_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        torch.cuda.empty_cache()
        gc.collect()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
        val_acc, val_f1, val_auc = validate(model, val_loader)

        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)

        epoch_time = time.time() - epoch_start
        mem_alloc = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0

        print(f"\nEpoch {epoch:03d}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"  Val Acc: {val_acc:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}")
        print(f"  GPU: {mem_alloc:.2f}GB | LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")

        if epoch % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = DATA_DIR / f"quantum_checkpoint_epoch_{epoch}.pth"
            save_checkpoint(model, optimizer, epoch, history, best_auc, checkpoint_path)

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), DATA_DIR / "quantum_best_model.pth")
            print(f"  ★ NEW BEST! AUC: {val_auc:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\n⚠️ Early stopping at epoch {epoch}")
            break
        save_timed_checkpoint(model, optimizer, epoch, history, best_auc, DATA_DIR, interval_seconds=1800)
        torch.cuda.empty_cache()
        gc.collect()

    total_time = time.time() - total_start
    print(f"\n✅ Completed in {total_time/60:.2f} minutes!")
    print(f"Best AUC: {best_auc:.4f}")

    with open(DATA_DIR / 'quantum_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    return model, history


if __name__ == "__main__":
    model, history = run_experiment()