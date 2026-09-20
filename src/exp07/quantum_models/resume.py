#!/usr/bin/env python
# ============================================================================
# RESUME QUANTUM TRAINING FROM CHECKPOINT
# ============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc
import time
import json
import glob
import torch
import torch.optim as optim
import numpy as np
import pennylane as qml
from torch_geometric.loader import DataLoader

from utils.config import (
    DEVICE, DATA_DIR, N_QUBITS, N_LAYERS, NODE_FEATURE_DIM,
    BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE,
    CHECKPOINT_INTERVAL, print_config
)
from utils.data_loader import load_data, split_data
from utils.graph_utils import prepare_graphs
from utils.training_utils import train_one_epoch, validate, save_checkpoint
from quantum_models.train_qgcnn_vectorized import HybridQGCNN_Vectorized


def resume_experiment():
    print_config()
    print("\n" + "="*60)
    print("RESUMING QUANTUM TRAINING")
    print("="*60)

    # Find latest checkpoint
    checkpoint_files = sorted(glob.glob(str(DATA_DIR / "quantum_checkpoint_epoch_*.pth")))
    
    if not checkpoint_files:
        print("❌ No checkpoint found. Start fresh.")
        return None, None
    
    latest = checkpoint_files[-1]
    print(f"✅ Found checkpoint: {latest}")
    
    checkpoint = torch.load(latest)
    start_epoch = checkpoint['epoch'] + 1
    best_auc = checkpoint.get('best_auc', -1.0)
    history = checkpoint['history']
    
    print(f"   Resuming from epoch {start_epoch}")
    print(f"   Best AUC so far: {best_auc:.4f}")

    # Setup quantum device
    dev = qml.device("lightning.gpu", wires=N_QUBITS, batch_obs=True, c_dtype=np.complex64)
    print(f"Quantum device: {dev}")

    # Load data
    print("\nLoading data...")
    X_full, _, y, subjects = load_data()
    data = split_data(X_full, y, subjects)

    # Build graphs
    print("\nBuilding graphs...")
    train_graphs = prepare_graphs(data['X_train'], data['y_train'])
    val_graphs = prepare_graphs(data['X_val'], data['y_val'])
    test_graphs = prepare_graphs(data['X_test'], data['y_test'])

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    model = HybridQGCNN_Vectorized(dev).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    patience_counter = 0
    print("\n" + "="*60)
    print(f"CONTINUING FROM EPOCH {start_epoch}")
    print("="*60)

    total_start = time.time()

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start = time.time()
        torch.cuda.empty_cache()
        gc.collect()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
        
        if epoch % 5 == 0 or epoch == 1:
            val_acc, val_f1, val_auc = validate(model, val_loader)
        else:
            val_acc, val_f1, val_auc = history['val_acc'][-1], history['val_f1'][-1], history['val_auc'][-1]

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

        if epoch % 10 == 0:
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

        torch.cuda.empty_cache()
        gc.collect()

    total_time = time.time() - total_start
    print(f"\n✅ Completed in {total_time/60:.2f} minutes!")
    print(f"Best AUC: {best_auc:.4f}")

    return model, history


if __name__ == "__main__":
    model, history = resume_experiment()