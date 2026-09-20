#!/usr/bin/env python
# ============================================================================
# TRAIN CLASSICAL GCN - Fast baseline run on GPU
# ============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gc
import time
import json
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader


from utils.config import (
    DEVICE, DATA_DIR, BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DECAY, PATIENCE,
    CHECKPOINT_INTERVAL, print_config
)
from utils.data_loader import load_data, split_data
from utils.graph_utils import prepare_graphs
#from utils.training_utils import train_one_epoch, validate, save_checkpoint
from utils.training_utils import train_one_epoch, validate, save_checkpoint, save_timed_checkpoint

from classical_models.model_classical_gcn import ClassicalGCN


def run_experiment():
    """Run classical GCN training experiment."""
    
    print_config()
    print("\n" + "="*60)
    print("CLASSICAL GCN EXPERIMENT")
    print("="*60)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    X_full, _, y, subjects = load_data()
    data = split_data(X_full, y, subjects)
    
    # Build graphs
    print("\nBuilding graphs...")
    train_graphs = prepare_graphs(data['X_train'], data['y_train'])
    val_graphs = prepare_graphs(data['X_val'], data['y_val'])
    test_graphs = prepare_graphs(data['X_test'], data['y_test'])
    
    print(f"Training graphs: {len(train_graphs)}")
    print(f"Validation graphs: {len(val_graphs)}")
    print(f"Test graphs: {len(test_graphs)}")
    
    # Dataloaders
    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Initialize model
    model = ClassicalGCN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_acc': [], 'val_f1': [], 'val_auc': []
    }
    
    best_auc = -1.0
    patience_counter = 0
    
    print("\n" + "="*60)
    print("STARTING CLASSICAL GCN TRAINING")
    print("="*60)
    
    total_start = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()
        
        # Clear memory
        torch.cuda.empty_cache()
        gc.collect()
        
        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)
        
        # Validate
        val_acc, val_f1, val_auc = validate(model, val_loader)
        
        # Scheduler
        scheduler.step(val_auc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # History
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)
        
        epoch_time = time.time() - epoch_start
        
        # GPU memory
        if torch.cuda.is_available():
            mem_alloc = torch.cuda.memory_allocated() / 1024**3
            mem_reserved = torch.cuda.memory_reserved() / 1024**3
            mem_info = f"GPU: {mem_alloc:.2f}GB/{mem_reserved:.2f}GB"
        else:
            mem_info = "CPU"
        
        print(f"\nEpoch {epoch:03d}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"  Val Acc: {val_acc:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}")
        print(f"  {mem_info} | LR: {current_lr:.2e}")
        print(f"  Time: {epoch_time:.1f}s")
        
        # Checkpoint
        if epoch % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = DATA_DIR / f"classical_checkpoint_epoch_{epoch}.pth"
            save_checkpoint(model, optimizer, epoch, history, best_auc, checkpoint_path)
        
        # Best model
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), DATA_DIR / "classical_best_model.pth")
            print(f"  ★ NEW BEST! AUC: {val_auc:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n⚠️ Early stopping at epoch {epoch}")
            break
        save_timed_checkpoint(model, optimizer, epoch, history, best_auc, DATA_DIR, interval_seconds=1800)
    total_time = time.time() - total_start
    print(f"\n✅ Classical GCN completed in {total_time/60:.2f} minutes!")
    print(f"Best Validation AUC: {best_auc:.4f}")
    
    # Save history
    with open(DATA_DIR / 'classical_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    return model, history


if __name__ == "__main__":
    model, history = run_experiment()