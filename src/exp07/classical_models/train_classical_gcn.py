#!/usr/bin/env python
# Train Classical GCN baseline model.

import gc
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader

from exp07.classical_models.model_classical_gcn import ClassicalGCN
from exp07.utils.config import (
    BATCH_SIZE,
    CHECKPOINT_INTERVAL,
    CHECKPOINTS_DIR,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    PATIENCE,
    RESULTS_DIR,
    SEED,
    WEIGHT_DECAY,
    print_config,
)
from exp07.utils.data_loader import load_data, split_data
from exp07.utils.graph_utils import prepare_graphs
from exp07.utils.training_utils import save_checkpoint, save_timed_checkpoint, train_one_epoch, validate


def run_experiment(
    data_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    checkpoints_dir: Optional[Path] = None,
    device: Optional[torch.device] = None,
    seed: int = SEED,
) -> Tuple[ClassicalGCN, Dict[str, Any]]:
    """
    Run classical GCN training experiment.
    
    Args:
        data_dir: Directory containing historical X_combined_full.npy,
            y_combined.npy, and subjects_combined.npy.
        results_dir: Directory to save metrics and history.
        checkpoints_dir: Directory to save model checkpoints.
        device: Torch compute device (default: autodetect CUDA/CPU).
        seed: Random seed used for dataset splitting and RNG initialization (default: 42).
        
    Returns:
        model: Trained ClassicalGCN model.
        history: Training and validation metric history dictionary.
    """
    active_device = device if device is not None else DEVICE
    active_results_dir = Path(results_dir) if results_dir is not None else RESULTS_DIR
    active_checkpoints_dir = Path(checkpoints_dir) if checkpoints_dir is not None else CHECKPOINTS_DIR

    # Seed initialization
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print_config()
    print("\nClassical GCN Experiment (Track B)")
    print(f"Device: {active_device}")
    if torch.cuda.is_available() and active_device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load real data (fails loudly if absent)
    print("\nLoading data...")
    X, y, subjects = load_data(data_dir=data_dir)
    data = split_data(X, y, subjects, random_state=seed)

    # Build graphs
    print("\nBuilding graphs...")
    train_graphs = prepare_graphs(data['X_train'], data['y_train'])
    val_graphs = prepare_graphs(data['X_val'], data['y_val'])
    test_graphs = prepare_graphs(data['X_test'], data['y_test'])

    print(f"Training graphs:   {len(train_graphs)}")
    print(f"Validation graphs: {len(val_graphs)}")
    print(f"Test graphs:       {len(test_graphs)}")

    # Dataloaders
    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model
    model = ClassicalGCN().to(active_device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_acc': [], 'val_f1': [], 'val_auc': []
    }

    best_auc = -1.0
    patience_counter = 0

    # Ensure directories exist only upon execution
    active_results_dir.mkdir(parents=True, exist_ok=True)
    active_checkpoints_dir.mkdir(parents=True, exist_ok=True)

    print("\nStarting Classical GCN Training")

    total_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        if torch.cuda.is_available():
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

        print(f"\nEpoch {epoch:03d}/{EPOCHS}")
        print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"  Val Acc: {val_acc:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.4f}")
        print(f"  Time: {epoch_time:.1f}s")

        # Periodic Checkpoint
        if epoch % CHECKPOINT_INTERVAL == 0:
            checkpoint_path = active_checkpoints_dir / f"classical_checkpoint_epoch_{epoch}.pth"
            save_checkpoint(model, optimizer, epoch, history, best_auc, checkpoint_path)

        # Best model
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            best_model_path = active_checkpoints_dir / "classical_best_model.pth"
            torch.save(model.state_dict(), best_model_path)
            print(f"  * NEW BEST! AUC: {val_auc:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n[Early stopping at epoch {epoch}]")
            break

        save_timed_checkpoint(model, optimizer, epoch, history, best_auc, active_checkpoints_dir, interval_seconds=1800)

    total_time = time.time() - total_start
    print(f"\nClassical GCN completed in {total_time/60:.2f} minutes.")
    print(f"Best Validation AUC: {best_auc:.4f}")

    # Save training history to results directory
    history_file = active_results_dir / 'classical_history.json'
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Saved history to: {history_file}")

    # Evaluate best model on held-out test cohort
    best_model_path = active_checkpoints_dir / "classical_best_model.pth"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=active_device))
        test_acc, test_f1, test_auc = validate(model, test_loader)
        print(f"\nHeld-out Test Evaluation (N={len(test_graphs)}):")
        print(f"  Test Accuracy: {test_acc:.4f} | F1: {test_f1:.4f} | AUC: {test_auc:.4f}")

    return model, history


if __name__ == "__main__":
    run_experiment()
