#!/usr/bin/env python
# ============================================================================
# RESUME QUANTUM TRAINING FROM CHECKPOINT
# ============================================================================

import gc
import glob
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

from common.paths import checkpoints_root, data_root, results_root
from exp07.quantum_models.train_qgcnn_vectorized import HybridQGCNN_Vectorized, create_quantum_device
from exp07.utils.config import (
    BATCH_SIZE,
    DEVICE,
    EPOCHS,
    LEARNING_RATE,
    N_QUBITS,
    PATIENCE,
    RANDOM_SEED,
    WEIGHT_DECAY,
    print_config,
)
from exp07.utils.data_loader import load_data, split_data
from exp07.utils.graph_utils import prepare_graphs
from exp07.utils.training_utils import save_checkpoint, train_one_epoch, validate


def resume_experiment(
    data_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    checkpoints_dir: Optional[Path] = None,
    seed: int = RANDOM_SEED,
    quantum_device: str = "lightning.gpu",
) -> Tuple[Optional[nn.Module], Optional[Dict]]:
    data_dir = Path(data_dir) if data_dir is not None else data_root()
    results_dir = Path(results_dir) if results_dir is not None else results_root() / "exp07"
    checkpoints_dir = Path(checkpoints_dir) if checkpoints_dir is not None else checkpoints_root() / "exp07"

    print_config()
    print("\n" + "=" * 60)
    print("RESUMING QUANTUM TRAINING")
    print("=" * 60)

    # Search for latest checkpoint in checkpoints_dir, then fallback to data_dir
    checkpoint_files = sorted(glob.glob(str(checkpoints_dir / "quantum_checkpoint_epoch_*.pth")))
    if not checkpoint_files:
        checkpoint_files = sorted(glob.glob(str(data_dir / "quantum_checkpoint_epoch_*.pth")))

    if not checkpoint_files:
        print(f"No checkpoint found in {checkpoints_dir} or {data_dir}. Start a fresh training run.")
        return None, None

    latest = checkpoint_files[-1]
    print(f"Found checkpoint: {latest}")

    checkpoint = torch.load(latest, map_location=DEVICE)
    start_epoch = checkpoint["epoch"] + 1
    best_auc = checkpoint.get("best_auc", -1.0)
    history = checkpoint.get("history", {"train_loss": [], "train_acc": [], "val_acc": [], "val_f1": [], "val_auc": []})

    print(f"   Resuming from epoch {start_epoch}")
    print(f"   Best AUC so far: {best_auc:.4f}")

    dev = create_quantum_device(quantum_device, N_QUBITS)
    print(f"Quantum device: {dev}")

    # Load data
    print("\nLoading data...")
    X_full, _, y, subjects = load_data(data_dir=data_dir)
    data = split_data(X_full, y, subjects, seed=seed)

    # Build graphs
    print("\nBuilding graphs...")
    train_graphs = prepare_graphs(data["X_train"], data["y_train"])
    val_graphs = prepare_graphs(data["X_val"], data["y_val"])

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    model = HybridQGCNN_Vectorized(dev).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    patience_counter = 0
    print("\n" + "=" * 60)
    print(f"CONTINUING FROM EPOCH {start_epoch}")
    print("=" * 60)

    total_start = time.time()

    for epoch in range(start_epoch, EPOCHS + 1):
        epoch_start = time.time()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer)

        if epoch % 5 == 0 or epoch == start_epoch:
            val_acc, val_f1, val_auc = validate(model, val_loader)
        else:
            val_acc = history["val_acc"][-1] if history["val_acc"] else 0.0
            val_f1 = history["val_f1"][-1] if history["val_f1"] else 0.0
            val_auc = history["val_auc"][-1] if history["val_auc"] else 0.0

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

        if epoch % 10 == 0:
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
    resume_experiment()
