#!/usr/bin/env python
# ============================================================================
# TRAINING UTILITIES FOR EXPERIMENT 7 (Track B)
# Provides training loop, validation metrics, and model checkpointing.
# ============================================================================

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from exp07.utils.config import DEVICE

_LAST_TIMED_SAVE = time.time()


def train_one_epoch(model, train_loader, optimizer, criterion=None):
    """
    Execute a single training epoch across graph batches.
    
    Returns:
        mean_loss (float), accuracy (float)
    """
    model.train()
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch in train_loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()

        logits = model(batch)
        loss = criterion(logits, batch.y)
        loss.backward()
        optimizer.step()

        batch_size = batch.y.size(0)
        total_loss += loss.item() * batch_size
        preds = logits.argmax(dim=-1)
        correct += (preds == batch.y).sum().item()
        total += batch_size

    mean_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return mean_loss, accuracy


def validate(model, val_loader):
    """
    Evaluate model performance on a validation or test set.
    
    Returns:
        accuracy (float), f1 (float), auc (float)
    """
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(DEVICE)
            logits = model(batch)
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            all_targets.extend(batch.y.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    acc = float(accuracy_score(y_true, y_pred)) if len(y_true) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, average='weighted', zero_division=0)) if len(y_true) > 0 else 0.0

    try:
        if len(np.unique(y_true)) > 1:
            auc = float(roc_auc_score(y_true, y_prob))
        else:
            auc = 0.5
    except Exception:
        auc = 0.5

    return acc, f1, auc


def save_checkpoint(model, optimizer, epoch, history, best_auc, checkpoint_path):
    """Save model and optimizer state dictionary to disk."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict() if optimizer else None,
        'history': history,
        'best_auc': best_auc,
        'timestamp': time.time()
    }
    torch.save(state, checkpoint_path)


def save_timed_checkpoint(model, optimizer, epoch, history, best_auc, checkpoint_dir, interval_seconds=1800):
    """Save a periodic checkpoint if interval_seconds have passed since last save."""
    global _LAST_TIMED_SAVE
    current_time = time.time()
    if current_time - _LAST_TIMED_SAVE >= interval_seconds:
        timed_path = Path(checkpoint_dir) / f"timed_checkpoint_epoch_{epoch}.pth"
        save_checkpoint(model, optimizer, epoch, history, best_auc, timed_path)
        _LAST_TIMED_SAVE = current_time
