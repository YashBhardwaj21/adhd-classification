#!/usr/bin/env python
# Training utilities for Experiment 7 (Track B).
# Training loop, evaluation metrics, and model checkpointing.

import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

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

    device = next(model.parameters()).device
    for batch in train_loader:
        batch = batch.to(device)
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


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Compute classification metrics with explicit averaging conventions.

    Historical result artifacts (results/exp07/checkpoint_analysis.json) record
    weighted precision, recall, and F1. Both weighted and macro values are
    reported here to avoid ambiguity.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    if len(y_true) == 0:
        return {
            "accuracy": 0.0,
            "precision_weighted": 0.0,
            "recall_weighted": 0.0,
            "f1_weighted": 0.0,
            "precision_macro": 0.0,
            "recall_macro": 0.0,
            "f1_macro": 0.0,
            "auc": 0.5,
        }

    acc = float(accuracy_score(y_true, y_pred))
    p_wt = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    r_wt = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_wt = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    p_ma = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    r_ma = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_ma = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    try:
        if len(np.unique(y_true)) > 1:
            auc = float(roc_auc_score(y_true, y_prob))
        else:
            auc = 0.5
    except ValueError:
        auc = 0.5

    return {
        "accuracy": acc,
        "precision_weighted": p_wt,
        "recall_weighted": r_wt,
        "f1_weighted": f1_wt,
        "precision_macro": p_ma,
        "recall_macro": r_ma,
        "f1_macro": f1_ma,
        "auc": auc,
    }


def validate(model, val_loader) -> Tuple[float, float, float]:
    """
    Evaluate model performance on a validation or test set.

    Returns:
        accuracy (float), f1_weighted (float), auc (float)
    """
    metrics = evaluate_full(model, val_loader)
    return metrics["accuracy"], metrics["f1_weighted"], metrics["auc"]


def evaluate_full(model, loader) -> Dict[str, Any]:
    """
    Full model evaluation collecting true labels, predictions, probabilities, and metrics.
    """
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []

    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = F.softmax(logits, dim=-1)
            preds = logits.argmax(dim=-1)

            all_targets.extend(batch.y.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs[:, 1].cpu().numpy().tolist())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)

    metrics = evaluate_predictions(y_true, y_pred, y_prob)
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    metrics["y_prob"] = y_prob
    return metrics


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


def save_timed_checkpoint(model, optimizer, epoch, history, best_auc, checkpoint_dir, interval_seconds=120):
    """Periodically write checkpoint to safeguard against execution interruptions."""
    global _LAST_TIMED_SAVE
    current_time = time.time()
    if (current_time - _LAST_TIMED_SAVE) >= interval_seconds:
        timed_path = Path(checkpoint_dir) / f"checkpoint_epoch_{epoch}.pth"
        save_checkpoint(model, optimizer, epoch, history, best_auc, timed_path)
        _LAST_TIMED_SAVE = current_time
