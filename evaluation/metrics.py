"""
evaluation/metrics.py
----------------------
All metric computation for EEGNet evaluation.

Separated from the training code so metrics can be computed independently
(e.g. loading a checkpoint and re-evaluating without retraining).

Metrics computed per subject:
  - Accuracy
  - Balanced Accuracy
  - Precision (macro)
  - Recall (macro)
  - F1 (macro)
  - ROC-AUC
  - Confusion Matrix

Aggregate metrics across LOSO folds:
  - Mean, std, min, max of each scalar metric across subjects
"""

from __future__ import annotations
from typing import Dict, List, Tuple
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference and collect ground-truth labels, hard predictions,
    and raw probabilities.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader — yields (x, y) batches
    device : torch.device

    Returns
    -------
    y_true  : np.ndarray[int],   shape [N]
    y_pred  : np.ndarray[int],   shape [N]  (thresholded at 0.5)
    y_prob  : np.ndarray[float], shape [N]  (raw sigmoid output)
    """
    model.eval()
    all_true, all_pred, all_prob = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            prob = model(x).cpu()          # [batch], values in (0, 1)
            pred = (prob >= 0.5).long()

            all_true.extend(y.long().tolist())
            all_pred.extend(pred.tolist())
            all_prob.extend(prob.tolist())

    return (
        np.array(all_true, dtype=int),
        np.array(all_pred, dtype=int),
        np.array(all_prob, dtype=float),
    )


# ---------------------------------------------------------------------------
# Per-subject metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict:
    """
    Compute all classification metrics from arrays of labels/predictions.

    Returns a dict with scalar metrics and a nested confusion matrix.
    """
    # Guard against degenerate cases (only one class present in labels)
    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = float("nan")
        logger.warning("ROC-AUC undefined (only one class present in y_true)")

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision":         float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall":            float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro":          float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc":           roc_auc,
        "confusion_matrix":  cm,
        "n_samples":         int(len(y_true)),
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict:
    """
    Run inference and return metrics.

    Convenience wrapper around collect_predictions + compute_metrics.
    """
    y_true, y_pred, y_prob = collect_predictions(model, loader, device)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    logger.info(
        "Evaluation — acc: %.4f  bal_acc: %.4f  f1: %.4f  auc: %.4f  n=%d",
        metrics["accuracy"],
        metrics["balanced_accuracy"],
        metrics["f1_macro"],
        metrics["roc_auc"],
        metrics["n_samples"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Aggregate across LOSO folds
# ---------------------------------------------------------------------------

# Metrics to aggregate (confusion_matrix and n_samples are excluded from
# the scalar summary — they are kept per-subject only).
_SCALAR_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1_macro",
    "roc_auc",
]


def aggregate_metrics(per_subject_metrics: List[Dict]) -> Dict:
    """
    Compute mean, std, min, max of each scalar metric across subjects.

    Parameters
    ----------
    per_subject_metrics : list of dicts, one per LOSO fold.
        Each dict must have the keys in _SCALAR_METRICS (NaN values are
        excluded from the aggregation).

    Returns
    -------
    Dict mapping metric name → {"mean", "std", "min", "max"}.
    """
    aggregate = {}
    for metric in _SCALAR_METRICS:
        values = [
            m[metric]
            for m in per_subject_metrics
            if m and not np.isnan(m.get(metric, float("nan")))
        ]
        if not values:
            aggregate[metric] = {"mean": float("nan"), "std": float("nan"),
                                  "min": float("nan"), "max": float("nan")}
        else:
            arr = np.array(values)
            aggregate[metric] = {
                "mean": float(np.mean(arr)),
                "std":  float(np.std(arr)),
                "min":  float(np.min(arr)),
                "max":  float(np.max(arr)),
            }
    return aggregate
