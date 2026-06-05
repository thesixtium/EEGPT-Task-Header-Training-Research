"""
training/loso_runner.py
------------------------
Leave-One-Subject-Out (LOSO) evaluation protocol.

For each held-out subject:
  1. Build a train set from all other subjects.
  2. Train a fresh EEGNet from scratch.
  3. Evaluate on the held-out subject.
  4. Save per-subject metrics and checkpoint.

After all folds complete, aggregate metrics across subjects and print a
summary.

Design notes
------------
- A fresh model is created for each fold (no cross-contamination).
- The evaluator is imported from evaluation/metrics.py to keep the training
  and metrics concerns separated.
- All results are returned as plain dicts so they are easy to serialise
  or pass to the analytics utilities.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import json
import logging

import torch

from config import Config
from models.eegnet import EEGNet
from data.dataset_interface import P300Dataset, make_loader
from training.trainer import train_model
from evaluation.metrics import evaluate_model, aggregate_metrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_loso(
    dataset: P300Dataset,
    cfg: Config,
) -> List[Dict]:
    """
    Run the full LOSO protocol.

    Parameters
    ----------
    dataset : P300Dataset
        Complete dataset containing all subjects.
    cfg : Config
        Experiment configuration.

    Returns
    -------
    List of per-subject result dicts, one per held-out subject.
    Each dict has keys: subject_id, metrics, checkpoint_path.
    """
    device = _get_device()
    all_subject_ids = dataset.get_subject_ids()

    print(f"\nTotal subjects available: {len(all_subject_ids)}")
    print(f"Subject IDs: {all_subject_ids}\n")

    # Determine which subjects to hold out
    if cfg.test_subjects is not None:
        test_subjects = cfg.test_subjects
        invalid = [s for s in test_subjects if s not in all_subject_ids]
        if invalid:
            raise ValueError(f"test_subjects contains IDs not found in dataset: {invalid}")
    else:
        test_subjects = all_subject_ids

    print(f"Held-out subjects for LOSO: {test_subjects}")
    print(f"Folds to run: {len(test_subjects)}\n")

    all_results = []

    for fold_idx, held_out in enumerate(test_subjects, start=1):
        logger.info("=" * 60)
        logger.info("LOSO fold %d/%d — held-out subject: %d", fold_idx, len(test_subjects), held_out)
        print(f"\n--- Fold {fold_idx}/{len(test_subjects)}: held-out subject {held_out} ---")

        result = _run_fold(dataset, held_out, cfg, device)
        all_results.append(result)

        _save_subject_result(result, cfg)
        _print_subject_summary(result)

    # Aggregate across all folds
    print("\n" + "=" * 60)
    print("LOSO AGGREGATE RESULTS")
    print("=" * 60)
    aggregate = aggregate_metrics([r["metrics"] for r in all_results])
    _print_aggregate_summary(aggregate)
    _save_aggregate_results(all_results, aggregate, cfg)

    return all_results


# ---------------------------------------------------------------------------
# Single fold
# ---------------------------------------------------------------------------

def _run_fold(
    dataset: P300Dataset,
    held_out_subject: int,
    cfg: Config,
    device: torch.device,
) -> Dict:
    """Train and evaluate one LOSO fold."""
    # Split
    train_set = dataset.exclude_subject(held_out_subject)
    test_set = dataset.filter_by_subject(held_out_subject)

    logger.info(
        "Subject %d: %d train samples, %d test samples",
        held_out_subject, len(train_set), len(test_set),
    )

    if len(train_set) == 0:
        logger.warning("No training data for fold (held-out=%d) — skipping", held_out_subject)
        return {"subject_id": held_out_subject, "metrics": {}, "checkpoint_path": None}

    # Data loaders
    # balanced=True helps with the typical P300 class imbalance (~80/20 split)
    train_loader = make_loader(train_set, cfg.batch_size, shuffle=True,  balanced=True,  num_workers=cfg.num_workers)
    test_loader  = make_loader(test_set,  cfg.batch_size, shuffle=False, balanced=False, num_workers=cfg.num_workers)

    # Fresh model for each fold
    model = EEGNet(n_channels=cfg.n_channels, n_timepoints=cfg.n_timepoints).to(device)

    checkpoint_path = (
        cfg.checkpoints_dir() / f"subject_{held_out_subject}.pt"
        if cfg.save_checkpoints else None
    )

    # Train
    train_model(
        model=model,
        train_loader=train_loader,
        n_epochs=cfg.n_epochs,
        learning_rate=cfg.learning_rate,
        device=device,
        val_loader=test_loader,
        checkpoint_path=checkpoint_path,
    )

    # Evaluate
    metrics = evaluate_model(model, test_loader, device)

    return {
        "subject_id": held_out_subject,
        "metrics": metrics,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
    }


# ---------------------------------------------------------------------------
# Saving and printing
# ---------------------------------------------------------------------------

def _save_subject_result(result: Dict, cfg: Config) -> None:
    cfg.logs_dir().mkdir(parents=True, exist_ok=True)
    path = cfg.logs_dir() / f"subject_{result['subject_id']}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Subject result saved to %s", path)


def _save_aggregate_results(all_results: List[Dict], aggregate: Dict, cfg: Config) -> None:
    cfg.logs_dir().mkdir(parents=True, exist_ok=True)
    path = cfg.logs_dir() / "loso_summary.json"
    summary = {
        "per_subject": all_results,
        "aggregate": aggregate,
    }
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("LOSO summary saved to %s", path)


def _print_subject_summary(result: Dict) -> None:
    m = result["metrics"]
    if not m:
        print(f"  Subject {result['subject_id']}: no results (insufficient data)")
        return
    print(f"  Subject {result['subject_id']}:")
    print(f"    Accuracy:          {m.get('accuracy', float('nan')):.4f}")
    print(f"    Balanced Accuracy: {m.get('balanced_accuracy', float('nan')):.4f}")
    print(f"    F1 (macro):        {m.get('f1_macro', float('nan')):.4f}")
    print(f"    ROC-AUC:           {m.get('roc_auc', float('nan')):.4f}")


def _print_aggregate_summary(aggregate: Dict) -> None:
    for metric, stats in aggregate.items():
        print(f"  {metric}:")
        print(f"    mean={stats['mean']:.4f}  std={stats['std']:.4f}  "
              f"min={stats['min']:.4f}  max={stats['max']:.4f}")


def _get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using GPU: %s", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        logger.info("GPU not available — using CPU")
    return device
