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
- cfg.test_subjects is an integer count (or None for full LOSO).  Subjects
  are distributed as evenly as possible across datasets using round-robin
  allocation, so no single dataset is over-represented in the held-out set.
  Each fold is identified by a (dataset_id, subject_id) pair to correctly
  handle the case where the same numeric subject ID appears in multiple
  datasets.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import logging
import random

import torch

from config import Config
from models.eegnet import EEGNet
from data.dataset_interface import P300Dataset, make_loader
from training.trainer import train_model
from evaluation.metrics import evaluate_model, aggregate_metrics, collect_predictions

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Subject selection
# ---------------------------------------------------------------------------

def _select_test_subjects(
    per_dataset_subject_ids: Dict[str, List],
    n_total: int,
    seed: int,
) -> List[Tuple[str, object]]:
    """
    Choose ``n_total`` (dataset_id, subject_id) pairs distributed as evenly
    as possible across datasets.

    Algorithm
    ---------
    1. Shuffle each dataset's subject list independently (seeded).
    2. Allocate slots via round-robin across datasets so the counts differ
       by at most 1.  Datasets with more subjects are preferred when there
       is a remainder (i.e. slots are filled from the front of the sorted
       dataset list first).

    Examples (3 datasets A=10, B=10, C=10, n_total=4)
    --------------------------------------------------
      A gets 2, B gets 1, C gets 1  →  4 folds

    Parameters
    ----------
    per_dataset_subject_ids : {dataset_id: [subject_id, ...]}
    n_total : int
        Total number of held-out subjects wanted.
    seed : int

    Returns
    -------
    List of (dataset_id, subject_id) tuples -- the folds to run.
    """
    dataset_ids = sorted(per_dataset_subject_ids.keys())  # deterministic order
    n_datasets = len(dataset_ids)

    if n_datasets == 0:
        raise ValueError("No datasets found in per_dataset_subject_ids.")

    # Validate n_total against available subjects
    total_available = sum(len(v) for v in per_dataset_subject_ids.values())
    if n_total > total_available:
        raise ValueError(
            f"test_subjects={n_total} requested but only {total_available} "
            f"subjects are available across all datasets."
        )

    # Shuffle each dataset's subjects independently
    rng = random.Random(seed)
    shuffled: Dict[str, List] = {}
    for did in dataset_ids:
        subjs = list(per_dataset_subject_ids[did])
        rng.shuffle(subjs)
        shuffled[did] = subjs

    # Compute per-dataset allocation via round-robin
    # Base quota: floor(n_total / n_datasets), remainder r distributed to
    # the first r datasets (sorted alphabetically for reproducibility).
    base, remainder = divmod(n_total, n_datasets)
    quotas: Dict[str, int] = {}
    for i, did in enumerate(dataset_ids):
        quota = base + (1 if i < remainder else 0)
        # Never allocate more than the dataset actually has
        quota = min(quota, len(shuffled[did]))
        quotas[did] = quota

    # If capping caused us to fall short, redistribute leftover slots to
    # datasets that still have capacity (greedy, front-to-back).
    allocated = sum(quotas.values())
    if allocated < n_total:
        for did in dataset_ids:
            if allocated >= n_total:
                break
            headroom = len(shuffled[did]) - quotas[did]
            extra = min(headroom, n_total - allocated)
            quotas[did] += extra
            allocated += extra

    # Build the final list of (dataset_id, subject_id) folds
    selected: List[Tuple[str, object]] = []
    for did in dataset_ids:
        for subj in shuffled[did][: quotas[did]]:
            selected.append((did, subj))

    # Print a readable allocation report
    print("\n" + "=" * 60)
    print("TEST SUBJECT ALLOCATION")
    print("=" * 60)
    for did in dataset_ids:
        subjs = [s for d, s in selected if d == did]
        print(f"  {did}: {len(subjs)} subject(s) → {subjs}")
    print(f"  Total folds: {len(selected)}")
    print("=" * 60 + "\n")

    return selected


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_loso(
    dataset: P300Dataset,
    cfg: Config,
    per_dataset_subject_ids: Dict[str, List],
) -> List[Dict]:
    """
    Run the full LOSO protocol.

    Parameters
    ----------
    dataset : P300Dataset
        Complete dataset containing all subjects.
    cfg : Config
        Experiment configuration.  cfg.test_subjects is an integer count
        of subjects to hold out (distributed across datasets), or None
        for a full LOSO over every subject.
    per_dataset_subject_ids : dict
        {dataset_id: [subject_id, ...]} -- used to distribute held-out
        subjects evenly across datasets when cfg.test_subjects is an int.

    Returns
    -------
    List of per-subject result dicts, one per held-out fold.
    Each dict has keys: subject_id, dataset_id, metrics, checkpoint_path.
    """
    device = _get_device()
    all_subject_ids = dataset.get_subject_ids()

    print(f"\nTotal subjects available: {len(all_subject_ids)}")
    print(f"Subject IDs: {all_subject_ids}\n")

    # Determine which (dataset_id, subject_id) pairs to hold out
    if cfg.test_subjects is None:
        # Full LOSO -- one fold per unique (dataset_id, subject_id) pair.
        # We reconstruct the pairs from per_dataset_subject_ids so the fold
        # label carries the dataset context.
        test_folds: List[Tuple[str, object]] = [
            (did, sid)
            for did, sids in sorted(per_dataset_subject_ids.items())
            for sid in sids
        ]
        print(f"Full LOSO: holding out all {len(test_folds)} subjects across all datasets.")
    else:
        test_folds = _select_test_subjects(
            per_dataset_subject_ids,
            n_total=cfg.test_subjects,
            seed=cfg.seed,
        )

    print(f"Folds to run: {len(test_folds)}\n")

    all_results = []

    for fold_idx, (held_out_dataset, held_out_subject) in enumerate(test_folds, start=1):
        logger.info("=" * 60)
        logger.info(
            "LOSO fold %d/%d -- held-out subject: %s (dataset: %s)",
            fold_idx, len(test_folds), held_out_subject, held_out_dataset,
        )
        print(
            f"\n--- Fold {fold_idx}/{len(test_folds)}: "
            f"subject {held_out_subject} from {held_out_dataset} ---"
        )

        result = _run_fold(dataset, held_out_subject, held_out_dataset, cfg, device)
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
    held_out_subject: object,
    held_out_dataset: str,
    cfg: Config,
    device: torch.device,
) -> Dict:
    """
    Train and evaluate one LOSO fold.

    The held-out set is filtered to the specific (dataset_id, subject_id)
    pair so that subjects with the same numeric ID in different datasets are
    never confused.

    Returns a result dict with keys:
      subject_id       -- held-out subject ID
      dataset_id       -- dataset the held-out subject belongs to
      metrics          -- test-set metrics including y_true / y_prob arrays
      train_metrics    -- final-epoch train-set metrics (for generalization gap plot)
      training_history -- per-epoch loss and accuracy curves (for training curve plots)
      checkpoint_path  -- path to saved checkpoint, or None
    """
    # Split -- exclude exactly this (dataset_id, subject_id) pair from training.
    # P300Dataset.exclude_subject / filter_by_subject only accept a subject ID,
    # so we filter on both fields manually via the underlying EEGSampleDataset
    # to avoid confusing subjects that share the same numeric ID across datasets.
    from framework.eeg_dataset import EEGSampleDataset

    inner = dataset.source  # the wrapped EEGSampleDataset
    train_samples = [
        s for s in inner.samples
        if not (s.subject_id == held_out_subject and s.dataset_id == held_out_dataset)
    ]
    test_samples = [
        s for s in inner.samples
        if s.subject_id == held_out_subject and s.dataset_id == held_out_dataset
    ]
    train_set = P300Dataset(EEGSampleDataset(train_samples))
    test_set  = P300Dataset(EEGSampleDataset(test_samples))

    logger.info(
        "Subject %s [%s]: %d train samples, %d test samples",
        held_out_subject, held_out_dataset, len(train_set), len(test_set),
    )

    if len(train_set) == 0:
        logger.warning(
            "No training data for fold (held-out=%s, dataset=%s) -- skipping",
            held_out_subject, held_out_dataset,
        )
        return {
            "subject_id": held_out_subject,
            "dataset_id": held_out_dataset,
            "metrics": {},
            "checkpoint_path": None,
        }

    # Data loaders
    # balanced=True helps with the typical P300 class imbalance (~80/20 split)
    train_loader = make_loader(train_set, cfg.batch_size, shuffle=True,  balanced=True,  num_workers=cfg.num_workers)
    test_loader  = make_loader(test_set,  cfg.batch_size, shuffle=False, balanced=False, num_workers=cfg.num_workers)

    # Fresh model for each fold
    model = EEGNet(n_channels=cfg.n_channels, n_timepoints=cfg.n_timepoints).to(device)

    checkpoint_path = (
        cfg.checkpoints_dir() / f"subject_{held_out_dataset}_{held_out_subject}.pt"
        if cfg.save_checkpoints else None
    )

    # Train -- history is a dict of lists keyed by metric name
    history = train_model(
        model=model,
        train_loader=train_loader,
        n_epochs=cfg.n_epochs,
        learning_rate=cfg.learning_rate,
        device=device,
        val_loader=test_loader,
        checkpoint_path=checkpoint_path,
    )

    # Test-set evaluation
    y_true, y_pred, y_prob = collect_predictions(model, test_loader, device)
    metrics = evaluate_model(model, test_loader, device)
    metrics["y_true"] = y_true.tolist()
    metrics["y_prob"] = y_prob.tolist()

    # Train-set evaluation (final epoch, unshuffled) -- used for generalization gap
    train_eval_loader = make_loader(
        train_set, cfg.batch_size, shuffle=False, balanced=False,
        num_workers=cfg.num_workers,
    )
    train_metrics = evaluate_model(model, train_eval_loader, device)
    train_metrics.pop("confusion_matrix", None)

    return {
        "subject_id":       held_out_subject,
        "dataset_id":       held_out_dataset,
        "metrics":          metrics,
        "train_metrics":    train_metrics,
        "training_history": history if isinstance(history, dict) else {},
        "checkpoint_path":  str(checkpoint_path) if checkpoint_path else None,
    }


# ---------------------------------------------------------------------------
# Saving and printing
# ---------------------------------------------------------------------------

def _save_subject_result(result: Dict, cfg: Config) -> None:
    cfg.logs_dir().mkdir(parents=True, exist_ok=True)
    # Include dataset_id in filename to avoid collisions when the same
    # numeric subject ID appears in multiple datasets.
    path = cfg.logs_dir() / f"subject_{result['dataset_id']}_{result['subject_id']}.json"
    serialisable = {
        k: v for k, v in result.items()
        if k not in ("training_history",)
    }
    serialisable["metrics"] = {
        k: v for k, v in result.get("metrics", {}).items()
        if k not in ("y_true", "y_prob")
    }
    with open(path, "w") as f:
        json.dump(serialisable, f, indent=2)
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
        print(
            f"  Subject {result['subject_id']} [{result.get('dataset_id', '?')}]: "
            f"no results (insufficient data)"
        )
        return
    mcc = result.get("metrics", {}).get("mcc", float("nan"))
    if mcc != mcc:  # nan check
        cm = m.get("confusion_matrix")
        if cm and len(cm) == 2:
            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            denom = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) ** 0.5
            mcc = (tp*tn - fp*fn) / denom if denom else float("nan")
    print(f"  Subject {result['subject_id']} [{result.get('dataset_id', '?')}]:")
    print(f"    Accuracy:          {m.get('accuracy',          float('nan')):.4f}")
    print(f"    Balanced Accuracy: {m.get('balanced_accuracy', float('nan')):.4f}")
    print(f"    F1 (macro):        {m.get('f1_macro',          float('nan')):.4f}")
    print(f"    MCC:               {mcc:.4f}")
    print(f"    ROC-AUC:           {m.get('roc_auc',           float('nan')):.4f}")


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
        logger.info("GPU not available -- using CPU")
    return device