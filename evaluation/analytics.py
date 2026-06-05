"""
evaluation/analytics.py
------------------------
Reporting and visualisation for LOSO results.

Adapted from the existing EEGPT analytics.py.  The core metric computation
stays in metrics.py; this module handles presenting results to the researcher:
  - Console summary tables
  - Per-subject confusion matrix plots
  - Aggregate summary CSV

All matplotlib calls are guarded so the module works in headless environments.
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional
import csv
import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False
    logger.warning("matplotlib not available — plots will be skipped")


# ---------------------------------------------------------------------------
# Console reporting
# ---------------------------------------------------------------------------

def print_loso_table(all_results: List[Dict]) -> None:
    """
    Print a formatted table of per-subject metrics to stdout.

    Parameters
    ----------
    all_results : list of per-subject result dicts from loso_runner.run_loso.
    """
    header = f"{'Subject':>8}  {'Acc':>8}  {'Bal Acc':>8}  {'F1':>8}  {'AUC':>8}  {'N':>6}"
    print("\n" + "=" * len(header))
    print("Per-subject LOSO results")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for result in all_results:
        sid = result["subject_id"]
        m = result.get("metrics", {})
        if not m:
            print(f"{sid:>8}  {'(no data)':>44}")
            continue
        print(
            f"{sid:>8}  "
            f"{m.get('accuracy', float('nan')):>8.4f}  "
            f"{m.get('balanced_accuracy', float('nan')):>8.4f}  "
            f"{m.get('f1_macro', float('nan')):>8.4f}  "
            f"{m.get('roc_auc', float('nan')):>8.4f}  "
            f"{m.get('n_samples', 0):>6}"
        )


def print_aggregate_table(aggregate: Dict) -> None:
    """Print the aggregate statistics table."""
    metrics_to_print = [
        ("accuracy",          "Accuracy"),
        ("balanced_accuracy", "Balanced Accuracy"),
        ("precision",         "Precision (macro)"),
        ("recall",            "Recall (macro)"),
        ("f1_macro",          "F1 (macro)"),
        ("roc_auc",           "ROC-AUC"),
    ]
    print("\n" + "=" * 60)
    print("Aggregate LOSO results (mean ± std  [min, max])")
    print("=" * 60)
    for key, label in metrics_to_print:
        stats = aggregate.get(key, {})
        mean = stats.get("mean", float("nan"))
        std  = stats.get("std",  float("nan"))
        mn   = stats.get("min",  float("nan"))
        mx   = stats.get("max",  float("nan"))
        print(f"  {label:<22}  {mean:.4f} ± {std:.4f}  [{mn:.4f}, {mx:.4f}]")
    print()


# ---------------------------------------------------------------------------
# Confusion matrix plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: List[List[int]],
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
) -> None:
    """
    Save a normalised confusion matrix PNG.

    Parameters
    ----------
    cm : raw confusion matrix as a list-of-lists (from compute_metrics).
    class_names : e.g. ["NonTarget", "Target"]
    title : plot title
    save_path : if None the plot is shown interactively (or discarded).
    """
    if not _MPL:
        return

    cm_arr = np.array(cm, dtype=float)
    row_sums = cm_arr.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm_arr / row_sums

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(4, n), max(4, n)))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=range(n),
        yticks=range(n),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, f"{cm_norm[i, j]:.2f}",
                ha="center", va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
            )
    fig.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Confusion matrix saved to %s", save_path)

    plt.close(fig)


def save_all_confusion_matrices(
    all_results: List[Dict],
    class_names: List[str],
    output_dir: Path,
) -> None:
    """Save a confusion matrix PNG for every subject."""
    for result in all_results:
        sid = result["subject_id"]
        cm = result.get("metrics", {}).get("confusion_matrix")
        if cm is None:
            continue
        plot_confusion_matrix(
            cm,
            class_names=class_names,
            title=f"Subject {sid}",
            save_path=output_dir / f"confusion_matrix_subject_{sid}.png",
        )


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_results_csv(all_results: List[Dict], path: Path) -> None:
    """
    Write one CSV row per subject with all scalar metrics.

    Parameters
    ----------
    all_results : list of dicts from loso_runner.run_loso
    path : destination file path
    """
    scalar_keys = [
        "accuracy", "balanced_accuracy", "precision",
        "recall", "f1_macro", "roc_auc", "n_samples",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject_id"] + scalar_keys)
        writer.writeheader()
        for result in all_results:
            row = {"subject_id": result["subject_id"]}
            for k in scalar_keys:
                row[k] = result.get("metrics", {}).get(k, "")
            writer.writerow(row)
    logger.info("Results CSV written to %s", path)
