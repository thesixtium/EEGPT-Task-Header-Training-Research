"""
train.py
--------
Main entry point for the EEGNet P300 LOSO experiment.

Edit config.py to change hyperparameters or dataset settings.
Then run:

    python train.py

What this script does
---------------------
1. Load configuration.
2. Set random seeds for reproducibility.
3. Load the dataset using the existing MOABB pipeline.
4. Wrap it in the P300Dataset adapter.
5. Run LOSO evaluation.
6. Print and save results.

Plugging in your own data
-------------------------
If you want to use a different data source, replace the `load_dataset`
section below with any code that returns a P300Dataset.  The LOSO runner
only needs a P300Dataset that implements `get_subject_ids()`,
`filter_by_subject()`, and `exclude_subject()`.
"""

import logging
import os
import random
from pathlib import Path

import numpy as np
import torch

from config import Config
from data.dataset_interface import P300Dataset
from training.loso_runner import run_loso
from evaluation.analytics import (
    print_loso_table,
    print_aggregate_table,
    save_results_csv,
    save_all_confusion_matrices,
)
from evaluation.metrics import aggregate_metrics
from utils.logging_utils import setup_logging


# ---------------------------------------------------------------------------
# Configuration — edit this block to customise the experiment
# ---------------------------------------------------------------------------

cfg = Config(
    experiment_name="eegnet_p300_loso",
    seed=42,

    # Hold out these subjects; None = hold out all subjects (full LOSO)
    test_subjects=[0],

    # Must match the shape of your preprocessed EEG trials
    n_channels=8,
    n_timepoints=120,

    # Training
    n_epochs=2,
    batch_size=32,
    learning_rate=1e-3,

    # Paths
    moabb_download_dir=os.path.expanduser("~/mne_data"),
    dataset_cache_dir="data/cache",
    output_dir="results",
    save_checkpoints=True,
)

CLASS_NAMES = ["NonTarget", "Target"]  # index 0 and 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(cfg: Config) -> P300Dataset:
    """
    Load the P300 dataset using the existing MOABB pipeline.

    Returns a P300Dataset wrapping the full EEGSampleDataset.

    ── To use MOABB ────────────────────────────────────────────────────────
    Uncomment and adapt the block below.  You need the MOABB and the
    framework (eeg_dataset, dataset_registry, label_schema, preprocessing)
    on the Python path.

    from moabb.datasets import BNCI2014_009
    from moabb.utils import set_download_dir
    from dataset_registry import MoabbDatasetLoader
    from label_schema import LabelSchema
    from eeg_dataset import EEGSampleDataset

    set_download_dir(cfg.moabb_download_dir)

    schema = LabelSchema("BNCI2014_009")
    loader = MoabbDatasetLoader(
        moabb_dataset=BNCI2014_009(),
        dataset_id="BNCI2014_009",
        label_schema=schema,
        n_classes=2,
    )
    raw_dataset: EEGSampleDataset = loader.load_all_subjects()
    return P300Dataset(raw_dataset)

    ── Synthetic fallback (for quick testing without MOABB) ────────────────
    """
    logging.warning(
        "load_dataset: using synthetic data.  "
        "Replace this function body with real MOABB loading for actual experiments."
    )
    from data.dataset_interface import _make_synthetic_dataset
    return _make_synthetic_dataset(
        n_subjects=10,
        n_trials_per_subject=200,
        n_channels=cfg.n_channels,
        n_timepoints=cfg.n_timepoints,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Logging
    setup_logging(cfg.logs_dir() / "experiment.log")
    logger = logging.getLogger(__name__)
    logger.info("Starting experiment: %s", cfg.experiment_name)
    logger.info("Config: %s", cfg)

    # Reproducibility
    set_seed(cfg.seed)

    # Data
    logger.info("Loading dataset…")
    dataset = load_dataset(cfg)
    logger.info("Dataset loaded: %d samples", len(dataset))

    # LOSO
    all_results = run_loso(dataset, cfg)

    # Reporting
    print_loso_table(all_results)
    aggregate = aggregate_metrics([r["metrics"] for r in all_results])
    print_aggregate_table(aggregate)

    # Save CSV
    csv_path = cfg.logs_dir() / "loso_results.csv"
    save_results_csv(all_results, csv_path)

    # Save confusion matrices
    cm_dir = cfg.logs_dir() / "confusion_matrices"
    save_all_confusion_matrices(all_results, CLASS_NAMES, cm_dir)

    logger.info("Experiment complete.  Results in: %s", cfg.results_dir())


if __name__ == "__main__":
    main()
