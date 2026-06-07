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
3. Load all configured datasets via MOABB, compute the channel intersection,
   and set cfg.n_channels automatically.
4. Run LOSO evaluation.
5. Print and save results.

Plugging in your own data
-------------------------
If you want to use a different data source, replace the `load_datasets`
section below with any code that returns a P300Dataset.  The LOSO runner
only needs a P300Dataset that implements `get_subject_ids()`,
`filter_by_subject()`, and `exclude_subject()`.
"""

import logging
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from config import Config
from data.dataset_interface import P300Dataset
from training.loso_runner import run_loso
from evaluation.analytics import run_full_analytics
from evaluation.metrics import aggregate_metrics
from utils.logging_utils import setup_logging


# ---------------------------------------------------------------------------
# Results-directory helper
# ---------------------------------------------------------------------------

def _next_results_dir(base: str = "results") -> str:
    """
    Return the next available numbered results directory name.
    Scans for existing results0, results1, ... and returns the first
    name whose directory does not yet exist.  If none exist, returns results0.
    """
    i = 0
    while Path(f"{base}{i}").exists():
        i += 1
    return f"{base}{i}"


# ---------------------------------------------------------------------------
# Configuration — edit this block to customise the experiment
# ---------------------------------------------------------------------------

cfg = Config(
    experiment_name="eegnet_p300_loso",
    seed=42,

    # Number of subjects to hold out, distributed evenly across datasets.
    # e.g. test_subjects=4 with 3 datasets -> 2 from one dataset, 1 from each other.
    # None = hold out all subjects (full LOSO).
    test_subjects=4,

    # n_channels and n_timepoints are set automatically after load_datasets():
    #   n_channels  = size of the channel intersection across all datasets
    #   n_timepoints = TARGET_SAMPLE_RATE (256) = 1 second at 256 Hz
    # No need to set them here.

    # Training
    n_epochs=20,
    batch_size=32,
    learning_rate=1e-3,

    # Use 100% of each dataset.  Set e.g. 0.1 for a quick 10% smoke-test.
    data_fraction=0.2 ,

    # Paths
    moabb_download_dir=os.path.expanduser("~/mne_data"),
    dataset_cache_dir="data/cache",
    output_dir=_next_results_dir(),  # auto-increments: results0, results1, …
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

# ---------------------------------------------------------------------------
# Experiment manifest
# ---------------------------------------------------------------------------

def save_experiment_manifest(
    cfg: Config,
    per_dataset_channels: dict,
    intersection_channels: List[str],
    n_classes: int,
    shared_labels: list,
) -> None:
    """
    Write a human-readable JSON manifest to results/<experiment>/manifest.json
    capturing dataset info, channel intersection, and class info — recorded
    before any training begins so the experiment is always reproducible.
    """
    import json
    from datetime import datetime

    manifest = {
        "experiment_name": cfg.experiment_name,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "seed": cfg.seed,
        "n_epochs": cfg.n_epochs,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "data_fraction": cfg.data_fraction,
        "n_timepoints": cfg.n_timepoints,  # auto-set: TARGET_SAMPLE_RATE = 1 s at 256 Hz
        "test_subjects_requested": cfg.test_subjects,  # int count or None (full LOSO)
        "datasets": {
            did: {
                "n_channels": len(channels),
                "channels": channels,
                "channels_dropped_from_intersection": [
                    c for c in channels if c not in set(intersection_channels)
                ],
            }
            for did, channels in per_dataset_channels.items()
        },
        "channel_intersection": {
            "n_channels": len(intersection_channels),
            "channels": intersection_channels,
        },
        "classes": {
            "n_classes": n_classes,
            "labels": shared_labels,
            "class_names": CLASS_NAMES,
        },
    }

    out_path = cfg.logs_dir() / "manifest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Experiment manifest saved → {out_path}\n")
    logging.getLogger(__name__).info("Manifest saved to %s", out_path)

# ---------------------------------------------------------------------------
# Channel-intersection utility
# ---------------------------------------------------------------------------

def _compute_channel_intersection(
    per_dataset_channels: dict,
) -> List[str]:
    """
    Given {dataset_id: [channel_names]}, return the ordered list of channels
    present in every dataset (intersection), preserving canonical ordering.

    Prints a detailed report to stdout so you always know which channels
    are being used.
    """
    from framework.canonical_channels import CANONICAL_CHANNELS

    # Start with the full canonical list and narrow down
    intersection = list(CANONICAL_CHANNELS)
    for did, channels in per_dataset_channels.items():
        ch_set = set(channels)
        intersection = [c for c in intersection if c in ch_set]

    print("\n" + "=" * 60)
    print("CHANNEL INTERSECTION REPORT")
    print("=" * 60)
    for did, channels in per_dataset_channels.items():
        ch_set = set(channels)
        present = [c for c in intersection if c in ch_set]
        missing = [c for c in intersection if c not in ch_set]  # always empty after intersection
        dropped = [c for c in channels if c not in set(intersection)]
        print(f"\n  {did}:")
        print(f"    Raw channels : {len(channels)}")
        print(f"    In intersection : {len(present)}")
        if dropped:
            print(f"    Dropped (not in all datasets): {dropped}")

    print(f"\n  Joint intersection : {len(intersection)} channels")
    print(f"  Channels           : {intersection}")
    print("=" * 60 + "\n")

    if not intersection:
        raise ValueError(
            "No channels remain after intersecting all datasets. "
            "Check that channel names are normalised correctly in "
            "canonical_channels.py."
        )

    return intersection


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_datasets(cfg: Config) -> P300Dataset:
    """
    Load one or more P300 datasets via MOABB, compute the channel intersection
    across all of them, apply the data-fraction subsampling, and return a
    single P300Dataset.

    cfg.n_channels is set in-place to the number of intersected channels so
    the EEGNet model is built with the correct input size.

    To add more datasets, extend the two lists below.
    """
    from moabb.datasets import BNCI2014_009, BI2014a, BI2013a, Lee2019_ERP, Mainsah2025_A
    from moabb.utils import set_download_dir
    from framework.dataset_registry import MoabbDatasetLoader
    from framework.label_schema import LabelSchema, infer_shared_label_schema
    from framework.eeg_dataset import EEGSampleDataset, EEGSample
    from framework.preprocessing import TARGET_SAMPLE_RATE

    # ── 1. Declare which datasets to load ────────────────────────────────
    datasets = [
        Mainsah2025_A(),
        BNCI2014_009(),
        BI2014a(),
        BI2013a(),
    ]
    dataset_ids = [
        "Mainsah2025_A",
        "BNCI2014_009",
        "BI2014a",
        "BI2013a",
    ]
    # n_classes is inferred automatically from the shared label schema.

    set_download_dir(cfg.moabb_download_dir)

    shared_labels, schemas = infer_shared_label_schema(dataset_ids)
    n_classes = len(shared_labels)
    print(f"\nShared label space ({n_classes} classes): {shared_labels}")

    # ── 2. Load each dataset ──────────────────────────────────────────────
    per_dataset: dict[str, EEGSampleDataset] = {}
    for moabb_ds, did in zip(datasets, dataset_ids):
        print(f"\nLoading {did}…")
        loader = MoabbDatasetLoader(
            moabb_dataset=moabb_ds,
            dataset_id=did,
            label_schema=schemas[did],
            n_classes=n_classes,
        )
        ds = loader.load_all_subjects()
        per_dataset[did] = ds
        print(f"  → {len(ds)} samples, {len(ds.get_subject_ids())} subjects")

    # ── 3. Compute channel intersection and report ────────────────────────
    per_dataset_channels = {
        did: ds.get_channel_names() for did, ds in per_dataset.items()
    }
    intersection_channels = _compute_channel_intersection(per_dataset_channels)

    # Update cfg so EEGNet is built with the right number of channels and timepoints
    cfg.n_channels = len(intersection_channels)
    cfg.n_timepoints = TARGET_SAMPLE_RATE  # 1 s x 256 Hz = 256 samples
    print(f"cfg.n_channels automatically set to {cfg.n_channels}")
    print(f"cfg.n_timepoints automatically set to {cfg.n_timepoints} (1 s at {TARGET_SAMPLE_RATE} Hz)\n")

    # ── 4. Trim each dataset to intersection channels ─────────────────────
    def _trim_to_intersection(
        ds: EEGSampleDataset,
        intersection: List[str],
    ) -> EEGSampleDataset:
        """Keep only the intersection channels from every sample."""
        if not ds.samples:
            return ds
        existing = ds.get_channel_names()
        ch_set = {ch: i for i, ch in enumerate(existing)}
        indices = [ch_set[ch] for ch in intersection if ch in ch_set]
        trimmed = []
        for s in ds.samples:
            trimmed.append(EEGSample(
                eeg=s.eeg[indices, :],
                channel_names=intersection,
                label=s.label,
                subject_id=s.subject_id,
                session_id=s.session_id,
                dataset_id=s.dataset_id,
            ))
        return EEGSampleDataset(trimmed)

    trimmed_datasets = {
        did: _trim_to_intersection(ds, intersection_channels)
        for did, ds in per_dataset.items()
    }

    # ── 5. Apply data_fraction (stratified by class) ─────────────────────
    if cfg.data_fraction < 1.0:
        rng = random.Random(cfg.seed)
        for did in list(trimmed_datasets.keys()):
            ds = trimmed_datasets[did]
            by_label: dict = defaultdict(list)
            for i, s in enumerate(ds.samples):
                by_label[s.label].append(i)

            keep = []
            for label_indices in by_label.values():
                rng.shuffle(label_indices)
                n_keep = max(1, round(len(label_indices) * cfg.data_fraction))
                keep.extend(label_indices[:n_keep])

            trimmed_datasets[did] = EEGSampleDataset(
                [ds.samples[i] for i in sorted(keep)]
            )
            print(
                f"data_fraction={cfg.data_fraction:.2f}: {did} "
                f"{len(ds)} → {len(trimmed_datasets[did])} samples"
            )

    # ── 6. Concatenate all datasets and wrap in P300Dataset ───────────────
    combined = EEGSampleDataset.concat(list(trimmed_datasets.values()))
    print(
        f"\nCombined dataset: {len(combined)} samples, "
        f"{len(combined.get_subject_ids())} subjects\n"
    )

    # Build {dataset_id: [subject_ids]} for use by the LOSO subject selector
    per_dataset_subject_ids = {
        did: ds.get_subject_ids() for did, ds in trimmed_datasets.items()
    }

    return (
        P300Dataset(combined),
        per_dataset_channels,
        intersection_channels,
        n_classes,
        shared_labels,
        per_dataset_subject_ids,
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

    # Data — also sets cfg.n_channels via the intersection logic
    logger.info("Loading datasets…")
    dataset, per_dataset_channels, intersection_channels, n_classes, shared_labels, per_dataset_subject_ids = load_datasets(cfg)
    logger.info(
        "Dataset loaded: %d samples, n_channels=%d, n_timepoints=%d",
        len(dataset), cfg.n_channels, cfg.n_timepoints,
    )

    # Save manifest before any training begins
    save_experiment_manifest(cfg, per_dataset_channels, intersection_channels, n_classes, shared_labels)

    # LOSO
    all_results = run_loso(dataset, cfg, per_dataset_subject_ids)

    # Analytics — generates all plots, CSV, and console tables in one call
    aggregate = aggregate_metrics([r["metrics"] for r in all_results])
    run_full_analytics(all_results, aggregate, CLASS_NAMES, cfg.logs_dir())

    logger.info("Experiment complete.  Results in: %s", cfg.results_dir())


if __name__ == "__main__":
    main()