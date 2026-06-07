"""
config.py
---------
All experiment hyperparameters in one place.

Edit this file to change datasets, training settings, or output paths.
Nothing else in the codebase contains magic numbers.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Any


@dataclass
class Config:
    # ── Experiment identity ───────────────────────────────────────────────
    experiment_name: str = "eegnet_p300_loso"
    seed: int = 42

    # ── LOSO evaluation ───────────────────────────────────────────────────
    # Which subjects to hold out, one at a time.
    # None means hold out every subject (full LOSO over all subjects).
    test_subjects: Optional[List[int]] = None

    # ── Model ─────────────────────────────────────────────────────────────
    # EEGNet expects input shaped [batch, 1, n_channels, n_timepoints].
    # n_channels is set automatically after load_datasets() resolves the
    # channel intersection across all datasets — you do not need to set it
    # manually.
    n_channels: int = 8      # overwritten by load_datasets()
    n_timepoints: int = 120  # time samples per trial

    # ── Training ──────────────────────────────────────────────────────────
    n_epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3

    # ── Data ──────────────────────────────────────────────────────────────
    # Path where MOABB downloads raw data.
    moabb_download_dir: str = "~/mne_data"
    # Path for preprocessed .pt cache files (speeds up repeat runs).
    dataset_cache_dir: str = "data/cache"

    # Fraction of each dataset to use, in (0.0, 1.0].
    # 1.0 = use all data; 0.1 = use 10% of each dataset (stratified by class).
    data_fraction: float = 1.0

    # ── Output ────────────────────────────────────────────────────────────
    output_dir: str = "results"
    save_checkpoints: bool = True

    # ── Reproducibility ───────────────────────────────────────────────────
    num_workers: int = 0   # DataLoader workers; 0 = main process (safest)

    def results_dir(self) -> Path:
        return Path(self.output_dir) / self.experiment_name

    def checkpoints_dir(self) -> Path:
        return self.results_dir() / "checkpoints"

    def logs_dir(self) -> Path:
        return self.results_dir() / "logs"