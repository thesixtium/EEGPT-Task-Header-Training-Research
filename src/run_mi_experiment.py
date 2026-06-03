"""
run_mi_experiment.py
--------------------
Trains the EEGPT model using the leave-subjects-out protocol with
progressive session adaptation across all configured datasets.

Training strategy:
  1. Train ONE shared base model on all subjects EXCEPT those in lso_subjects.
  2. For each held-out subject:
       a. Load the shared base checkpoint → zero-shot evaluation
       b. Progressive session adaptation (session 0 → session 1 → ...)
       c. Save per-subject results

Edit the ExperimentConfig block below to change datasets, epochs,
learning rate, or output paths.
"""

import logging
import os
from pathlib import Path

from moabb.utils import set_download_dir
from moabb.datasets import (
    BNCI2014_009,
    BI2014a
)

from framework.experiment_runner import ExperimentConfig, run_experiment
from framework.label_schema import DATASET_LABEL_MAPS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
)

# ---------------------------------------------------------------------------
# Pin MOABB's raw-data download cache to your existing mne_data folder.
# This prevents re-downloading on every run.
# ---------------------------------------------------------------------------
MOABB_DOWNLOAD_DIR = r'C:\Users\ajrbe\mne_data'
set_download_dir(MOABB_DOWNLOAD_DIR)
os.makedirs(MOABB_DOWNLOAD_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Configure your experiment here
# ---------------------------------------------------------------------------
cfg = ExperimentConfig(
    experiment_name='mi_lso',
    base_model_path=r'C:\Users\ajrbe\Documents\Git\EEGPT-Task-Header-Training-Research\lib\checkpoints\eegpt_mcae_58chs_4s_large4E.ckpt',

    # Add or remove MOABB dataset objects and their matching registry IDs.
    # The ID must exist as a key in label_schema.py → DATASET_LABEL_MAPS.
    datasets=[
        BNCI2014_009(),
        BI2014a()
    ],
    dataset_ids=[
        'BNCI2014_009', 'BI2014a'
    ],

    max_lr=4e-4,
    base_epochs=2,
    adapt_epochs=2,
    data_fraction = 0.05,
    batch_size=32,

    lso_enabled=True,
    # None = hold out ALL subjects one at a time.
    # Set to e.g. [1, 2, 3] to hold out only those original subject IDs
    # (matched across every dataset that has them).
    lso_subjects=[1],
    test_session_idx=0,     # index of the held-out test session per subject

    # Separate folder for preprocessed .pt cache files.
    # Do NOT point this at mne_data — that's for raw MOABB downloads.
    # This folder will be created automatically if it doesn't exist.
    dataset_cache_dir=r'C:\Users\ajrbe\Documents\Git\EEGPT-Task-Header-Training-Research\data\dataset_cache',

    output_dir='results',
    seed=7_11_2002,
    force_retrain_base=True,   # dataset/channel set changed — retrain the base model
)

# ---------------------------------------------------------------------------
if __name__ == '__main__':
    run_experiment(cfg)