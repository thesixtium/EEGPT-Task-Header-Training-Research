"""
run_mi_experiment.py
--------------------
Trains the EEGPT model using the leave-subjects-out protocol with
progressive session adaptation across all configured datasets.

Training strategy:
  1. Randomly select lso_subjects_per_test_set subjects from each dataset
     to hold out (one subject per "test set slot", seeded for reproducibility).
  2. Train ONE shared base model on all remaining subjects simultaneously.
  3. For each held-out subject:
       a. Load the shared base checkpoint → zero-shot evaluation
       b. Progressive session adaptation (session 0 session 1 ...)
       c. Save per-subject results

All paths are relative to the directory this script is run from.

Edit the ExperimentConfig block below to change datasets, epochs,
learning rate, or output paths.
"""

import logging
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Silence MNE and MOABB before any other imports so their internal loggers
# are never registered at INFO level.  The messages suppressed here are:
#   "Adding metadata with N columns"  — MNE bookkeeping during epoch loading
#   "X matching events found"         — MNE internal event-detection report
# Both indicate normal operation; they are not errors or warnings.
# ---------------------------------------------------------------------------
import mne
mne.set_log_level("WARNING")
logging.getLogger("moabb").setLevel(logging.WARNING)

from moabb.utils import set_download_dir
from moabb.datasets import (
    BNCI2014_009, BI2014a, BI2013a, BI2014b, BI2015a, BI2015b, BNCI2014_008,
    Chailloux2020, Cattan2019_VR, EPFLP300, ErpCore2021_ERN, ErpCore2021_LRP,
    ErpCore2021_MMN, ErpCore2021_N170, ErpCore2021_N2pc, ErpCore2021_N400,
    ErpCore2021_P3, GuttmannFlury2025_P300, Lee2019_ERP, Lee2024_AC,
    Lee2024_BS, Lee2024_DL, Lee2024_EL, Lee2024_TV, Mainsah2025_A,
    Mainsah2025_B, Mainsah2025_C, Mainsah2025_D, Mainsah2025_E, Mainsah2025_F,
    Mainsah2025_G, Mainsah2025_H, Mainsah2025_I, Mainsah2025_J, Mainsah2025_K,
    Mainsah2025_L, Mainsah2025_M, Mainsah2025_P, Mainsah2025_N, Mainsah2025_Q,
    Mainsah2025_S1, Mainsah2025_R, Mainsah2025_S2, Speier2017, Zheng2020,
    Mainsah2025_O, Zhang2025
)

from framework.experiment_runner import ExperimentConfig, run_experiment
from framework.label_schema import DATASET_LABEL_MAPS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
)

# ---------------------------------------------------------------------------
# Pin MOABB's raw-data download cache relative to the working directory.
# Override MOABB_DOWNLOAD_DIR below if you want it somewhere else on disk.
# ---------------------------------------------------------------------------
MOABB_DOWNLOAD_DIR = os.environ.get('MOABB_DOWNLOAD_DIR', str(Path('data') / 'mne_data'))
set_download_dir(MOABB_DOWNLOAD_DIR)
os.makedirs(MOABB_DOWNLOAD_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Verify the pretrained EEGPT backbone is present before doing anything else.
# ---------------------------------------------------------------------------
_CKPT_PATH = Path('src') / 'lib' / 'checkpoints' / 'eegpt_mcae_58chs_4s_large4E.ckpt'

if not _CKPT_PATH.exists():
    raise FileNotFoundError(
        f'EEGPT checkpoint not found: {_CKPT_PATH}\n'
        'Download it from https://figshare.com/ndownloader/files/46452745?private_link=e37df4f8a907a866df4b\n'
        f'and place it at {_CKPT_PATH.resolve()}'
    )

# ---------------------------------------------------------------------------
# Configure your experiment here
# ---------------------------------------------------------------------------
cfg = ExperimentConfig(
    experiment_name='mi_lso',

    # Path to the pretrained EEGPT backbone, relative to wherever you invoke
    # this script from (i.e. the project root).
    base_model_path=str(
        Path('src') / 'lib' / 'checkpoints' / 'eegpt_mcae_58chs_4s_large4E.ckpt'
    ),

    # Add or remove MOABB dataset objects and their matching registry IDs.
    # The ID must exist as a key in label_schema.py → DATASET_LABEL_MAPS.
    datasets=[
        BNCI2014_009(), BI2014a(), BI2013a(), BI2014b(), BI2015a(), BI2015b(), BNCI2014_008(),
        #Chailloux2020(),
        Cattan2019_VR(), EPFLP300(), ErpCore2021_ERN(), ErpCore2021_LRP(),
        ErpCore2021_MMN(), ErpCore2021_N170(), ErpCore2021_N2pc(), ErpCore2021_N400(),
    #    ErpCore2021_P3(), GuttmannFlury2025_P300(), Lee2019_ERP(), Lee2024_AC(),
    #    Lee2024_BS(), Lee2024_DL(), Lee2024_EL(), Lee2024_TV(), Mainsah2025_A(),
    #    Mainsah2025_B(), Mainsah2025_C(), Mainsah2025_D(), Mainsah2025_E(), Mainsah2025_F(),
    #    Mainsah2025_G(), Mainsah2025_H(), Mainsah2025_I(), Mainsah2025_J(), Mainsah2025_K(),
    #    Mainsah2025_L(), Mainsah2025_M(), Mainsah2025_P(), Mainsah2025_N(), Mainsah2025_Q(),
    #    Mainsah2025_S1(), Mainsah2025_R(), Mainsah2025_S2(), Speier2017(), Zheng2020(),
    #    Mainsah2025_O(), Zhang2025()
    ],
    dataset_ids=[
        'BNCI2014_009', 'BI2014a', 'BI2013a', 'BI2014b', 'BI2015a', 'BI2015b', 'BNCI2014_008',
        #'Chailloux2020',
        'Cattan2019_VR', 'EPFLP300', 'ErpCore2021_ERN', 'ErpCore2021_LRP',
        'ErpCore2021_MMN', 'ErpCore2021_N170', 'ErpCore2021_N2pc', 'ErpCore2021_N400',
    #    'ErpCore2021_P3', 'GuttmannFlury2025_P300', 'Lee2019_ERP', 'Lee2024_AC',
    #    'Lee2024_BS', 'Lee2024_DL', 'Lee2024_EL', 'Lee2024_TV', 'Mainsah2025_A',
    #    'Mainsah2025_B', 'Mainsah2025_C', 'Mainsah2025_D', 'Mainsah2025_E', 'Mainsah2025_F',
    #    'Mainsah2025_G', 'Mainsah2025_H', 'Mainsah2025_I', 'Mainsah2025_J', 'Mainsah2025_K',
    #    'Mainsah2025_L', 'Mainsah2025_M', 'Mainsah2025_P', 'Mainsah2025_N', 'Mainsah2025_Q',
    #    'Mainsah2025_S1', 'Mainsah2025_R', 'Mainsah2025_S2', 'Speier2017', 'Zheng2020',
    #    'Mainsah2025_O', 'Zhang2025'
    ],

    max_lr=4e-4,
    base_epochs=100,
    adapt_epochs=10,
    data_fraction=1.0,
    batch_size=32,

    lso_enabled=True,
    # Number of subjects to randomly hold out per dataset (one "test set slot"
    # per dataset).  For example, lso_subjects_per_test_set=1 holds out 1
    # subject from BNCI2014_009 AND 1 subject from BI2014a → 2 held-out
    # subjects total.  The selection is seeded by `seed` for reproducibility.
    lso_subjects_per_test_set=1,
    test_session_idx=0,     # index of the held-out test session per subject

    # Separate folder for preprocessed .pt cache files.
    # Do NOT point this at mne_data — that's for raw MOABB downloads.
    # This folder will be created automatically if it doesn't exist.
    dataset_cache_dir=str(Path('data') / 'dataset_cache'),

    output_dir='results',
    seed=7_11_2002,
    force_retrain_base=True,   # dataset/channel set changed — retrain the base model
)

# ---------------------------------------------------------------------------
if __name__ == '__main__':
    run_experiment(cfg)
