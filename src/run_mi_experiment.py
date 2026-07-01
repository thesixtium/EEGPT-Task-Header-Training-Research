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
from moabb.datasets import BNCI2014_001

from framework.experiment_runner import ExperimentConfig, run_experiment
from framework.label_schema import DATASET_LABEL_MAPS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(name)s  %(message)s',
)

# ---------------------------------------------------------------------------
# Where MOABB downloads raw EEG data to.
#
# IMPORTANT: this is a plain, persistent directory under the project root —
# NOT a SLURM scratch path (e.g. /scratch/$SLURM_JOB_ID). Scratch dirs are
# unique per job and get cleaned up when the job ends, which previously
# caused failures: MNE persists its own on-disk config file
# (~/.mne-python/mne-python.json) independently of whatever the shell's
# MNE_DATA env var says, so a later job could end up resolving a dead path
# left over from an earlier job.
#
# Using one fixed directory sidesteps that whole class of bug — the path
# always exists, and re-running simply reuses whatever MOABB already
# downloaded there instead of re-fetching from the network every time.
# ---------------------------------------------------------------------------
_DATA_DIR = Path(os.environ.get('MOABB_DOWNLOAD_DIR', Path('data') / 'mne_data')).resolve()
_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Set the path in every place MNE/MOABB might look for it, so there is no
# ambiguity: the environment variable, the process's own copy of it, and
# MNE's persisted JSON config (which is what previously went stale).
os.environ['MNE_DATA'] = str(_DATA_DIR)
os.environ['MOABB_DOWNLOAD_DIR'] = str(_DATA_DIR)
mne.utils.set_config('MNE_DATA', str(_DATA_DIR), set_env=True)
set_download_dir(str(_DATA_DIR))

logging.getLogger(__name__).info("MNE_DATA / MOABB download dir: %s", _DATA_DIR)

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

    datasets=[BNCI2014_001()],
    dataset_ids=['BNCI2014_001'],

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

    output_dir='results',
    seed=7_11_2002,
    force_retrain_base=True,   # dataset/channel set changed — retrain the base model
)

# ---------------------------------------------------------------------------
if __name__ == '__main__':
    run_experiment(cfg)