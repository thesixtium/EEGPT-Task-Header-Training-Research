#!/bin/bash
#SBATCH --job-name=eegpt_mi_lso
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=aleksander.berezowsk@ucalgary.ca

echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Started  : $(date)"

cd /home/aleksander.berezowsk/EEGPT-Task-Header-Training-Research
echo "Directory: $(pwd)"

module load cuda/11.3.0
source ~/venv/bin/activate

# setuptools must come first — pkg_resources depends on it
pip install "setuptools==69.5.1"

# Install everything except torch
grep -v "^torch\|^torchvision\|^torchaudio" requirements.txt > ~/requirements_notorch.txt
pip install --no-deps -r ~/requirements_notorch.txt
pip install pyhealth==1.1.6 --no-deps
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 torchmetrics

export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export MKL_NUM_THREADS=8
export PYTHONUNBUFFERED=1

# Persistent, job-independent data directory. NOT /scratch/$SLURM_JOB_ID —
# that path disappears the moment the job ends, which is what caused the
# "stale MNE_DATA" failures. This directory always exists, and reusing it
# across runs means MOABB doesn't have to re-download the dataset every
# single job.
export MOABB_DOWNLOAD_DIR="$(pwd)/data/mne_data"
export MNE_DATA="$MOABB_DOWNLOAD_DIR"
mkdir -p "$MNE_DATA"
echo "MNE_DATA = $MNE_DATA"

CKPT="src/lib/checkpoints/eegpt_mcae_58chs_4s_large4E.ckpt"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found at $CKPT - aborting."
    exit 1
fi

export PYTHONPATH=$(pwd):$(pwd)/src:$PYTHONPATH
python3 src/run_mi_experiment.py

echo "Finished : $(date)"
echo "Exit code: $?"
