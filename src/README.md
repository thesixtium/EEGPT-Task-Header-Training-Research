# Running the EEG Experiment on ARC — Complete Guide

Do this **tonight**, in this exact order.

---

## What you need before starting

- University of Calgary VPN connected (if off-campus)
- Your git repo URL
- The EEGPT checkpoint file downloaded to your laptop:
  https://figshare.com/ndownloader/files/46452745?private_link=e37df4f8a907a866df4b
  Save it as: `eegpt_mcae_58chs_4s_large4E.ckpt`

---

## Step 1 — SSH into ARC

```bash
ssh YOURUSERNAME@arc.ucalgary.ca
```

You are now on the **login node**. Don't run experiments here.

---

## Step 2 — Get a /work directory

Email support@hpc.ucalgary.ca and ask for a /work allocation.
While waiting, you can use /home (500 GB limit). Steps below use /work.
If you don't have /work yet, replace every `/work/YOURUSERNAME` below with `~/`.

---

## Step 3 — Clone your repo into /work

```bash
cd /work/YOURUSERNAME
git clone https://github.com/YOURREPO/eegpt.git
cd eegpt
```

Your project is now at `/work/YOURUSERNAME/eegpt/`.
**All remaining steps run from inside this folder unless told otherwise.**

---

## Step 4 — Upload the EEGPT checkpoint

From your **laptop** (open a new terminal, don't close the ARC one):

```bash
scp /path/to/eegpt_mcae_58chs_4s_large4E.ckpt \
    YOURUSERNAME@arc.ucalgary.ca:/work/YOURUSERNAME/eegpt/lib/checkpoints/
```

If the folder doesn't exist yet, create it first on ARC:
```bash
mkdir -p /work/YOURUSERNAME/eegpt/lib/checkpoints
```

---

## Step 5 — Place the provided files into the repo

Put these three files from this bundle into the right places:

| File | Where it goes in your repo |
|------|---------------------------|
| `experiment_runner.py` | `framework/experiment_runner.py` (replaces existing) |
| `run_mi_experiment.slurm` | project root (`/work/YOURUSERNAME/eegpt/`) |
| `requirements.txt` | project root (`/work/YOURUSERNAME/eegpt/`) |

Upload them from your laptop:
```bash
scp experiment_runner.py \
    YOURUSERNAME@arc.ucalgary.ca:/work/YOURUSERNAME/eegpt/framework/

scp run_mi_experiment.slurm requirements.txt \
    YOURUSERNAME@arc.ucalgary.ca:/work/YOURUSERNAME/eegpt/
```

---

## Step 6 — Edit the SLURM file (one line)

On ARC:
```bash
nano run_mi_experiment.slurm
```

Find this line near the top:
```
#SBATCH --mail-user=YOUREMAIL@ucalgary.ca
```

Change it to your actual email. Save with `Ctrl+O`, exit with `Ctrl+X`.

---

## Step 7 — Request an interactive compute node to set up Python

**Never install packages on the login node.** Do this instead:

```bash
salloc --mem=8G -c 4 -N 1 -n 1 -t 02:00:00 -p cpu2019
```

Wait a moment. When you see your prompt change to show a compute node name
(e.g. `[you@fc12 eegpt]$`), you're on the compute node. Continue.

---

## Step 8 — Create the virtual environment

```bash
# Load Python (check available versions with: module avail python)
module load python/3.12.5

# Create the venv in your home directory
python -m venv ~/venv

# Activate it
source ~/venv/bin/activate

# Your prompt should now show (venv) at the start
```

---

## Step 9 — Install PyTorch (must be done separately first)

```bash
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

This downloads ~2 GB. Takes a few minutes. Wait for it to finish.

---

## Step 10 — Install everything else

```bash
pip install -r requirements.txt
```

---

## Step 11 — Verify GPU will work

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

This will print `CUDA available: False` because you're on a CPU node right now.
That's fine — it will be True when your actual job runs on a GPU node.

---

## Step 12 — Exit the interactive session

```bash
exit
```

You're back on the login node.

---

## Step 13 — Edit run_mi_experiment.py for the full run

```bash
nano run_mi_experiment.py
```

Make sure these lines in the `cfg = ExperimentConfig(...)` block are set:
```python
data_fraction=1.0,       # was 0.1
base_epochs=20,          # or your desired number
adapt_epochs=10,
force_retrain_base=False, # False = reuse checkpoint if it exists (good for resuming)
```

And **uncomment all the datasets** in the `datasets=[...]` and `dataset_ids=[...]` lists.

Save and exit (`Ctrl+O`, `Ctrl+X`).

---

## Step 14 — Make sure you're in the project directory

```bash
cd /work/YOURUSERNAME/eegpt
iconv -f UTF-16 -t UTF-8 requirements.txt -o requirements.txt
sed -i 's/\r//' requirements.txt
```

---

## Step 15 — Submit the first job

```bash
sbatch run_mi_experiment.slurm
```

You'll see something like:
```
Submitted batch job 1234567
```

Write down that number.

---

## Step 16 — Chain jobs (so the experiment continues past 24 hours)

The GPU partition has a 24-hour limit. Chain 3–4 jobs so they run back-to-back
automatically. Each one picks up exactly where the last one stopped.

```bash
# Do this right after Step 15, using the job ID from that step
JOB1=1234567   # replace with your actual job ID from Step 15

JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 run_mi_experiment.slurm)
echo "Job 2: $JOB2"

JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 run_mi_experiment.slurm)
echo "Job 3: $JOB3"

JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 run_mi_experiment.slurm)
echo "Job 4: $JOB4"

for i in 1 2 3 4 5 6 7 8 9; do
  J=$(sbatch --parsable run_job.slurm)
  J=$(sbatch --parsable --dependency=afterok:$J run_job.slurm)
  J=$(sbatch --parsable --dependency=afterok:$J run_job.slurm)
  J=$(sbatch --parsable --dependency=afterok:$J run_job.slurm)
  J=$(sbatch --parsable --dependency=afterok:$J run_job.slurm)
  J=$(sbatch --parsable --dependency=afterok:$J run_job.slurm)
done
```

`afterok` means Job 2 only starts if Job 1 finishes successfully (exit code 0).
If Job 1 crashes, the chain stops. That's a good thing — you don't want to
submit a chain if something is broken.

The resume logic in `experiment_runner.py` means each new job automatically
skips subjects that already have a completed result file.

---

## How to watch the experiment while it runs

```bash
# See all your jobs in the queue
squeue -u $USER

# Watch the live log output (Ctrl+C to stop watching, job keeps running)
tail -f eegpt_mi_lso_1234567.out

# Watch the status file your code writes (updates every epoch)
watch -n 10 cat /work/YOURUSERNAME/eegpt/results/mi_lso/status.txt

# Check GPU usage (live, attaches to the running node)
srun --jobid 1234567 --pty watch -n 10 nvidia-smi
```

The log file is named `eegpt_mi_lso_JOBID.out` — it's in the same folder
where you ran `sbatch`.

---

## Where are all the outputs?

Everything important is in `/work/YOURUSERNAME/eegpt/results/mi_lso/`:

```
results/mi_lso/
├── status.txt                  ← live progress, updated every epoch
├── subject_results/            ← one JSON per subject (saved as each finishes)
│   ├── BNCI2014_009_s1.json
│   ├── BNCI2014_009_s2.json
│   └── ...
├── summary.csv                 ← final results table
├── checkpoints/
│   └── base_model.ckpt         ← trained base model (reused on resume)
└── logs/                       ← per-run training curves (CSV + plots)
```

The dataset cache (preprocessed tensors) is at:
```
data/dataset_cache/             ← reused between jobs, saves hours of preprocessing
```

---

## What survives between chained jobs?

| Item | Location | Survives? |
|------|----------|-----------|
| Code & results | `/work/` | ✅ Yes, forever |
| Dataset cache (.pt files) | `data/dataset_cache/` in `/work/` | ✅ Yes |
| Base model checkpoint | `results/mi_lso/checkpoints/base_model.ckpt` | ✅ Yes |
| Per-subject JSONs | `results/mi_lso/subject_results/` | ✅ Yes |
| Raw MOABB downloads | `/scratch/$JOBID/` | ❌ Deleted 5 days after each job — but the .pt cache means you never need to re-download |
| Log files (.out/.err) | project root | ✅ Yes |

---

## If something goes wrong

**Check why a job failed:**
```bash
cat eegpt_mi_lso_1234567.err
```

**Cancel a job:**
```bash
scancel 1234567
```

**Cancel the whole chain:**
```bash
scancel 1234567 1234568 1234569 1234570
```

**See all your current and pending jobs:**
```bash
squeue -u $USER
```

**If you run out of memory (job killed with no output):** edit the SLURM file
and change `--mem=64G` to `--mem=128G`, then resubmit.

**If the job sits pending for a long time:** the gpu-v100 partition is busy.
Either wait, or try `--partition=gpu-a100` in the SLURM file.

---

## Quick reference — commands you'll use most

```bash
sbatch run_mi_experiment.slurm          # submit a job
squeue -u $USER                         # see your jobs
tail -f eegpt_mi_lso_JOBID.out          # watch live output
scancel JOBID                           # cancel a job
watch -n 10 cat results/mi_lso/status.txt  # watch experiment progress
```