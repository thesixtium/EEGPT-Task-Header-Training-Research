"""
experiment_runner.py
--------------------
Top-level orchestration for the EEG motor imagery framework.

Training strategy (LSO enabled):
  1.  Load and preprocess all datasets (with optional .pt disk cache).
  2.  Build a joint dataset with globally-unique subject IDs.
  3.  Train ONE shared base model on all subjects NOT selected as holdouts.
  4.  For each held-out subject:
        a. Load the shared base checkpoint  →  zero-shot evaluation
        b. Progressive session adaptation   →  per-session eval
        c. Save per-subject results
  5.  Generate analytics and summary CSV.

When lso_enabled=False a simple random train/val/test baseline is run instead.

Usage example:

    from experiment_runner import ExperimentConfig, run_experiment
    from moabb.datasets import BNCI2014_004, BNCI2015_001

    cfg = ExperimentConfig(
        experiment_name='mi_2class',
        base_model_path='checkpoints/eegpt_mcae_58chs_4s_large4E.ckpt',
        datasets=[BNCI2014_004(), BNCI2015_001()],
        dataset_ids=['BNCI2014_004', 'BNCI2015_001'],
        max_lr=4e-4,
        base_epochs=20,
        adapt_epochs=10,
        batch_size=32,
        output_dir='results',
    )
    run_experiment(cfg)
"""

from __future__ import annotations

import hashlib
import logging
import time
import datetime
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Silence MNE and MOABB before any other imports so their internal loggers
# are never registered at INFO level.  The messages suppressed here are:
#   "Adding metadata with N columns"  — MNE bookkeeping during epoch loading
#   "X matching events found"         — MNE internal event-detection report
# Both indicate normal operation; they are not errors or warnings.
# ---------------------------------------------------------------------------
import mne
mne.set_log_level("WARNING")

import torch
import pytorch_lightning as pl

from .canonical_channels import CANONICAL_CHANNELS
from .label_schema import LabelSchema, infer_shared_label_schema, DATASET_LABEL_MAPS
from .dataset_registry import MoabbDatasetLoader
from .eeg_dataset import EEGSample, EEGSampleDataset
from .multi_dataset_loader import ConcatDataLoader, subject_split, train_val_test_split
from .leave_subjects_out import LeaveSubjectsOutEvaluator
from .analytics import ExperimentLogger, compute_metrics_from_model, TrainingCurvePlotter
from .preprocessing import TARGET_SAMPLE_RATE

logger = logging.getLogger(__name__)

# Silence MOABB here as well (belt-and-suspenders alongside dataset_registry.py)
logging.getLogger("moabb").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Status logger
# ---------------------------------------------------------------------------

class StatusLogger:
    """
    Writes a continuously-updated status.txt to the experiment output directory.

    Tracks:
      - Current training phase and live epoch progress (via EpochStatusCallback)
      - Per-epoch ETA for both base-model and adaptation training
      - Dataset inventory: subjects per dataset, sample counts, estimated disk size
      - Channel intersection report
      - Per-subject zero-shot and best-adapted results as they complete
      - Overall experiment ETA based on completed fold timings
    """

    def __init__(
        self,
        output_dir: Path,
        experiment_name: str,
        total_subjects: int = 0,
    ) -> None:
        self.path = Path(output_dir) / experiment_name / 'status.txt'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.experiment_name = experiment_name
        self.total_subjects = total_subjects

        self.experiment_start: float = time.time()
        self.current_phase: str = 'Initialising'
        self.current_phase_start: float = time.time()

        self.fold_durations: List[float] = []
        self.folds_completed: int = 0
        self.subject_start_time: Optional[float] = None

        self.stage_start_time: Optional[float] = None
        self.stage_durations: List[float] = []

        self.completed_subjects: List[str] = []

        self._config_block: str = ''
        self._scope_block: str = ''
        self._channel_block: str = ''
        self._dataset_inventory_block: str = ''

        # Live epoch state — updated by EpochStatusCallback
        self._epoch_line: str = ''

        self._refresh()

    # ------------------------------------------------------------------
    # Setup / configuration hooks
    # ------------------------------------------------------------------

    def start(self, config: Dict[str, Any]) -> None:
        lines = ['  ' + f'{k}: {v}' for k, v in config.items()]
        self._config_block = '\n'.join(lines)
        self._refresh()

    def log_dataset_inventory(
        self,
        dataset_map: Dict[str, Any],          # dataset_id → EEGSampleDataset
        subject_id_map: Dict[int, str],        # global_id → "did:sN"
        lso_global_ids: List[int],
    ) -> None:
        """
        Build and store the dataset inventory block shown in status.txt.

        Reports per-dataset: subject count, trial count, session count,
        class distribution, and rough in-memory size estimate.
        """
        from collections import Counter

        lines: List[str] = ['  {:25s}  {:>8s}  {:>8s}  {:>8s}  {:>10s}  {}'.format(
            'Dataset', 'Subjects', 'Sessions', 'Trials', 'Size (MB)', 'Class dist',
        )]
        lines.append('  ' + '-' * 80)

        total_trials = 0
        total_mb = 0.0

        for did, ds in sorted(dataset_map.items()):
            n_subjects = len(ds.get_subject_ids())
            n_sessions = len(ds.get_session_ids())
            n_trials   = len(ds.samples)

            # Size estimate: each sample is [C, 1024] float32
            if ds.samples:
                sample_bytes = ds.samples[0].eeg.numel() * 4   # float32
            else:
                sample_bytes = 0
            mb = (sample_bytes * n_trials) / (1024 ** 2)
            total_mb += mb

            label_counts = Counter(s.label for s in ds.samples)
            dist_str = '  '.join(f'cls{k}:{v}' for k, v in sorted(label_counts.items()))

            lines.append('  {:25s}  {:>8d}  {:>8d}  {:>8d}  {:>10.1f}  {}'.format(
                did, n_subjects, n_sessions, n_trials, mb, dist_str,
            ))
            total_trials += n_trials

        lines.append('  ' + '-' * 80)
        lines.append('  {:25s}  {:>8s}  {:>8s}  {:>8d}  {:>10.1f}'.format(
            'TOTAL', '', '', total_trials, total_mb,
        ))

        # Held-out subjects per dataset
        held_set = set(lso_global_ids)
        per_dataset_held: Dict[str, List[str]] = {}
        for gid, label in subject_id_map.items():
            did, orig = label.split(':s')
            if gid in held_set:
                per_dataset_held.setdefault(did, []).append(orig)

        lines.append('')
        lines.append('  Held-out subjects per dataset:')
        for did in sorted(per_dataset_held):
            origs = ', '.join(sorted(per_dataset_held[did], key=int))
            lines.append(f'    {did}: subject(s) {origs}')

        self._dataset_inventory_block = '\n'.join(lines)
        self._refresh()

    def set_scope(
        self,
        datasets: List[str],
        total_subjects: int,
        base_epochs: int,
        adapt_epochs: int,
        adapt_sessions_per_subject: int,
        training_subjects: int,
    ) -> None:
        stages_per_fold = 1 + adapt_sessions_per_subject
        total_adapt_runs = total_subjects * stages_per_fold
        total_epochs = (
            base_epochs
            + total_subjects * adapt_sessions_per_subject * adapt_epochs
        )

        self._scope_block = '\n'.join([
            f'  Datasets              : {", ".join(datasets)}',
            f'  Training subjects     : {training_subjects}  (used for shared base model)',
            f'  Held-out subjects     : {total_subjects}  (zero-shot + adaptation)',
            f'  Stages per subject    : {stages_per_fold}  '
            f'(zero-shot + {adapt_sessions_per_subject} adaptation)',
            f'  Base model epochs     : {base_epochs}  (trained once)',
            f'  Total adapt runs      : {total_adapt_runs}',
            f'  Total epochs (est)    : {total_epochs}  '
            f'({base_epochs} base + {total_subjects}×{adapt_sessions_per_subject}×{adapt_epochs} adapt)',
        ])
        self.total_subjects = total_subjects
        logger.info(
            '[Status] Scope: %d training subj, %d held-out subj, %d adapt runs, ~%d total epochs',
            training_subjects, total_subjects, total_adapt_runs, total_epochs,
        )
        self._refresh()

    def log_channel_info(self, channel_report: str) -> None:
        """Store the channel intersection report for display in status.txt."""
        self._channel_block = channel_report
        self._refresh()

    # ------------------------------------------------------------------
    # Phase / fold / stage hooks
    # ------------------------------------------------------------------

    def phase(self, description: str) -> None:
        self.current_phase = description
        self.current_phase_start = time.time()
        self._epoch_line = ''
        logger.info('[Status] Phase: %s', description)
        self._refresh()

    def subject_start(
        self,
        subject_id: int,
        dataset_id: str,
        fold_idx: int,
        total_folds: int,
    ) -> None:
        self.current_phase = (
            f'LSO fold {fold_idx}/{total_folds} — '
            f'dataset={dataset_id}, held-out subject={subject_id}'
        )
        self.subject_start_time = time.time()
        self.stage_durations = []
        self._epoch_line = ''
        self._refresh()

    def stage_start(self, subject_id: int, stage: int, total_stages: int) -> None:
        label = 'zero-shot eval' if stage == 0 else f'adaptation stage {stage}/{total_stages - 1}'
        self.current_phase = f'Subject {subject_id} — {label}'
        self.stage_start_time = time.time()
        self._epoch_line = ''
        self._refresh()

    def stage_end(self, subject_id: int, stage: int, metrics: Dict[str, float]) -> None:
        duration = time.time() - (self.stage_start_time or time.time())
        self.stage_durations.append(duration)
        bal_acc = metrics.get('valid_balanced_accuracy', float('nan'))
        acc     = metrics.get('valid_accuracy', float('nan'))
        f1      = metrics.get('valid_f1_macro', float('nan'))
        label   = 'zero-shot' if stage == 0 else f'stage {stage}'
        logger.info(
            '[Status] Subject %d %s — bal_acc=%.3f  acc=%.3f  f1_macro=%.3f  (%.1fs)',
            subject_id, label, bal_acc, acc, f1, duration,
        )
        self._epoch_line = ''
        self._refresh()

    def subject_end(
        self,
        subject_id: int,
        zero_shot: Dict[str, float],
        best_adapted: Dict[str, float],
    ) -> None:
        duration = time.time() - (self.subject_start_time or time.time())
        self.fold_durations.append(duration)
        self.folds_completed += 1

        zs_ba = zero_shot.get('valid_balanced_accuracy', float('nan'))
        ad_ba = best_adapted.get('valid_balanced_accuracy', float('nan'))
        gain  = (ad_ba - zs_ba) if not (zs_ba != zs_ba or ad_ba != ad_ba) else float('nan')

        summary = (
            f'  Subject {subject_id:>3d} | '
            f'zero-shot={zs_ba:.3f}  best-adapted={ad_ba:.3f}  '
            f'gain={gain:+.3f}  ({_fmt_duration(duration)})'
        )
        self.completed_subjects.append(summary)
        self._epoch_line = ''
        self._refresh()

    # ------------------------------------------------------------------
    # Live epoch update — called by EpochStatusCallback every epoch
    # ------------------------------------------------------------------

    def update_epoch(
        self,
        current_epoch: int,
        max_epochs: int,
        epoch_duration_s: float,
        train_loss: Optional[float],
        val_loss: Optional[float],
        val_acc: Optional[float],
        val_bal_acc: Optional[float],
        run_label: str = '',
    ) -> None:
        """Update the live epoch progress line and refresh status.txt."""
        remaining = max_epochs - current_epoch
        eta_epoch  = _fmt_duration(epoch_duration_s * remaining) if remaining > 0 else 'done'
        finish_at  = (
            datetime.datetime.now() + datetime.timedelta(seconds=epoch_duration_s * remaining)
        ).strftime('%H:%M:%S') if remaining > 0 else '—'

        parts = [
            f'  Epoch     : {current_epoch}/{max_epochs}',
            f'  Epoch time: {_fmt_duration(epoch_duration_s)}',
            f'  ETA (run) : {eta_epoch}  (est. {finish_at})',
        ]
        if run_label:
            parts.insert(0, f'  Run       : {run_label}')
        if train_loss is not None:
            parts.append(f'  train_loss: {train_loss:.4f}')
        if val_loss is not None:
            parts.append(f'  val_loss  : {val_loss:.4f}')
        if val_acc is not None:
            parts.append(f'  val_acc   : {val_acc:.4f}')
        if val_bal_acc is not None:
            parts.append(f'  val_bal_ac: {val_bal_acc:.4f}')

        self._epoch_line = '\n'.join(parts)
        self._refresh()

    # ------------------------------------------------------------------
    # Finish
    # ------------------------------------------------------------------

    def finish(self, final_metrics: Dict[str, Any]) -> None:
        self.current_phase = 'COMPLETE'
        self._epoch_line = ''
        total = time.time() - self.experiment_start
        lines = [f'  {k}: {v}' for k, v in final_metrics.items()]
        footer = '\n'.join(lines)
        self._refresh(footer=footer, total_elapsed=total)
        logger.info('[Status] Experiment complete. Total time: %s', _fmt_duration(total))

    # ------------------------------------------------------------------
    # Internal render
    # ------------------------------------------------------------------

    def _refresh(
        self,
        footer: str = '',
        total_elapsed: Optional[float] = None,
    ) -> None:
        now     = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        elapsed = total_elapsed if total_elapsed is not None else (
            time.time() - self.experiment_start
        )
        phase_elapsed = time.time() - self.current_phase_start
        eta_str       = self._eta_string()

        lines = [
            '=' * 72,
            f'  EXPERIMENT: {self.experiment_name}',
            f'  Updated   : {now}',
            f'  Elapsed   : {_fmt_duration(elapsed)}',
            f'  ETA       : {eta_str}',
            '=' * 72,
            '',
            'CONFIGURATION',
            '-' * 40,
            self._config_block,
            '',
            'CURRENT STATUS',
            '-' * 40,
            f'  Phase     : {self.current_phase}',
            f'  Phase time: {_fmt_duration(phase_elapsed)}',
            f'  Folds done: {self.folds_completed} / {self.total_subjects}',
        ]

        if self._epoch_line:
            lines += ['', 'EPOCH PROGRESS', '-' * 40, self._epoch_line]

        if self.stage_durations:
            avg_stage = sum(self.stage_durations) / len(self.stage_durations)
            lines.append(f'  Avg stage : {_fmt_duration(avg_stage)} per stage')

        if self.fold_durations:
            avg_fold = sum(self.fold_durations) / len(self.fold_durations)
            lines.append(f'  Avg fold  : {_fmt_duration(avg_fold)} per subject')

        if self._dataset_inventory_block:
            lines += [
                '',
                'DATASET INVENTORY',
                '-' * 40,
                self._dataset_inventory_block,
            ]

        if self._channel_block:
            lines += [
                '',
                'CHANNEL INTERSECTION',
                '-' * 40,
                *('  ' + l for l in self._channel_block.splitlines()),
            ]

        if self._scope_block:
            lines += [
                '',
                'EXPERIMENT SCOPE',
                '-' * 40,
                self._scope_block,
            ]

        if self.completed_subjects:
            lines += [
                '',
                'COMPLETED SUBJECTS',
                '-' * 40,
            ] + self.completed_subjects

        if footer:
            lines += [
                '',
                'FINAL RESULTS',
                '-' * 40,
                footer,
            ]

        lines += ['', '=' * 72]

        try:
            self.path.write_text('\n'.join(lines), encoding='utf-8')
        except OSError as exc:
            logger.warning('StatusLogger: could not write status.txt: %s', exc)

    def _eta_string(self) -> str:
        if not self.fold_durations or self.total_subjects == 0:
            return 'estimating...'
        remaining_folds = self.total_subjects - self.folds_completed
        if remaining_folds <= 0:
            return 'done'
        avg = sum(self.fold_durations) / len(self.fold_durations)
        eta_secs = avg * remaining_folds
        finish_at = datetime.datetime.now() + datetime.timedelta(seconds=eta_secs)
        return f'{_fmt_duration(eta_secs)} remaining  (est. finish {finish_at.strftime("%H:%M:%S")})'


# ---------------------------------------------------------------------------
# PyTorch Lightning callback — feeds live epoch data back into StatusLogger
# ---------------------------------------------------------------------------

class EpochStatusCallback(pl.Callback):
    """
    Attaches to a pl.Trainer and writes one status update per epoch end
    into the shared StatusLogger, including epoch duration and val metrics.
    """

    def __init__(self, status: 'StatusLogger', max_epochs: int, run_label: str = '') -> None:
        super().__init__()
        self._status    = status
        self._max_epochs = max_epochs
        self._run_label  = run_label
        self._epoch_start: float = time.time()

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        self._epoch_start = time.time()

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        # Skip the sanity-check pass (epoch == 0 before training starts)
        if trainer.sanity_checking:
            return

        duration = time.time() - self._epoch_start
        current  = trainer.current_epoch + 1   # 0-indexed → 1-indexed for display
        logged   = trainer.callback_metrics

        def _get(key: str) -> Optional[float]:
            v = logged.get(key)
            return float(v) if v is not None else None

        self._status.update_epoch(
            current_epoch    = current,
            max_epochs       = self._max_epochs,
            epoch_duration_s = duration,
            train_loss       = _get('train_loss'),
            val_loss         = _get('valid_loss'),
            val_acc          = _get('valid_acc'),
            val_bal_acc      = _get('valid_balanced_accuracy'),
            run_label        = self._run_label,
        )


def _fmt_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f'{h}h {m:02d}m {s:02d}s'
    if m:
        return f'{m}m {s:02d}s'
    return f'{s}s'


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    experiment_name: str
    base_model_path: str                        # path to pretrained EEGPT .ckpt

    # Datasets — parallel lists: moabb objects + their registry IDs
    datasets: List[Any]                         # MOABB dataset instances
    dataset_ids: List[str]                      # matching keys in DATASET_LABEL_MAPS

    # Training hypers
    max_lr: float = 4e-4
    base_epochs: int = 20
    adapt_epochs: int = 10
    batch_size: int = 32
    data_fraction: float = 1.0

    # Evaluation
    lso_enabled: bool = True
    # Number of subjects to randomly hold out *per dataset* (one "test-set
    # slot" per dataset).  For example, lso_subjects_per_test_set=1 holds
    # out 1 subject from each dataset → N_datasets held-out subjects total.
    # The selection is seeded by `seed` for full reproducibility.
    # All held-out subjects are excluded from base-model training so the
    # base model is trained exactly once on the remaining pool.
    lso_subjects_per_test_set: int = 1
    test_session_idx: int = 0

    # Dataset loading
    subjects_per_dataset: Optional[List[Optional[List[int]]]] = None
    fmin: float = 0.5
    fmax: float = 40.0
    tmin: float = 0.0
    tmax: float = 4.0
    resample: Optional[float] = float(TARGET_SAMPLE_RATE)

    # Preprocessed dataset cache directory.
    # Set to a path to enable caching; None disables caching.
    dataset_cache_dir: Optional[str] = None

    # Base model checkpoint behaviour.
    # Set force_retrain_base=True to delete any existing base_model.ckpt and
    # retrain from scratch.  Useful after changing datasets, channel set, or
    # label space.  Defaults to False so resuming a run is fast.
    force_retrain_base: bool = False

    # Outputs
    output_dir: str = 'results'
    seed: int = 711_2002

    # Analytics
    analytics_metrics: List[str] = field(default_factory=lambda: [
        'valid_accuracy', 'valid_balanced_accuracy',
        'valid_f1_macro', 'valid_mcc', 'valid_cohen_kappa',
    ])


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def make_model_factory(base_model_path: str, max_lr: float):
    """
    Return a callable that constructs a fresh GenericEEGPTModel loaded from
    the pretrained backbone at base_model_path.
    """
    from src.core.genericEEGPTModel import GenericEEGPTModel
    from src.core.generic_eegpt_model_lib.modelMethods import seed_torch

    def factory(
        channel_names: List[str],
        n_classes: int,
        steps_per_epoch: int = 100,
        max_epochs: int = 20,
    ) -> GenericEEGPTModel:
        seed_torch(711_2002)
        return GenericEEGPTModel(
            load_path=base_model_path,
            use_channels_names=channel_names,
            output_classes=n_classes,
            max_lr=max_lr,
            steps_per_epoch=steps_per_epoch,
            max_epochs=max_epochs,
        )

    return factory


# ---------------------------------------------------------------------------
# Dataset loading (with optional .pt cache)
# ---------------------------------------------------------------------------

def load_datasets(
    cfg: ExperimentConfig,
) -> Dict[str, EEGSampleDataset]:
    """
    Load all configured datasets.

    Per-dataset cache behaviour (when cfg.dataset_cache_dir is set):
      - The cache key is an MD5 hash of dataset ID + preprocessing params +
        label map.  If the matching .pt file exists it is loaded from disk
        and MOABB / preprocessing are skipped entirely.
      - A cache HIT is logged as:
            [X/N] DatasetName — loaded from cache: path/to/file.pt (M samples)
      - A cache MISS is logged as:
            [X/N] DatasetName — not cached, running MOABB download + preprocessing
        followed by the per-subject progress lines emitted by MoabbDatasetLoader.
      - After a successful MOABB load the result is written to cache:
            [X/N] DatasetName — saved to cache: path/to/file.pt

    Delete the cache directory (or individual .pt files) to force a reload.
    """
    shared_labels, schemas = infer_shared_label_schema(cfg.dataset_ids)
    logger.info("Shared label space: %s", shared_labels)

    cache_dir: Optional[Path] = None
    if cfg.dataset_cache_dir:
        cache_dir = Path(cfg.dataset_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Dataset cache directory: %s", cache_dir)
    else:
        logger.info("Dataset caching disabled (set dataset_cache_dir to enable)")

    loaded: Dict[str, EEGSampleDataset] = {}
    n_datasets = len(cfg.datasets)

    for i, (moabb_ds, did) in enumerate(zip(cfg.datasets, cfg.dataset_ids), start=1):
        schema = schemas[did]
        subjects = None
        if cfg.subjects_per_dataset and (i - 1) < len(cfg.subjects_per_dataset):
            subjects = cfg.subjects_per_dataset[i - 1]

        ds_prefix = f"[{i}/{n_datasets}] {did}"

        # ── Try whole-dataset cache first ─────────────────────────────────
        # This is a dataset-level cache (one .pt per dataset).  The
        # per-subject cache inside MoabbDatasetLoader is a finer-grained
        # fallback used when this file does not exist yet.
        if cache_dir is not None:
            schema_key = "|".join(sorted(DATASET_LABEL_MAPS[did].keys()))
            cache_key = hashlib.md5(
                f"{did}|{subjects}|{cfg.fmin}|{cfg.fmax}|"
                f"{cfg.tmin}|{cfg.tmax}|{cfg.resample}|{schema_key}".encode()
            ).hexdigest()[:10]
            cache_path = cache_dir / f"{did}_{cache_key}.pt"

            if cache_path.exists():
                logger.info(
                    "%s — loading full dataset from cache: %s",
                    ds_prefix, cache_path,
                )
                try:
                    ds = torch.load(cache_path, weights_only=False)
                    loaded[did] = ds
                    logger.info(
                        "%s — cache hit: %d samples, %d classes",
                        ds_prefix, len(ds), schema.n_classes,
                    )
                    continue
                except Exception as exc:
                    logger.warning(
                        "%s — cache load failed (%s) — re-running MOABB",
                        ds_prefix, exc,
                    )
            else:
                logger.info(
                    "%s — no full-dataset cache found, running MOABB download + preprocessing",
                    ds_prefix,
                )

        # ── Load from MOABB + preprocess ──────────────────────────────────
        # dataset_index / dataset_total are passed so MoabbDatasetLoader can
        # emit the same [X/N] prefix on its per-subject progress lines.
        loader = MoabbDatasetLoader(
            moabb_dataset=moabb_ds,
            dataset_id=did,
            label_schema=schema,
            n_classes=len(DATASET_LABEL_MAPS[did]),
            fmin=cfg.fmin,
            fmax=cfg.fmax,
            tmin=cfg.tmin,
            tmax=cfg.tmax,
            resample=cfg.resample,
            subjects=subjects,
            dataset_index=i,
            dataset_total=n_datasets,
            cache_dir=cfg.dataset_cache_dir,   # per-subject cache inside the loader
        )
        ds = loader.load_all_subjects()
        loaded[did] = ds
        logger.info(
            "%s — loaded %d samples, %d classes",
            ds_prefix, len(ds), schema.n_classes,
        )

        # ── Save whole-dataset cache ──────────────────────────────────────
        if cache_dir is not None:
            try:
                torch.save(ds, cache_path)
                logger.info("%s — saved full dataset to cache: %s", ds_prefix, cache_path)
            except Exception as exc:
                logger.warning("%s — could not write cache: %s", ds_prefix, exc)

    # ── Optional data fraction ────────────────────────────────────────────
    if cfg.data_fraction < 1.0:
        import random
        from collections import defaultdict

        rng = random.Random(cfg.seed)
        for did in list(loaded.keys()):
            ds = loaded[did]
            # Stratified subsample: preserve class balance
            by_label = defaultdict(list)
            for idx, s in enumerate(ds.samples):
                by_label[s.label].append(idx)

            keep = []
            for label_indices in by_label.values():
                rng.shuffle(label_indices)
                n_keep = max(1, round(len(label_indices) * cfg.data_fraction))
                keep.extend(label_indices[:n_keep])

            loaded[did] = EEGSampleDataset([ds.samples[idx] for idx in sorted(keep)])
            logger.info(
                "data_fraction=%.2f: %s reduced %d → %d samples",
                cfg.data_fraction, did, len(ds), len(loaded[did]),
            )

    return loaded


# ---------------------------------------------------------------------------
# Joint dataset builder (globally unique subject IDs + channel intersection)
# ---------------------------------------------------------------------------

def build_joint_dataset(
    dataset_map: Dict[str, EEGSampleDataset],
    channel_names: List[str],
) -> Tuple[EEGSampleDataset, Dict[int, str]]:
    """
    Concatenate all per-dataset EEGSampleDatasets into one joint pool.

    Subject IDs are remapped to globally unique integers so that the LSO
    exclusion logic works correctly across datasets (e.g. both BNCI2014_004
    and BNCI2015_001 have a "subject 1" — they become distinct global IDs).

    Every sample's EEG tensor is trimmed to the joint channel intersection
    so all samples share the same [C_joint, 1024] layout.

    Note: dataset_map is consumed (emptied) as each dataset is processed so
    that the original per-dataset tensors are released before the remapped
    copies are built, keeping peak memory to ~1x rather than ~2x.

    Returns
    -------
    joint_dataset : EEGSampleDataset  -- all samples, globally unique subject IDs
    subject_id_map : Dict[global_id -> "dataset_id:s{orig_id}"]  -- for logging
    """
    remapped_datasets: List[EEGSampleDataset] = []
    subject_id_map: Dict[int, str] = {}
    global_id_counter = 0

    for did in list(dataset_map.keys()):
        ds = dataset_map.pop(did)   # release original once remapped copy is built
        orig_subject_ids = ds.get_subject_ids()
        id_remap: Dict[int, int] = {}
        for orig_id in orig_subject_ids:
            global_id_counter += 1
            id_remap[orig_id] = global_id_counter
            subject_id_map[global_id_counter] = f'{did}:s{orig_id}'

        sample_ch = ds.get_channel_names()
        ch_set = {ch: i for i, ch in enumerate(sample_ch)}
        # channel_names is already the intersection of all datasets, so every
        # channel here is guaranteed to be present in this dataset.
        channel_indices = [ch_set[ch] for ch in channel_names]

        remapped_samples = []
        for sample in ds.samples:
            eeg_trimmed = sample.eeg[channel_indices, :]   # [C_joint, 1024]
            remapped_samples.append(EEGSample(
                eeg=eeg_trimmed,
                channel_names=channel_names,
                label=sample.label,
                subject_id=id_remap[sample.subject_id],
                session_id=sample.session_id,
                dataset_id=sample.dataset_id,
            ))
        remapped_datasets.append(EEGSampleDataset(remapped_samples))

    joint_dataset = EEGSampleDataset.concat(remapped_datasets)
    logger.info(
        "Joint dataset: %d samples, %d unique subjects across %d datasets",
        len(joint_dataset),
        len(joint_dataset.get_subject_ids()),
        len(remapped_datasets),
    )
    return joint_dataset, subject_id_map


# ---------------------------------------------------------------------------
# Shared base model training
# ---------------------------------------------------------------------------

def train_base_model(
    training_dataset: EEGSampleDataset,
    channel_names: List[str],
    n_classes: int,
    cfg: ExperimentConfig,
    model_factory,
    checkpoints_dir: Path,
    status: StatusLogger,
    accelerator: str = 'cpu',
    logs_dir: Optional[Path] = None,
    exp_logger=None,
) -> Path:
    """
    Train a single shared base model on training_dataset (all non-held-out
    subjects) and save the checkpoint.

    Returns the path to the saved checkpoint.
    """
    base_ckpt_path = checkpoints_dir / 'base_model.ckpt'

    if base_ckpt_path.exists() and cfg.force_retrain_base:
        logger.warning(
            "force_retrain_base=True — deleting existing checkpoint: %s",
            base_ckpt_path,
        )
        base_ckpt_path.unlink()

    if base_ckpt_path.exists():
        logger.warning(
            "Reusing base_model.ckpt from a PREVIOUS run: %s\n"
            "  This checkpoint may have been trained on different data, channels, "
            "or labels.\n"
            "  Set force_retrain_base=True in ExperimentConfig to retrain from scratch.",
            base_ckpt_path,
        )
        status.phase(
            "Shared base model checkpoint found — skipping retraining "
            "(set force_retrain_base=True to override)"
        )
        return base_ckpt_path

    status.phase('Training shared base model on all training subjects')
    logger.info(
        "Training shared base model: %d subjects, %d samples",
        len(training_dataset.get_subject_ids()),
        len(training_dataset),
    )

    train_set, val_set, _ = train_val_test_split(
        training_dataset,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=cfg.seed,
    )

    train_loader_obj = ConcatDataLoader(
        [train_set],
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    val_loader_obj = ConcatDataLoader(
        [val_set],
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    model = model_factory(
        channel_names=channel_names,
        n_classes=n_classes,
        steps_per_epoch=train_loader_obj.steps_per_epoch,
        max_epochs=cfg.base_epochs,
    )

    trainer = pl.Trainer(
        accelerator=accelerator,
        max_epochs=cfg.base_epochs,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        default_root_dir=str(checkpoints_dir),
        callbacks=[EpochStatusCallback(status, cfg.base_epochs, run_label='Base model training')],
        logger=[
            pl.loggers.CSVLogger(
                str(logs_dir) if logs_dir else str(checkpoints_dir / 'logs'),
                name='base_model_training',
            )
        ],
    )
    trainer.fit(
        model,
        train_loader_obj.get_loader(),
        val_loader_obj.get_loader(),
    )

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(base_ckpt_path))
    logger.info("Shared base model saved to: %s", base_ckpt_path)

    # ── Plot training curves ───────────────────────────────────────────────
    _effective_logs_dir = Path(logs_dir) if logs_dir else checkpoints_dir / 'logs'
    if exp_logger is not None:
        exp_logger.generate_base_model_training_curves(
            logs_dir=_effective_logs_dir,
            run_name='base_model_training',
        )
    else:
        from .analytics import TrainingCurvePlotter
        TrainingCurvePlotter.plot_from_trainer(
            logs_dir=_effective_logs_dir,
            run_name='base_model_training',
            save_dir=_effective_logs_dir / 'base_model_training' / 'plots',
        )

    return base_ckpt_path


# ---------------------------------------------------------------------------
# Per-subject zero-shot + progressive adaptation
# ---------------------------------------------------------------------------

def run_subject_fold(
    held_out_global_id: int,
    joint_dataset: EEGSampleDataset,
    base_ckpt_path: Path,
    channel_names: List[str],
    n_classes: int,
    cfg: ExperimentConfig,
    model_factory,
    checkpoints_dir: Path,
    logs_dir: Path,
    status: StatusLogger,
    subject_label: str,
    fold_idx: int,
    total_folds: int,
    accelerator: str = 'cpu',
) -> Dict[str, Any]:
    """
    For one held-out subject:
      stage 0 : zero-shot  — evaluate the shared base model directly
      stage 1+: progressive adaptation on each training session in order,
                held-out test session is never used for training

    Returns a dict with per-stage metrics and subject metadata.
    """
    from src.core.genericEEGPTModel import GenericEEGPTModel

    subject_data = joint_dataset.filter_by_subject(held_out_global_id)
    session_ids = subject_data.get_session_ids()

    # Identify test session (held out from adaptation)
    if cfg.test_session_idx < len(session_ids):
        test_session_id = session_ids[cfg.test_session_idx]
    else:
        logger.warning(
            "test_session_idx=%d but subject %d only has %d sessions — "
            "using last session as test.",
            cfg.test_session_idx, held_out_global_id, len(session_ids),
        )
        test_session_id = session_ids[-1]

    test_data = subject_data.filter_by_session(test_session_id)
    adapt_sessions = [sid for sid in session_ids if sid != test_session_id]

    test_loader = ConcatDataLoader(
        [test_data], batch_size=cfg.batch_size, shuffle=False
    ).get_loader()

    total_stages = 1 + len(adapt_sessions)   # zero-shot + one per adapt session
    stage_results: List[Dict[str, Any]] = []

    # ── Stage 0: zero-shot eval ───────────────────────────────────────────
    status.stage_start(held_out_global_id, stage=0, total_stages=total_stages)

    model = model_factory(
        channel_names=channel_names,
        n_classes=n_classes,
        steps_per_epoch=1,
        max_epochs=cfg.adapt_epochs,
    )
    # Load the shared base checkpoint weights into this fresh model instance
    base_state = torch.load(str(base_ckpt_path), map_location='cpu', weights_only=False)
    # pytorch-lightning checkpoints nest state under 'state_dict'
    state_dict = base_state.get('state_dict', base_state)
    model.load_state_dict(state_dict, strict=False)

    zero_shot_metrics = compute_metrics_from_model(model, test_loader)
    logger.info(
        "Subject %s — zero-shot: %s", subject_label, zero_shot_metrics
    )
    stage_results.append({'stage': 0, 'session_id': None, 'metrics': zero_shot_metrics})
    status.stage_end(held_out_global_id, stage=0, metrics=zero_shot_metrics)

    # ── Stages 1+: progressive session adaptation ─────────────────────────
    # Start from the base checkpoint each time? No — we carry forward the
    # adapted model weights so adaptation is truly progressive (each session
    # builds on the previous one).
    current_model = model   # already loaded from base ckpt above

    for stage_idx, sess_id in enumerate(adapt_sessions, start=1):
        status.stage_start(held_out_global_id, stage=stage_idx, total_stages=total_stages)

        adapt_data = subject_data.filter_by_session(sess_id)
        if len(adapt_data) == 0:
            logger.warning(
                "Subject %s session %d has no samples — skipping adaptation stage %d.",
                subject_label, sess_id, stage_idx,
            )
            stage_results.append({
                'stage': stage_idx,
                'session_id': sess_id,
                'metrics': {},
            })
            status.stage_end(held_out_global_id, stage=stage_idx, metrics={})
            continue

        adapt_train, adapt_val, _ = train_val_test_split(
            adapt_data,
            val_ratio=0.2,
            test_ratio=0.0,   # no held-out split within the adapt set
            seed=cfg.seed,
        ) if len(adapt_data) > 5 else (adapt_data, adapt_data, adapt_data)

        adapt_train_loader = ConcatDataLoader(
            [adapt_train], batch_size=min(cfg.batch_size, len(adapt_train)), shuffle=True
        )
        adapt_val_loader = ConcatDataLoader(
            [adapt_val], batch_size=cfg.batch_size, shuffle=False
        )

        # Re-instantiate the model so the OneCycleLR scheduler is built fresh
        # with the correct total_steps for this adaptation loader.  Simply
        # mutating hparams on an already-configured model leaves the live
        # scheduler's total_steps stale, causing "Tried to step N times" errors.
        subject_ckpt_dir = checkpoints_dir / f'subject_{held_out_global_id}'
        subject_ckpt_dir.mkdir(parents=True, exist_ok=True)

        prev_state = {k: v.clone() for k, v in current_model.state_dict().items()}
        current_model = model_factory(
            channel_names=channel_names,
            n_classes=n_classes,
            steps_per_epoch=adapt_train_loader.steps_per_epoch,
            max_epochs=cfg.adapt_epochs,
        )
        current_model.load_state_dict(prev_state, strict=False)

        adapt_trainer = pl.Trainer(
            accelerator=accelerator,
            max_epochs=cfg.adapt_epochs,
            log_every_n_steps=1,
            num_sanity_val_steps=0,
            default_root_dir=str(subject_ckpt_dir),
            callbacks=[EpochStatusCallback(
                status, cfg.adapt_epochs,
                run_label=f'Subject {subject_label} — adapt stage {stage_idx} (session {sess_id})',
            )],
        )
        adapt_trainer.fit(
            current_model,
            adapt_train_loader.get_loader(),
            adapt_val_loader.get_loader(),
        )

        stage_metrics = compute_metrics_from_model(current_model, test_loader)
        logger.info(
            "Subject %s — stage %d (session %d): %s",
            subject_label, stage_idx, sess_id, stage_metrics,
        )
        stage_results.append({
            'stage': stage_idx,
            'session_id': sess_id,
            'metrics': stage_metrics,
        })
        status.stage_end(held_out_global_id, stage=stage_idx, metrics=stage_metrics)

    # ── Summarise ─────────────────────────────────────────────────────────
    best_adapted = max(
        (r['metrics'] for r in stage_results if r['stage'] > 0 and r['metrics']),
        key=lambda m: m.get('valid_balanced_accuracy', 0.0),
        default={},
    )

    return {
        'global_subject_id': held_out_global_id,
        'subject_label': subject_label,
        'stage_results': stage_results,
        'zero_shot_metrics': zero_shot_metrics,
        'best_adapted_metrics': best_adapted,
        'n_adapt_sessions': len(adapt_sessions),
        'n_test_trials': len(test_data),
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_experiment(cfg: ExperimentConfig) -> None:
    """
    Execute the full experiment pipeline described in ExperimentConfig.

    Steps:
      1. Load and preprocess all datasets (with optional disk cache)
      2. Build joint dataset with globally unique subject IDs
      3. Train ONE shared base model on all subjects NOT selected as holdouts
      4. For each held-out subject:
           a. Zero-shot evaluation using the shared base model
           b. Progressive session adaptation
           c. Save per-subject results
      5. Generate analytics and summary CSV
    """
    import random
    import numpy as np

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # ── Allow MOABB download dir to be overridden by environment variable ──
    # The SLURM script sets MOABB_DOWNLOAD_DIR to /scratch/$SLURM_JOB_ID/mne_data
    # so raw downloads land on fast scratch storage and don't eat /home quota.
    import os as _os
    _moabb_env_dir = _os.environ.get('MOABB_DOWNLOAD_DIR')
    if _moabb_env_dir:
        try:
            from moabb.utils import set_download_dir as _set_dl
            _set_dl(_moabb_env_dir)
            Path(_moabb_env_dir).mkdir(parents=True, exist_ok=True)
            logger.info("MOABB download dir set from env: %s", _moabb_env_dir)
        except Exception as _e:
            logger.warning("Could not set MOABB download dir from env: %s", _e)

    # ── Detect accelerator ────────────────────────────────────────────────
    if torch.cuda.is_available():
        _accelerator = 'gpu'
        _device_name = torch.cuda.get_device_name(0)
        logger.info("GPU detected — using CUDA device: %s", _device_name)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        _accelerator = 'mps'
        logger.info("Apple Silicon GPU detected — using MPS accelerator")
    else:
        _accelerator = 'cpu'
        logger.info("No GPU detected — running on CPU")

    output_root = Path(cfg.output_dir) / cfg.experiment_name
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = output_root / 'checkpoints'
    logs_dir = output_root / 'logs'
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    exp_logger = ExperimentLogger(
        output_dir=Path(cfg.output_dir),
        experiment_name=cfg.experiment_name,
    )
    config_dict = {
        'experiment_name': cfg.experiment_name,
        'base_model_path': cfg.base_model_path,
        'dataset_ids': cfg.dataset_ids,
        'max_lr': cfg.max_lr,
        'base_epochs': cfg.base_epochs,
        'adapt_epochs': cfg.adapt_epochs,
        'batch_size': cfg.batch_size,
        'lso_enabled': cfg.lso_enabled,
        'lso_subjects_per_test_set': cfg.lso_subjects_per_test_set,
        'seed': cfg.seed,
    }
    exp_logger.log_config(config_dict)

    status = StatusLogger(
        output_dir=Path(cfg.output_dir),
        experiment_name=cfg.experiment_name,
        total_subjects=0,
    )
    status.start(config_dict)

    # ── 1. Load datasets ─────────────────────────────────────────────────
    status.phase('Loading and preprocessing datasets')
    dataset_map = load_datasets(cfg)

    # ── 2. Shared label space & channel intersection ──────────────────────
    shared_labels, schemas = infer_shared_label_schema(cfg.dataset_ids)
    n_classes = len(shared_labels)

    # Compute the intersection of channels present in every loaded dataset.
    # We iterate in a fixed canonical order so the result is deterministic.
    # A detailed per-dataset breakdown is written to the status log so the
    # researcher can see exactly which channels were dropped and why.
    channel_names: List[str] = list(CANONICAL_CHANNELS)
    per_dataset_ch: Dict[str, List[str]] = {}
    for did, ds in dataset_map.items():
        per_dataset_ch[did] = ds.get_channel_names()

    for did, ds_channels in per_dataset_ch.items():
        ch_set = set(ds_channels)
        dropped = [c for c in channel_names if c not in ch_set]
        channel_names = [c for c in channel_names if c in ch_set]
        if dropped:
            logger.info(
                "Channel intersection — %s drops %d canonical channel(s): %s",
                did, len(dropped), dropped,
            )

    assert channel_names, (
        "No channels remain after intersecting all datasets. "
        "Check that dataset channel names are being normalised correctly "
        "in canonical_channels.py."
    )

    # Build a human-readable per-dataset channel report for the status log.
    ch_report_lines = [
        f"Joint channel intersection: {len(channel_names)} channels",
        f"  Retained : {channel_names}",
        "",
        "  Per-dataset coverage:",
    ]
    for did, ds_channels in per_dataset_ch.items():
        ds_ch_set = set(ds_channels)
        present = [c for c in channel_names if c in ds_ch_set]
        absent  = [c for c in channel_names if c not in ds_ch_set]
        ch_report_lines.append(
            f"    {did}: {len(ds_channels)} raw channels → "
            f"{len(present)}/{len(channel_names)} intersection channels present"
            + (f", absent from intersection: {absent}" if absent else "")
        )
    ch_report = "\n".join(ch_report_lines)
    logger.info(ch_report)
    status.log_channel_info(ch_report)

    model_factory = make_model_factory(cfg.base_model_path, cfg.max_lr)

    # ── 3. [Optional] Simple baseline (LSO disabled) ──────────────────────
    if not cfg.lso_enabled:
        status.phase('Baseline training (LSO disabled) — random train/val/test split')
        status.phase('Building joint dataset')

        joint, _ = build_joint_dataset(dataset_map, channel_names)
        # dataset_map is now empty; do not reference it below
        train_set, val_set, test_set = train_val_test_split(joint)

        train_loader_obj = ConcatDataLoader(
            [train_set], batch_size=cfg.batch_size, shuffle=True
        )
        val_loader_obj = ConcatDataLoader(
            [val_set], batch_size=cfg.batch_size, shuffle=False
        )

        model = model_factory(
            channel_names,
            n_classes,
            steps_per_epoch=train_loader_obj.steps_per_epoch,
            max_epochs=cfg.base_epochs,
        )
        trainer = pl.Trainer(
            accelerator=_accelerator,
            max_epochs=cfg.base_epochs,
            log_every_n_steps=1,
            num_sanity_val_steps=0,
            callbacks=[EpochStatusCallback(status, cfg.base_epochs, run_label='Baseline training')],
        )
        trainer.fit(model, train_loader_obj.get_loader(), val_loader_obj.get_loader())
        test_metrics = compute_metrics_from_model(
            model,
            ConcatDataLoader([test_set], batch_size=cfg.batch_size, shuffle=False).get_loader(),
        )
        logger.info("Baseline test metrics: %s", test_metrics)
        exp_logger.log_config({'baseline_test_metrics': test_metrics})
        status.finish(test_metrics)
        return

    # ── 4. Build joint dataset ────────────────────────────────────────────
    # Snapshot per-dataset metadata for inventory logging BEFORE build_joint_dataset
    # consumes dataset_map (it pops each entry to keep peak memory at ~1x).
    _inventory_snapshot: Dict[str, EEGSampleDataset] = {
        did: ds for did, ds in dataset_map.items()
    }

    status.phase('Building joint dataset across all datasets')
    # dataset_map is emptied by build_joint_dataset (intentional — see its docstring).
    joint_dataset, subject_id_map = build_joint_dataset(dataset_map, channel_names)

    # ── 5. Resolve which subject IDs are held out ─────────────────────────
    # Randomly sample lso_subjects_per_test_set subjects from *each* dataset,
    # seeded for reproducibility.  All holdouts are resolved upfront so the
    # base model is trained exactly once on the remaining pool.
    import random as _random
    _rng = _random.Random(cfg.seed)

    # Group global IDs by dataset_id so we can sample per-dataset.
    from collections import defaultdict as _defaultdict
    _by_dataset: Dict[str, List[int]] = _defaultdict(list)
    for gid, label in subject_id_map.items():
        did = label.split(':s')[0]
        _by_dataset[did].append(gid)

    n_per = cfg.lso_subjects_per_test_set
    lso_global_ids: List[int] = []
    for did, gids in sorted(_by_dataset.items()):   # sorted for determinism
        if len(gids) < n_per:
            logger.warning(
                "Dataset %s has only %d subject(s) but lso_subjects_per_test_set=%d; "
                "holding out all %d.",
                did, len(gids), n_per, len(gids),
            )
            chosen = gids
        else:
            chosen = _rng.sample(sorted(gids), n_per)   # sorted before sampling -> deterministic
        lso_global_ids.extend(chosen)
        logger.info(
            "LSO sampling -- %s: holding out %d subject(s): global_ids=%s  "
            "(original IDs: %s)",
            did, len(chosen), chosen,
            [subject_id_map[g] for g in chosen],
        )

    lso_global_ids_set = set(lso_global_ids)
    training_global_ids = [
        sid for sid in joint_dataset.get_subject_ids()
        if sid not in lso_global_ids_set
    ]

    logger.info(
        "Held-out subjects: %d  |  Training subjects: %d  "
        "(base model trained exactly once on the training pool)",
        len(lso_global_ids),
        len(training_global_ids),
    )

    # ── 5b. Dataset composition charts + status inventory ────────────────
    # Uses the pre-build snapshot so dataset_map being empty doesn't matter.
    status.phase('Generating dataset composition summary')
    status.log_dataset_inventory(
        dataset_map=_inventory_snapshot,
        subject_id_map=subject_id_map,
        lso_global_ids=lso_global_ids,
    )
    exp_logger.log_dataset_composition(
        dataset_map=_inventory_snapshot,
        subject_id_map=subject_id_map,
        lso_global_ids=lso_global_ids,
        training_global_ids=training_global_ids,
    )
    del _inventory_snapshot  # no longer needed; free the per-dataset tensor memory

    # ── 6. Estimate adapt sessions per subject for scope display ──────────
    adapt_sessions_per_subject = max(
        (
            len(joint_dataset.filter_by_subject(sid).get_session_ids()) - 1
            for sid in lso_global_ids
        ),
        default=cfg.adapt_epochs,
    )

    status.set_scope(
        datasets=cfg.dataset_ids,
        total_subjects=len(lso_global_ids),
        base_epochs=cfg.base_epochs,
        adapt_epochs=cfg.adapt_epochs,
        adapt_sessions_per_subject=adapt_sessions_per_subject,
        training_subjects=len(training_global_ids),
    )

    # ── 7. Train ONE shared base model on non-held-out subjects ───────────
    if training_global_ids:
        training_dataset = EEGSampleDataset(
            [s for s in joint_dataset.samples if s.subject_id in set(training_global_ids)]
        )
    else:
        # All subjects were held out — no training data available.
        # Fall back to using the pretrained backbone directly as the base
        # checkpoint without any fine-tuning.
        logger.warning(
            "No training subjects remain after holdout selection "
            "(lso_subjects_per_test_set=%d covers all subjects). "
            "Using pretrained backbone directly as base checkpoint — "
            "consider reducing lso_subjects_per_test_set.",
            cfg.lso_subjects_per_test_set,
        )
        training_dataset = None

    if training_dataset is not None and len(training_dataset) > 0:
        base_ckpt_path = train_base_model(
            training_dataset=training_dataset,
            channel_names=channel_names,
            n_classes=n_classes,
            cfg=cfg,
            model_factory=model_factory,
            checkpoints_dir=checkpoints_dir,
            status=status,
            accelerator=_accelerator,
            logs_dir=logs_dir,
            exp_logger=exp_logger,
        )
    else:
        # Use the pretrained backbone directly as the base checkpoint.
        base_ckpt_path = Path(cfg.base_model_path)
        logger.info("Using pretrained backbone directly as base checkpoint: %s", base_ckpt_path)

    # ── 8. Per-subject zero-shot + progressive adaptation ─────────────────
    all_subject_results: List[Dict[str, Any]] = []

    for fold_idx, held_out in enumerate(lso_global_ids, start=1):
        subject_label = subject_id_map.get(held_out, str(held_out))
        logger.info(
            "LSO fold %d/%d — held-out: %s (global id=%d)",
            fold_idx, len(lso_global_ids), subject_label, held_out,
        )

        # ── Resume: skip subjects whose result JSON already exists ────────
        _result_filename = subject_label.replace(':', '_').replace('/', '_') + '.json'
        _result_path = output_root / 'subject_results' / _result_filename
        if _result_path.exists():
            logger.info(
                "RESUME: skipping subject %s — result already saved at %s",
                subject_label, _result_path,
            )
            status.phase(
                f'RESUME: skipping fold {fold_idx}/{len(lso_global_ids)} '
                f'({subject_label}) — already complete'
            )
            # Still count this fold so ETA and summary counts stay correct.
            status.folds_completed += 1
            continue
        # ─────────────────────────────────────────────────────────────────

        status.subject_start(
            subject_id=held_out,
            dataset_id=subject_label,
            fold_idx=fold_idx,
            total_folds=len(lso_global_ids),
        )

        subject_result = run_subject_fold(
            held_out_global_id=held_out,
            joint_dataset=joint_dataset,
            base_ckpt_path=base_ckpt_path,
            channel_names=channel_names,
            n_classes=n_classes,
            cfg=cfg,
            model_factory=model_factory,
            checkpoints_dir=checkpoints_dir,
            logs_dir=logs_dir,
            status=status,
            subject_label=subject_label,
            fold_idx=fold_idx,
            total_folds=len(lso_global_ids),
            accelerator=_accelerator,
        )

        status.subject_end(
            subject_id=held_out,
            zero_shot=subject_result['zero_shot_metrics'],
            best_adapted=subject_result['best_adapted_metrics'],
        )

        all_subject_results.append(subject_result)
        exp_logger.log_subject_result(_to_subject_result(subject_result))

        # Save per-subject JSON immediately so progress is preserved on crash
        _save_subject_result(subject_result, output_root)

    # ── 9. Analytics & summary ────────────────────────────────────────────
    status.phase('Generating analytics and saving results')
    _wrapped = [_to_subject_result(r) for r in all_subject_results]
    exp_logger.save_summary_csv(_wrapped)
    exp_logger.generate_adaptation_plots(_wrapped, cfg.analytics_metrics)

    import numpy as np

    zs_vals = [
        r['zero_shot_metrics'].get('valid_balanced_accuracy', float('nan'))
        for r in all_subject_results
    ]
    best_vals = [
        r['best_adapted_metrics'].get('valid_balanced_accuracy', float('nan'))
        for r in all_subject_results
    ]
    zs_clean = [v for v in zs_vals if not np.isnan(v)]
    ad_clean = [v for v in best_vals if not np.isnan(v)]

    zs_mean = np.mean(zs_clean) if zs_clean else float('nan')
    zs_std  = np.std(zs_clean)  if zs_clean else float('nan')
    ad_mean = np.mean(ad_clean) if ad_clean else float('nan')
    ad_std  = np.std(ad_clean)  if ad_clean else float('nan')

    logger.info("Zero-shot balanced accuracy : %.3f ± %.3f", zs_mean, zs_std)
    logger.info("Best-adapted balanced accuracy: %.3f ± %.3f", ad_mean, ad_std)
    logger.info("Experiment complete. Results in: %s", output_root)

    status.finish({
        'zero_shot_balanced_accuracy':    f'{zs_mean:.4f} ± {zs_std:.4f}',
        'best_adapted_balanced_accuracy': f'{ad_mean:.4f} ± {ad_std:.4f}',
        'total_subjects_evaluated':       len(all_subject_results),
        'results_directory':              str(output_root),
    })


# ---------------------------------------------------------------------------
# Helper: convert the runner's result dict to the object shape analytics expects
# ---------------------------------------------------------------------------

def _to_subject_result(result: Dict[str, Any]) -> Any:
    """
    Convert a plain dict returned by run_subject_fold into a lightweight
    object whose attributes match what ExperimentLogger / AdaptationCurveAnalyser
    expect:

      result.subject_id
      result.stage_results  — list of objects with:
          .stage
          .session_id
          .metrics            (dict)
          .n_adaptation_sessions  (always 1 per stage; analytics uses it for CSV)
          .checkpoint_path    (None — we don't save per-stage checkpoints)
    """
    import types

    def _stage_obj(d: Dict[str, Any]) -> Any:
        return types.SimpleNamespace(
            stage=d['stage'],
            session_id=d.get('session_id'),
            metrics=d.get('metrics', {}),
            n_adaptation_sessions=1,   # each stage adapts on one session
            checkpoint_path=None,
        )

    return types.SimpleNamespace(
        subject_id=result.get('subject_id', result.get('global_subject_id')),
        subject_label=result.get('subject_label', ''),
        stage_results=[_stage_obj(s) for s in result.get('stage_results', [])],
        zero_shot_metrics=result.get('zero_shot_metrics', {}),
        best_adapted_metrics=result.get('best_adapted_metrics', {}),
        n_adapt_sessions=result.get('n_adapt_sessions', 0),
        n_test_trials=result.get('n_test_trials', 0),
    )


# ---------------------------------------------------------------------------
# Helper: save one subject result to disk as JSON
# ---------------------------------------------------------------------------

def _save_subject_result(result: Dict[str, Any], output_root: Path) -> None:
    """
    Write per-subject results to a JSON file so progress is durable
    across crashes.  One file per subject in output_root/subject_results/.
    """
    import json

    results_dir = output_root / 'subject_results'
    results_dir.mkdir(parents=True, exist_ok=True)

    label = result['subject_label'].replace(':', '_').replace('/', '_')
    out_path = results_dir / f"{label}.json"

    # Metrics dicts may contain non-serialisable types (numpy floats etc.)
    def _convert(obj):
        if hasattr(obj, 'item'):   # numpy scalar
            return obj.item()
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    serialisable = _convert(result)

    try:
        out_path.write_text(json.dumps(serialisable, indent=2), encoding='utf-8')
    except Exception as exc:
        logger.warning("Could not save subject result to %s: %s", out_path, exc)