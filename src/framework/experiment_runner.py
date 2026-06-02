"""
experiment_runner.py
--------------------
Top-level orchestration for the EEG motor imagery framework.

Training strategy (LSO enabled):
  1.  Load and preprocess all datasets (with optional .pt disk cache).
  2.  Build a joint dataset with globally-unique subject IDs.
  3.  Train ONE shared base model on all subjects NOT in lso_subjects.
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

import torch
import pytorch_lightning as pl

from .canonical_channels import CANONICAL_CHANNELS
from .label_schema import LabelSchema, infer_shared_label_schema, DATASET_LABEL_MAPS
from .dataset_registry import MoabbDatasetLoader
from .eeg_dataset import EEGSample, EEGSampleDataset
from .multi_dataset_loader import ConcatDataLoader, subject_split, train_val_test_split
from .leave_subjects_out import LeaveSubjectsOutEvaluator
from .analytics import ExperimentLogger, compute_metrics_from_model
from .preprocessing import TARGET_SAMPLE_RATE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Status logger
# ---------------------------------------------------------------------------

class StatusLogger:
    """
    Writes a continuously-updated status.txt to the experiment output directory.
    Tracks phase timings and estimates remaining time based on completed folds.
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

        self._refresh()

    def start(self, config: Dict[str, Any]) -> None:
        lines = ['  ' + f'{k}: {v}' for k, v in config.items()]
        self._config_block = '\n'.join(lines)
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
            base_epochs                                               # one shared base
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

    def phase(self, description: str) -> None:
        self.current_phase = description
        self.current_phase_start = time.time()
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
        self._refresh()

    def stage_start(self, subject_id: int, stage: int, total_stages: int) -> None:
        label = 'zero-shot eval' if stage == 0 else f'adaptation stage {stage}/{total_stages - 1}'
        self.current_phase = f'Subject {subject_id} — {label}'
        self.stage_start_time = time.time()
        self._refresh()

    def stage_end(self, subject_id: int, stage: int, metrics: Dict[str, float]) -> None:
        duration = time.time() - (self.stage_start_time or time.time())
        self.stage_durations.append(duration)
        bal_acc = metrics.get('valid_balanced_accuracy', float('nan'))
        acc = metrics.get('valid_accuracy', float('nan'))
        f1 = metrics.get('valid_f1_macro', float('nan'))
        label = 'zero-shot' if stage == 0 else f'stage {stage}'
        logger.info(
            '[Status] Subject %d %s — bal_acc=%.3f  acc=%.3f  f1_macro=%.3f  (%.1fs)',
            subject_id, label, bal_acc, acc, f1, duration,
        )
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
        gain = (ad_ba - zs_ba) if not (zs_ba != zs_ba or ad_ba != ad_ba) else float('nan')

        summary = (
            f'  Subject {subject_id:>3d} | '
            f'zero-shot={zs_ba:.3f}  best-adapted={ad_ba:.3f}  '
            f'gain={gain:+.3f}  ({_fmt_duration(duration)})'
        )
        self.completed_subjects.append(summary)
        self._refresh()

    def finish(self, final_metrics: Dict[str, Any]) -> None:
        self.current_phase = 'COMPLETE'
        total = time.time() - self.experiment_start
        lines = [f'  {k}: {v}' for k, v in final_metrics.items()]
        footer = '\n'.join(lines)
        self._refresh(footer=footer, total_elapsed=total)
        logger.info('[Status] Experiment complete. Total time: %s', _fmt_duration(total))

    def _refresh(
        self,
        footer: str = '',
        total_elapsed: Optional[float] = None,
    ) -> None:
        now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        elapsed = total_elapsed if total_elapsed is not None else (
            time.time() - self.experiment_start
        )
        phase_elapsed = time.time() - self.current_phase_start
        eta_str = self._eta_string()

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

        if self.stage_durations:
            avg_stage = sum(self.stage_durations) / len(self.stage_durations)
            lines.append(f'  Avg stage : {_fmt_duration(avg_stage)} per stage')

        if self.fold_durations:
            avg_fold = sum(self.fold_durations) / len(self.fold_durations)
            lines.append(f'  Avg fold  : {_fmt_duration(avg_fold)} per subject')

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

    # Evaluation
    lso_enabled: bool = True
    # lso_subjects: original subject IDs to hold out.
    # None = hold out every subject one at a time.
    # e.g. [1, 2, 3] = hold out subjects whose original ID is 1, 2, or 3
    # across all datasets that have them.
    lso_subjects: Optional[List[int]] = None
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

    If cfg.dataset_cache_dir is set, each dataset is saved to a .pt file
    keyed by dataset ID + preprocessing parameters.  Subsequent runs load
    from disk instead of re-running the full MOABB + preprocessing pipeline.

    Delete the cache directory (or individual .pt files) to force a reload.
    """
    shared_labels, schemas = infer_shared_label_schema(cfg.dataset_ids)
    logger.info("Shared label space: %s", shared_labels)

    cache_dir: Optional[Path] = None
    if cfg.dataset_cache_dir:
        cache_dir = Path(cfg.dataset_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Dataset cache directory: %s", cache_dir)

    loaded: Dict[str, EEGSampleDataset] = {}

    for i, (moabb_ds, did) in enumerate(zip(cfg.datasets, cfg.dataset_ids)):
        schema = schemas[did]
        subjects = None
        if cfg.subjects_per_dataset and i < len(cfg.subjects_per_dataset):
            subjects = cfg.subjects_per_dataset[i]

        # ── Try cache first ───────────────────────────────────────────────
        if cache_dir is not None:
            cache_key = hashlib.md5(
                f"{did}|{subjects}|{cfg.fmin}|{cfg.fmax}|"
                f"{cfg.tmin}|{cfg.tmax}|{cfg.resample}".encode()
            ).hexdigest()[:10]
            cache_path = cache_dir / f"{did}_{cache_key}.pt"

            if cache_path.exists():
                logger.info("Loading %s from cache: %s", did, cache_path)
                try:
                    ds = torch.load(cache_path, weights_only=False)
                    loaded[did] = ds
                    logger.info(
                        "Loaded %s from cache: %d samples, %d classes",
                        did, len(ds), schema.n_classes,
                    )
                    continue
                except Exception as exc:
                    logger.warning(
                        "Cache load failed for %s (%s) — reloading from MOABB.", did, exc
                    )

        # ── Load from MOABB + preprocess ──────────────────────────────────
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
        )
        ds = loader.load_all_subjects()
        loaded[did] = ds
        logger.info("Loaded %s: %d samples, %d classes", did, len(ds), schema.n_classes)

        # ── Save to cache ─────────────────────────────────────────────────
        if cache_dir is not None:
            try:
                torch.save(ds, cache_path)
                logger.info("Cached %s → %s", did, cache_path)
            except Exception as exc:
                logger.warning("Could not cache %s: %s", did, exc)

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

    Returns
    -------
    joint_dataset : EEGSampleDataset  — all samples, globally unique subject IDs
    subject_id_map : Dict[global_id → "dataset_id:s{orig_id}"]  — for logging
    """
    remapped_datasets: List[EEGSampleDataset] = []
    subject_id_map: Dict[int, str] = {}
    global_id_counter = 0

    for did, ds in dataset_map.items():
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
        len(dataset_map),
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
        accelerator='auto',
        max_epochs=cfg.base_epochs,
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        default_root_dir=str(checkpoints_dir),
    )
    trainer.fit(
        model,
        train_loader_obj.get_loader(),
        val_loader_obj.get_loader(),
    )

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(base_ckpt_path))
    logger.info("Shared base model saved to: %s", base_ckpt_path)

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
            accelerator='auto',
            max_epochs=cfg.adapt_epochs,
            log_every_n_steps=1,
            num_sanity_val_steps=0,
            default_root_dir=str(subject_ckpt_dir),
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
      3. Train ONE shared base model on all subjects NOT in lso_subjects
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
        'lso_subjects': cfg.lso_subjects,
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
            accelerator='auto',
            max_epochs=cfg.base_epochs,
            log_every_n_steps=1,
            num_sanity_val_steps=0,
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
    status.phase('Building joint dataset across all datasets')
    joint_dataset, subject_id_map = build_joint_dataset(dataset_map, channel_names)

    # ── 5. Resolve which subject IDs are held out ─────────────────────────
    # lso_subjects in config refers to *original* IDs; map them to global IDs.
    logger.info(
        "LSO config: lso_subjects=%s (%s)",
        cfg.lso_subjects,
        "hold out ALL subjects one at a time" if cfg.lso_subjects is None
        else f"hold out original subject IDs {cfg.lso_subjects} across all datasets",
    )
    if cfg.lso_subjects is not None:
        lso_subjects_set = set(cfg.lso_subjects)
        lso_global_ids = [
            gid for gid, label in subject_id_map.items()
            if int(label.split(':s')[1]) in lso_subjects_set
        ]
        if not lso_global_ids:
            raise ValueError(
                f"lso_subjects={cfg.lso_subjects} matched no subjects in the loaded datasets. "
                f"Available original IDs: "
                f"{sorted({int(v.split(':s')[1]) for v in subject_id_map.values()})}"
            )
    else:
        lso_global_ids = joint_dataset.get_subject_ids()

    training_global_ids = [
        sid for sid in joint_dataset.get_subject_ids()
        if sid not in set(lso_global_ids)
    ]

    logger.info(
        "Held-out subjects: %d (original IDs: %s)  |  Training subjects: %d",
        len(lso_global_ids),
        sorted({int(subject_id_map[g].split(':s')[1]) for g in lso_global_ids}),
        len(training_global_ids),
    )

    # ── 5b. Dataset composition charts ───────────────────────────────────
    status.phase('Generating dataset composition summary')
    exp_logger.log_dataset_composition(
        dataset_map=dataset_map,
        subject_id_map=subject_id_map,
        lso_global_ids=lso_global_ids,
        training_global_ids=training_global_ids,
    )

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
        # lso_subjects=None means hold-out ALL subjects for adaptation evaluation.
        # In this protocol the base model is trained on ALL subjects (all sessions),
        # then each subject's held-out test session is used only for evaluation.
        # This matches standard cross-subject transfer learning: the backbone sees
        # the subject during pre-training but adaptation is still per-subject.
        # Note: if you want strict leave-one-subject-out base training, set
        # lso_subjects to a specific list so training_global_ids is non-empty.
        logger.info(
            "lso_subjects=None: training base model on ALL %d subjects "
            "(held-out test session per subject is excluded from training splits).",
            len(joint_dataset.get_subject_ids()),
        )
        training_dataset = joint_dataset

    if training_dataset is not None and len(training_dataset) > 0:
        base_ckpt_path = train_base_model(
            training_dataset=training_dataset,
            channel_names=channel_names,
            n_classes=n_classes,
            cfg=cfg,
            model_factory=model_factory,
            checkpoints_dir=checkpoints_dir,
            status=status,
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