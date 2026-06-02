"""
leave_subjects_out.py
---------------------
Leave-Subjects-Out (LSO) evaluation protocol with progressive
session-based fine-tuning.

For each held-out subject S:

  Stage 0  — Zero-shot: train on all other subjects, evaluate on S_test
  Stage 1  — Adapt with 1 non-test session from S, evaluate on S_test
  Stage 2  — Adapt with 2 non-test sessions from S, evaluate on S_test
  ...
  Stage K  — Adapt with all (n-1) non-test sessions, evaluate on S_test

The test session is fixed throughout — it is always the same held-out
session so comparisons across stages are meaningful.

Adaptation continues from the previous stage's model state (warm start),
not from the cross-subject baseline each time.
"""

from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import math
import json

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning import loggers as pl_loggers

from .eeg_dataset import EEGSampleDataset, collate_same_channels
from .multi_dataset_loader import ConcatDataLoader, make_balanced_sampler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    subject_id: int
    stage: int                      # 0 = zero-shot, n = n adaptation sessions
    n_adaptation_sessions: int
    metrics: Dict[str, float] = field(default_factory=dict)
    checkpoint_path: Optional[str] = None


@dataclass
class SubjectResult:
    subject_id: int
    stage_results: List[StageResult] = field(default_factory=list)

    def zero_shot_metrics(self) -> Dict[str, float]:
        for r in self.stage_results:
            if r.stage == 0:
                return r.metrics
        return {}

    def best_adapted_metrics(self) -> Dict[str, float]:
        adapted = [r for r in self.stage_results if r.stage > 0]
        if not adapted:
            return {}
        return max(adapted, key=lambda r: r.metrics.get('valid_balanced_accuracy', 0)).metrics


# ---------------------------------------------------------------------------
# Protocol implementation
# ---------------------------------------------------------------------------

class LeaveSubjectsOutEvaluator:
    """
    Orchestrates the full LSO + progressive adaptation experiment.

    Parameters
    ----------
    dataset : EEGSampleDataset
        Full dataset with all subjects and sessions loaded.
    model_factory : callable
        Called as model_factory(channel_names, n_classes) → pl.LightningModule.
        Must return a fresh or re-initialised model for each LSO fold.
    channel_names : List[str]
        Canonical channel names for this dataset.
    n_classes : int
    max_lr : float
    base_epochs : int — epochs for the cross-subject training stage
    adapt_epochs : int — epochs per adaptation stage
    batch_size : int
    checkpoints_dir : Path
    logs_dir : Path
    test_session_idx : int
        Index (0-based, by sorted session_id) of the held-out test session.
        Default 0 = first recorded session.
    subjects : Optional[List[int]]
        Subset of subject ids to evaluate. None = all subjects.
    """

    def __init__(
        self,
        dataset: EEGSampleDataset,
        model_factory,
        channel_names: List[str],
        n_classes: int,
        max_lr: float = 4e-4,
        base_epochs: int = 20,
        adapt_epochs: int = 10,
        batch_size: int = 32,
        checkpoints_dir: Path = Path('checkpoints'),
        logs_dir: Path = Path('logs'),
        test_session_idx: int = 0,
        subjects: Optional[List[int]] = None,
    ) -> None:
        self.dataset = dataset
        self.model_factory = model_factory
        self.channel_names = channel_names
        self.n_classes = n_classes
        self.max_lr = max_lr
        self.base_epochs = base_epochs
        self.adapt_epochs = adapt_epochs
        self.batch_size = batch_size
        self.checkpoints_dir = Path(checkpoints_dir)
        self.logs_dir = Path(logs_dir)
        self.test_session_idx = test_session_idx

        all_subjects = dataset.get_subject_ids()
        self.subjects = subjects if subjects is not None else all_subjects

        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def run(self) -> List[SubjectResult]:
        """Execute the full LSO protocol and return per-subject results."""
        all_results: List[SubjectResult] = []

        for held_out in self.subjects:
            logger.info("=" * 60)
            logger.info("LSO fold: held-out subject = %d", held_out)
            subject_result = self._run_fold(held_out)
            all_results.append(subject_result)
            self._save_subject_result(subject_result)

        self._save_summary(all_results)
        return all_results

    # ------------------------------------------------------------------
    def _run_fold(self, held_out_subject: int) -> SubjectResult:
        subject_result = SubjectResult(subject_id=held_out_subject)

        # Split dataset
        train_pool = self.dataset.exclude_subject(held_out_subject)
        subject_data = self.dataset.filter_by_subject(held_out_subject)

        # Determine fixed test session
        session_ids = subject_data.get_session_ids()
        if not session_ids:
            logger.warning("Subject %d has no data — skipping", held_out_subject)
            return subject_result

        test_session = session_ids[self.test_session_idx % len(session_ids)]
        adapt_sessions = [s for s in session_ids if s != test_session]

        test_set = subject_data.filter_by_session(test_session)
        logger.info(
            "Subject %d: test_session=%d, adapt_sessions=%s, test_samples=%d",
            held_out_subject, test_session, adapt_sessions, len(test_set),
        )

        # ── Stage 0: cross-subject training ──────────────────────────
        model = self.model_factory(self.channel_names, self.n_classes)

        cross_subject_ckpt = (
            self.checkpoints_dir / f"lso_subject{held_out_subject}_cross_subject.ckpt"
        )
        self._train(
            model=model,
            train_set=train_pool,
            val_set=test_set,
            epochs=self.base_epochs,
            run_name=f"s{held_out_subject}_stage0",
            ckpt_path=cross_subject_ckpt,
        )
        metrics_0 = self._evaluate(model, test_set)
        subject_result.stage_results.append(StageResult(
            subject_id=held_out_subject,
            stage=0,
            n_adaptation_sessions=0,
            metrics=metrics_0,
            checkpoint_path=str(cross_subject_ckpt),
        ))
        logger.info("Stage 0 metrics: %s", metrics_0)

        # ── Stages 1..K: progressive adaptation ─────────────────────
        current_model = model  # warm-start from cross-subject model

        for k, adapt_session in enumerate(adapt_sessions, start=1):
            # Accumulate sessions up to stage k
            adapt_set_sessions = adapt_sessions[:k]
            adapt_data_parts = [
                subject_data.filter_by_session(s) for s in adapt_set_sessions
            ]
            adapt_set = EEGSampleDataset.concat(adapt_data_parts)

            adapt_ckpt = (
                self.checkpoints_dir /
                f"lso_subject{held_out_subject}_stage{k}.ckpt"
            )
            self._train(
                model=current_model,
                train_set=adapt_set,
                val_set=test_set,
                epochs=self.adapt_epochs,
                run_name=f"s{held_out_subject}_stage{k}",
                ckpt_path=adapt_ckpt,
            )
            metrics_k = self._evaluate(current_model, test_set)
            subject_result.stage_results.append(StageResult(
                subject_id=held_out_subject,
                stage=k,
                n_adaptation_sessions=k,
                metrics=metrics_k,
                checkpoint_path=str(adapt_ckpt),
            ))
            logger.info("Stage %d (%d sessions) metrics: %s", k, k, metrics_k)

        return subject_result

    # ------------------------------------------------------------------
    def _make_loader(
        self,
        dataset: EEGSampleDataset,
        shuffle: bool,
    ) -> DataLoader:
        sampler = make_balanced_sampler(dataset) if shuffle else None
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(shuffle and sampler is None),
            sampler=sampler,
            num_workers=0,
            collate_fn=collate_same_channels,
        )

    def _train(
        self,
        model: pl.LightningModule,
        train_set: EEGSampleDataset,
        val_set: EEGSampleDataset,
        epochs: int,
        run_name: str,
        ckpt_path: Path,
    ) -> None:
        if len(train_set) == 0:
            logger.warning("Empty training set for run '%s' — skipping", run_name)
            return

        train_loader = self._make_loader(train_set, shuffle=True)
        val_loader = self._make_loader(val_set, shuffle=False)

        steps_per_epoch = math.ceil(len(train_set) / self.batch_size)

        # Update scheduler steps in model if it supports it
        if hasattr(model, 'steps_per_epoch'):
            model.steps_per_epoch = steps_per_epoch
        if hasattr(model, 'max_epochs'):
            model.max_epochs = epochs

        save_cb = pl.callbacks.ModelCheckpoint(
            dirpath=str(self.checkpoints_dir),
            filename=run_name + '-last',
            save_top_k=0,
            save_last=True,
        )
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

        trainer = pl.Trainer(
            accelerator='auto',
            max_epochs=epochs,
            log_every_n_steps=max(1, steps_per_epoch // 5),
            num_sanity_val_steps=0,
            enable_checkpointing=True,
            enable_progress_bar=True,
            callbacks=[lr_monitor, save_cb],
            logger=[
                pl_loggers.CSVLogger(
                    str(self.logs_dir),
                    name=run_name,
                )
            ],
        )
        trainer.fit(model, train_loader, val_loader)

        # Persist checkpoint under the desired name
        last = self.checkpoints_dir / (run_name + '-last.ckpt')
        if last.exists():
            last.rename(ckpt_path)

    def _evaluate(
        self,
        model: pl.LightningModule,
        test_set: EEGSampleDataset,
    ) -> Dict[str, float]:
        from .analytics import compute_metrics_from_model

        loader = self._make_loader(test_set, shuffle=False)
        return compute_metrics_from_model(model, loader)

    # ------------------------------------------------------------------
    def _save_subject_result(self, result: SubjectResult) -> None:
        path = self.logs_dir / f"subject_{result.subject_id}_results.json"
        serialisable = {
            'subject_id': result.subject_id,
            'stages': [
                {
                    'stage': r.stage,
                    'n_adaptation_sessions': r.n_adaptation_sessions,
                    'metrics': r.metrics,
                    'checkpoint_path': r.checkpoint_path,
                }
                for r in result.stage_results
            ],
        }
        with open(path, 'w') as f:
            json.dump(serialisable, f, indent=2)

    def _save_summary(self, all_results: List[SubjectResult]) -> None:
        path = self.logs_dir / 'lso_summary.json'
        summary = []
        for sr in all_results:
            summary.append({
                'subject_id': sr.subject_id,
                'zero_shot': sr.zero_shot_metrics(),
                'best_adapted': sr.best_adapted_metrics(),
                'n_stages': len(sr.stage_results),
            })
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info("LSO summary written to %s", path)
