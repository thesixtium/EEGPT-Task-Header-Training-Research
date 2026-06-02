"""
analytics.py
------------
Comprehensive analytics for the EEG motor imagery framework.

Provides:
  - compute_metrics_from_model  — run inference and collect all metrics
  - MetricsComputer             — batch-level metric computation
  - ConfusionMatrixAnalyser     — per-subject confusion matrices
  - AdaptationCurveAnalyser     — zero-shot vs adapted performance plots
  - DatasetCompositionAnalyser  — dataset breakdown charts (samples, subjects,
                                   sessions, class distribution, train/held-out split)
  - ExperimentLogger            — structured JSON + CSV experiment storage
  - plot_adaptation_curves      — matplotlib visualisation (optional)
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import logging
import math

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# Optional matplotlib — analytics will skip plots if unavailable
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available — plotting functions disabled")

# sklearn metrics
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    cohen_kappa_score,
)


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics_from_model(
    model: pl.LightningModule,
    loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Run inference with model on loader and return a dict of evaluation metrics.

    Metrics computed:
      accuracy, balanced_accuracy, f1_weighted, f1_macro, f1_micro,
      mcc (Matthews), cohen_kappa
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (tuple, list)):
                x, y = batch[0], batch[1]
            else:
                x, y = batch['eeg'], batch['label']

            x = x.to(device)
            _, logit = model(x)
            preds = logit.argmax(dim=-1).cpu().tolist()
            labels = y.cpu().tolist() if isinstance(y, torch.Tensor) else y

            all_preds.extend(preds)
            all_labels.extend(labels)

    return _compute_sklearn_metrics(np.array(all_labels), np.array(all_preds))


def _compute_sklearn_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute all standard classification metrics."""
    metrics: Dict[str, float] = {}

    metrics['valid_accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['valid_balanced_accuracy'] = float(balanced_accuracy_score(y_true, y_pred))
    metrics['valid_f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    metrics['valid_f1_macro'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['valid_f1_micro'] = float(f1_score(y_true, y_pred, average='micro', zero_division=0))
    metrics['valid_mcc'] = float(matthews_corrcoef(y_true, y_pred))
    metrics['valid_cohen_kappa'] = float(cohen_kappa_score(y_true, y_pred))

    return metrics


# ---------------------------------------------------------------------------
# Confusion matrix analysis
# ---------------------------------------------------------------------------

class ConfusionMatrixAnalyser:
    """
    Collect predictions over a dataset and produce normalised confusion matrices.
    """

    def __init__(self, class_names: List[str]) -> None:
        self.class_names = class_names
        self.n_classes = len(class_names)

    def compute(
        self,
        model: pl.LightningModule,
        loader: DataLoader,
        device: Optional[torch.device] = None,
    ) -> np.ndarray:
        """Return normalised confusion matrix [n_classes, n_classes]."""
        if device is None:
            device = next(model.parameters()).device

        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in loader:
                x, y = (batch[0], batch[1]) if isinstance(batch, (tuple, list)) else (
                    batch['eeg'], batch['label']
                )
                _, logit = model(x.to(device))
                all_preds.extend(logit.argmax(-1).cpu().tolist())
                all_labels.extend(
                    y.cpu().tolist() if isinstance(y, torch.Tensor) else y
                )

        cm = confusion_matrix(all_labels, all_preds, labels=list(range(self.n_classes)))
        cm_normalised = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        return cm_normalised

    def plot(
        self,
        cm: np.ndarray,
        title: str = 'Confusion Matrix',
        save_path: Optional[Path] = None,
    ) -> None:
        if not _MATPLOTLIB_AVAILABLE:
            return

        fig, ax = plt.subplots(figsize=(max(4, self.n_classes), max(4, self.n_classes)))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues', vmin=0, vmax=1)
        fig.colorbar(im, ax=ax)
        ax.set(
            xticks=range(self.n_classes),
            yticks=range(self.n_classes),
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            xlabel='Predicted',
            ylabel='True',
            title=title,
        )
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

        for i in range(self.n_classes):
            for j in range(self.n_classes):
                ax.text(j, i, f'{cm[i, j]:.2f}', ha='center', va='center',
                        color='white' if cm[i, j] > 0.5 else 'black')

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info("Confusion matrix saved to %s", save_path)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Adaptation curve analysis
# ---------------------------------------------------------------------------

class AdaptationCurveAnalyser:
    """
    Summarise and plot zero-shot vs adapted performance across subjects.
    """

    @staticmethod
    def summarise(
        all_subject_results: List[Any],  # List[SubjectResult]
        metric: str = 'valid_balanced_accuracy',
    ) -> Dict[str, Any]:
        """
        Build summary statistics across subjects per stage.

        Returns a dict:
        {
          'stages': [0, 1, 2, ...],
          'mean':   [float, ...],
          'std':    [float, ...],
          'per_subject': { subject_id: [float per stage] }
        }
        """
        # Find max number of stages
        max_stages = max(len(sr.stage_results) for sr in all_subject_results)

        per_subject: Dict[int, List[float]] = {}
        for sr in all_subject_results:
            vals = []
            for stage_idx in range(max_stages):
                matching = [r for r in sr.stage_results if r.stage == stage_idx]
                vals.append(matching[0].metrics.get(metric, float('nan')) if matching else float('nan'))
            per_subject[sr.subject_id] = vals

        stages = list(range(max_stages))
        means, stds = [], []
        for si in stages:
            vals = [per_subject[s][si] for s in per_subject if not math.isnan(per_subject[s][si])]
            means.append(float(np.mean(vals)) if vals else float('nan'))
            stds.append(float(np.std(vals)) if vals else float('nan'))

        return {'stages': stages, 'mean': means, 'std': stds, 'per_subject': per_subject}

    @staticmethod
    def plot(
        summary: Dict[str, Any],
        metric_label: str = 'Balanced Accuracy',
        title: str = 'Progressive Adaptation Curve',
        save_path: Optional[Path] = None,
    ) -> None:
        if not _MATPLOTLIB_AVAILABLE:
            return

        stages = summary['stages']
        means = summary['mean']
        stds = summary['std']
        per_subject = summary['per_subject']

        fig, ax = plt.subplots(figsize=(8, 5))

        # Per-subject thin lines
        for sid, vals in per_subject.items():
            ax.plot(stages, vals, color='steelblue', alpha=0.25, linewidth=1,
                    marker='o', markersize=3)

        # Mean ± std
        means_arr = np.array(means)
        stds_arr = np.array(stds)
        ax.fill_between(stages, means_arr - stds_arr, means_arr + stds_arr,
                        alpha=0.2, color='steelblue')
        ax.plot(stages, means_arr, color='steelblue', linewidth=2.5,
                marker='o', markersize=6, label='Mean ± std')

        ax.axvline(0, color='red', linestyle='--', linewidth=1, label='Zero-shot')
        ax.set(
            xlabel='Adaptation stage (# sessions)',
            ylabel=metric_label,
            title=title,
            xticks=stages,
        )
        ax.legend()
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info("Adaptation curve saved to %s", save_path)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Dataset composition analysis
# ---------------------------------------------------------------------------

class DatasetCompositionAnalyser:
    """
    Analyse and visualise dataset composition across multiple EEGSampleDatasets.

    Given a ``dataset_map`` (``{dataset_id: EEGSampleDataset}``) and an
    optional mapping of global subject IDs to their role (training vs held-out),
    this class produces:

      - A JSON summary written to ``<output_dir>/dataset_composition.json``
      - Up to five matplotlib charts saved alongside it:

        1. ``samples_per_dataset.png``      — total sample count per dataset
        2. ``subjects_per_dataset.png``     — unique subject count per dataset
        3. ``sessions_per_dataset.png``     — mean/min/max sessions per subject
        4. ``class_distribution.png``       — per-dataset class label histogram
        5. ``train_held_out_split.png``     — training vs held-out subject split

    All chart generation is silently skipped when matplotlib is unavailable.

    Usage
    -----
    ::

        analyser = DatasetCompositionAnalyser()
        summary = analyser.compute(dataset_map)
        analyser.save_json(summary, output_dir / "dataset_composition.json")
        analyser.plot_all(summary, output_dir / "dataset_composition")
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        dataset_map: Dict[str, Any],          # {dataset_id: EEGSampleDataset}
        subject_id_map: Optional[Dict[int, str]] = None,
        lso_global_ids: Optional[List[int]] = None,
        training_global_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Build a comprehensive composition summary.

        Parameters
        ----------
        dataset_map:
            Mapping from dataset ID string to an ``EEGSampleDataset`` (or any
            object with a ``.samples`` iterable of objects that have
            ``.label``, ``.subject_id``, and ``.session_id`` attributes).
        subject_id_map:
            Optional ``{global_id: "DATASET_ID:sN"}`` as produced by
            ``build_joint_dataset``.  Used to annotate train/held-out breakdown.
        lso_global_ids:
            Global subject IDs that are held out.
        training_global_ids:
            Global subject IDs used for base-model training.

        Returns
        -------
        dict with keys:
          ``n_datasets``, ``total_samples``, ``total_subjects``,
          ``per_dataset`` (list of per-dataset dicts),
          ``train_held_out`` (split summary, present only when IDs are supplied).
        """
        summary: Dict[str, Any] = {
            'n_datasets': len(dataset_map),
            'total_samples': 0,
            'total_subjects': 0,
            'per_dataset': [],
        }

        for did, ds in dataset_map.items():
            samples = ds.samples  # list of EEGSample-like objects

            # ── Per-sample counts ─────────────────────────────────────
            subject_ids = sorted({s.subject_id for s in samples})
            n_subjects = len(subject_ids)
            n_samples = len(samples)

            # Sessions per subject
            sessions_by_subject: Dict[int, set] = {}
            for s in samples:
                sessions_by_subject.setdefault(s.subject_id, set()).add(s.session_id)
            sessions_counts = [len(v) for v in sessions_by_subject.values()]
            n_sessions_total = sum(sessions_counts)
            mean_sessions = float(np.mean(sessions_counts)) if sessions_counts else 0.0
            min_sessions = int(min(sessions_counts)) if sessions_counts else 0
            max_sessions = int(max(sessions_counts)) if sessions_counts else 0

            # Steps (samples) per session
            steps_by_session: Dict[Any, int] = {}
            for s in samples:
                key = (s.subject_id, s.session_id)
                steps_by_session[key] = steps_by_session.get(key, 0) + 1
            steps_per_session_vals = list(steps_by_session.values())
            mean_steps = float(np.mean(steps_per_session_vals)) if steps_per_session_vals else 0.0
            min_steps = int(min(steps_per_session_vals)) if steps_per_session_vals else 0
            max_steps = int(max(steps_per_session_vals)) if steps_per_session_vals else 0

            # Class distribution
            label_counts: Dict[int, int] = {}
            for s in samples:
                label_counts[int(s.label)] = label_counts.get(int(s.label), 0) + 1

            ds_entry: Dict[str, Any] = {
                'dataset_id': did,
                'n_samples': n_samples,
                'n_subjects': n_subjects,
                'n_sessions_total': n_sessions_total,
                'sessions_per_subject': {
                    'mean': round(mean_sessions, 2),
                    'min': min_sessions,
                    'max': max_sessions,
                },
                'steps_per_session': {
                    'mean': round(mean_steps, 2),
                    'min': min_steps,
                    'max': max_steps,
                },
                'class_distribution': label_counts,
            }
            summary['per_dataset'].append(ds_entry)
            summary['total_samples'] += n_samples
            summary['total_subjects'] += n_subjects

        # ── Train / held-out breakdown ────────────────────────────────
        if subject_id_map and lso_global_ids is not None and training_global_ids is not None:
            summary['train_held_out'] = self._split_summary(
                subject_id_map, lso_global_ids, training_global_ids,
                summary['per_dataset'],
            )

        return summary

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save_json(summary: Dict[str, Any], path: Path) -> None:
        """Write the composition summary dict to ``path`` as indented JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Dataset composition JSON saved to %s", path)

    # ------------------------------------------------------------------
    # Charts
    # ------------------------------------------------------------------

    def plot_all(
        self,
        summary: Dict[str, Any],
        output_dir: Path,
    ) -> None:
        """
        Render all available composition charts into *output_dir*.

        Silently returns if matplotlib is unavailable.
        """
        if not _MATPLOTLIB_AVAILABLE:
            logger.warning("matplotlib not available — dataset composition plots skipped")
            return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._plot_samples_per_dataset(summary, output_dir / 'samples_per_dataset.png')
        self._plot_subjects_per_dataset(summary, output_dir / 'subjects_per_dataset.png')
        self._plot_sessions_per_dataset(summary, output_dir / 'sessions_per_dataset.png')
        self._plot_class_distribution(summary, output_dir / 'class_distribution.png')
        self._plot_steps_per_session(summary, output_dir / 'steps_per_session.png')
        if 'train_held_out' in summary:
            self._plot_train_held_out_split(
                summary['train_held_out'],
                output_dir / 'train_held_out_split.png',
            )

    # ------------------------------------------------------------------
    # Internal helpers — split summary
    # ------------------------------------------------------------------

    @staticmethod
    def _split_summary(
        subject_id_map: Dict[int, str],
        lso_global_ids: List[int],
        training_global_ids: List[int],
        per_dataset: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Build per-dataset train/held-out subject counts."""
        lso_set = set(lso_global_ids)
        train_set = set(training_global_ids)

        # Map dataset_id → {training: N, held_out: N}
        ds_split: Dict[str, Dict[str, int]] = {}
        for gid, label in subject_id_map.items():
            did = label.split(':s')[0]
            entry = ds_split.setdefault(did, {'training': 0, 'held_out': 0})
            if gid in lso_set:
                entry['held_out'] += 1
            elif gid in train_set:
                entry['training'] += 1

        return {
            'total_training': len(training_global_ids),
            'total_held_out': len(lso_global_ids),
            'per_dataset': ds_split,
        }

    # ------------------------------------------------------------------
    # Internal helpers — individual plots
    # ------------------------------------------------------------------

    @staticmethod
    def _bar_chart(
        ax,
        labels: List[str],
        values: List[float],
        color: str,
        ylabel: str,
        title: str,
        value_fmt: str = '{:.0f}',
    ) -> None:
        """Shared helper: simple horizontal bar chart with value annotations."""
        y_pos = range(len(labels))
        bars = ax.barh(list(y_pos), values, color=color, edgecolor='white', linewidth=0.6)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', labelsize=8)
        # Value labels at end of each bar
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_width() + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                value_fmt.format(val),
                va='center', ha='left', fontsize=8,
            )

    def _plot_samples_per_dataset(
        self, summary: Dict[str, Any], save_path: Path
    ) -> None:
        per_ds = summary['per_dataset']
        labels = [d['dataset_id'] for d in per_ds]
        values = [d['n_samples'] for d in per_ds]

        fig, ax = plt.subplots(figsize=(7, max(2.5, 0.6 * len(labels) + 1.5)))
        self._bar_chart(
            ax, labels, values,
            color='#4C72B0',
            ylabel='Samples (trials)',
            title=f'Samples per Dataset  (total = {sum(values):,})',
        )
        # Percentage annotation
        total = sum(values)
        for i, (bar, val) in enumerate(
            zip(ax.patches, values)
        ):
            pct = 100 * val / total if total else 0
            ax.text(
                bar.get_width() + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{val:,}  ({pct:.1f}%)',
                va='center', ha='left', fontsize=8,
            )
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info("Saved: %s", save_path)

    def _plot_subjects_per_dataset(
        self, summary: Dict[str, Any], save_path: Path
    ) -> None:
        per_ds = summary['per_dataset']
        labels = [d['dataset_id'] for d in per_ds]
        values = [d['n_subjects'] for d in per_ds]

        fig, ax = plt.subplots(figsize=(7, max(2.5, 0.6 * len(labels) + 1.5)))
        self._bar_chart(
            ax, labels, values,
            color='#55A868',
            ylabel='Unique subjects',
            title=f'Subjects per Dataset  (total = {sum(values)})',
        )
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info("Saved: %s", save_path)

    def _plot_sessions_per_dataset(
        self, summary: Dict[str, Any], save_path: Path
    ) -> None:
        """Grouped bar chart: mean sessions/subject with min/max error bars."""
        per_ds = summary['per_dataset']
        labels = [d['dataset_id'] for d in per_ds]
        means = [d['sessions_per_subject']['mean'] for d in per_ds]
        mins  = [d['sessions_per_subject']['min']  for d in per_ds]
        maxs  = [d['sessions_per_subject']['max']  for d in per_ds]

        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(max(5, 1.4 * len(labels) + 2), 4))
        bars = ax.bar(x, means, color='#C44E52', edgecolor='white', linewidth=0.6, zorder=3)
        # Error bars showing min–max range
        lower_err = [m - mn for m, mn in zip(means, mins)]
        upper_err = [mx - m  for m, mx in zip(means, maxs)]
        ax.errorbar(
            x, means,
            yerr=[lower_err, upper_err],
            fmt='none', color='#333333', capsize=5, linewidth=1.2, zorder=4,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel('Sessions per subject', fontsize=9)
        ax.set_title('Sessions per Subject per Dataset\n(bar = mean, whiskers = min/max)',
                     fontsize=11, fontweight='bold', pad=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', labelsize=8)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        # Value labels above bars
        for bar, mean, mn, mx in zip(bars, means, mins, maxs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(upper_err) * 0.08 + 0.05,
                f'{mean:.1f}\n[{mn}–{mx}]',
                ha='center', va='bottom', fontsize=7.5,
            )
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info("Saved: %s", save_path)

    def _plot_steps_per_session(
        self, summary: Dict[str, Any], save_path: Path
    ) -> None:
        """Grouped bar chart: mean steps (trials) per session with min/max error bars."""
        per_ds = summary['per_dataset']
        labels = [d['dataset_id'] for d in per_ds]
        means = [d['steps_per_session']['mean'] for d in per_ds]
        mins  = [d['steps_per_session']['min']  for d in per_ds]
        maxs  = [d['steps_per_session']['max']  for d in per_ds]

        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(max(5, 1.4 * len(labels) + 2), 4))
        bars = ax.bar(x, means, color='#8172B2', edgecolor='white', linewidth=0.6, zorder=3)
        lower_err = [m - mn for m, mn in zip(means, mins)]
        upper_err = [mx - m  for m, mx in zip(means, maxs)]
        ax.errorbar(
            x, means,
            yerr=[lower_err, upper_err],
            fmt='none', color='#333333', capsize=5, linewidth=1.2, zorder=4,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel('Trials (steps) per session', fontsize=9)
        ax.set_title('Steps per Session per Dataset\n(bar = mean, whiskers = min/max)',
                     fontsize=11, fontweight='bold', pad=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', labelsize=8)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        for bar, mean, mn, mx in zip(bars, means, mins, maxs):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(upper_err or [1]) * 0.08 + 0.5,
                f'{mean:.0f}\n[{mn}–{mx}]',
                ha='center', va='bottom', fontsize=7.5,
            )
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info("Saved: %s", save_path)

    @staticmethod
    def _plot_class_distribution(
        summary: Dict[str, Any], save_path: Path
    ) -> None:
        """Grouped bar chart showing class frequencies per dataset."""
        per_ds = summary['per_dataset']

        # Collect all class labels
        all_classes: List[int] = sorted(
            {cls for d in per_ds for cls in d['class_distribution']}
        )
        n_classes = len(all_classes)
        n_datasets = len(per_ds)

        x = np.arange(n_classes)
        width = 0.8 / max(n_datasets, 1)
        palette = plt.cm.tab10.colors  # type: ignore[attr-defined]

        fig, ax = plt.subplots(figsize=(max(5, 1.5 * n_classes + 2), 4))
        for ds_i, ds_entry in enumerate(per_ds):
            counts = [ds_entry['class_distribution'].get(c, 0) for c in all_classes]
            offset = (ds_i - n_datasets / 2 + 0.5) * width
            bars = ax.bar(
                x + offset, counts,
                width=width * 0.9,
                label=ds_entry['dataset_id'],
                color=palette[ds_i % len(palette)],
                edgecolor='white', linewidth=0.4,
            )
            for bar, count in zip(bars, counts):
                if count > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 1,
                        str(count),
                        ha='center', va='bottom', fontsize=7,
                    )

        ax.set_xticks(x)
        ax.set_xticklabels([f'Class {c}' for c in all_classes], fontsize=9)
        ax.set_ylabel('Sample count', fontsize=9)
        ax.set_title('Class Distribution per Dataset', fontsize=11, fontweight='bold', pad=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', labelsize=8)
        ax.legend(fontsize=8, framealpha=0.7)
        ax.set_ylim(bottom=0)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info("Saved: %s", save_path)

    @staticmethod
    def _plot_train_held_out_split(
        split_summary: Dict[str, Any], save_path: Path
    ) -> None:
        """
        Stacked horizontal bar chart showing training vs held-out subjects,
        one row per dataset plus a totals row.
        """
        per_ds_split = split_summary['per_dataset']   # {did: {training, held_out}}
        dataset_ids = list(per_ds_split.keys())
        training_counts = [per_ds_split[d]['training'] for d in dataset_ids]
        held_out_counts = [per_ds_split[d]['held_out'] for d in dataset_ids]

        # Append totals row
        labels = dataset_ids + ['ALL DATASETS']
        training_vals = training_counts + [split_summary['total_training']]
        held_out_vals = held_out_counts + [split_summary['total_held_out']]

        y_pos = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(8, max(3, 0.55 * len(labels) + 1.5)))

        bars_train = ax.barh(
            y_pos, training_vals,
            color='#4C72B0', label='Training', edgecolor='white', linewidth=0.5,
        )
        bars_held = ax.barh(
            y_pos, held_out_vals,
            left=training_vals,
            color='#C44E52', label='Held-out (LSO)', edgecolor='white', linewidth=0.5,
        )

        # Value labels inside bars
        for bar, val in zip(bars_train, training_vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    str(val),
                    ha='center', va='center', fontsize=8, color='white', fontweight='bold',
                )
        for bar, val in zip(bars_held, held_out_vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_y() + bar.get_height() / 2,
                    str(val),
                    ha='center', va='center', fontsize=8, color='white', fontweight='bold',
                )

        # Divider above the totals row
        ax.axhline(y=len(dataset_ids) - 0.5, color='#888888', linewidth=0.8, linestyle='--')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('Number of subjects', fontsize=9)
        ax.set_title('Training vs Held-out Subject Split', fontsize=11, fontweight='bold', pad=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='x', labelsize=8)
        ax.legend(fontsize=8, framealpha=0.7, loc='lower right')
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info("Saved: %s", save_path)


# ---------------------------------------------------------------------------
# Experiment logger — structured JSON + CSV storage
# ---------------------------------------------------------------------------

class ExperimentLogger:
    """
    Persist all experiment metadata and results to disk.

    Writes:
      <output_dir>/
        experiment_config.json   — hyperparameters and dataset info
        per_subject/
          subject_<N>.json       — per-subject stage results
        summary/
          lso_summary.json       — aggregated statistics
          adaptation_curves/     — plots per metric
          confusion_matrices/    — per subject × stage
    """

    def __init__(self, output_dir: Path, experiment_name: str) -> None:
        self.root = Path(output_dir) / experiment_name
        self.per_subject_dir = self.root / 'per_subject'
        self.summary_dir = self.root / 'summary'
        self.curves_dir = self.summary_dir / 'adaptation_curves'
        self.cm_dir = self.summary_dir / 'confusion_matrices'
        self.composition_dir = self.summary_dir / 'dataset_composition'

        for d in [self.root, self.per_subject_dir, self.summary_dir,
                  self.curves_dir, self.cm_dir, self.composition_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def log_config(self, config: Dict[str, Any]) -> None:
        with open(self.root / 'experiment_config.json', 'w') as f:
            json.dump(config, f, indent=2, default=str)

    def log_dataset_composition(
        self,
        dataset_map: Dict[str, Any],
        subject_id_map: Optional[Dict[int, str]] = None,
        lso_global_ids: Optional[List[int]] = None,
        training_global_ids: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Compute, persist, and plot the dataset composition summary.

        Call this once after ``build_joint_dataset`` has resolved subject ID
        mappings but before the per-subject fold loop begins, so that the
        charts reflect the exact split that will be used for training.

        Parameters
        ----------
        dataset_map:
            ``{dataset_id: EEGSampleDataset}`` as returned by ``load_datasets``.
        subject_id_map:
            ``{global_id: "DATASET_ID:sN"}`` from ``build_joint_dataset``.
        lso_global_ids:
            Resolved list of global subject IDs that are held out (LSO).
        training_global_ids:
            Resolved list of global subject IDs used for base-model training.

        Returns
        -------
        The composition summary dict (also written to disk as JSON + PNG charts).
        """
        analyser = DatasetCompositionAnalyser()
        summary = analyser.compute(
            dataset_map=dataset_map,
            subject_id_map=subject_id_map,
            lso_global_ids=lso_global_ids,
            training_global_ids=training_global_ids,
        )
        DatasetCompositionAnalyser.save_json(
            summary,
            self.composition_dir / 'dataset_composition.json',
        )
        analyser.plot_all(summary, self.composition_dir)

        # Log a concise summary at INFO level so it appears in the experiment log
        logger.info(
            "Dataset composition: %d dataset(s), %d total samples, %d total subjects",
            summary['n_datasets'],
            summary['total_samples'],
            summary['total_subjects'],
        )
        for ds_entry in summary['per_dataset']:
            sp = ds_entry['sessions_per_subject']
            st = ds_entry['steps_per_session']
            logger.info(
                "  %-30s  %5d samples  %3d subjects  "
                "sessions/subj: mean=%.1f [%d–%d]  steps/session: mean=%.0f [%d–%d]",
                ds_entry['dataset_id'],
                ds_entry['n_samples'],
                ds_entry['n_subjects'],
                sp['mean'], sp['min'], sp['max'],
                st['mean'], st['min'], st['max'],
            )
        if 'train_held_out' in summary:
            th = summary['train_held_out']
            logger.info(
                "  Split — training: %d subjects  held-out (LSO): %d subjects",
                th['total_training'], th['total_held_out'],
            )

        return summary

    def log_subject_result(self, result: Any) -> None:  # SubjectResult
        path = self.per_subject_dir / f"subject_{result.subject_id}.json"
        data = {
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
            json.dump(data, f, indent=2)

    def save_summary_csv(self, all_results: List[Any]) -> None:
        """Write a flat CSV with one row per (subject, stage)."""
        import csv
        path = self.summary_dir / 'all_results.csv'
        rows = []
        for sr in all_results:
            for stage_r in sr.stage_results:
                row = {
                    'subject_id': sr.subject_id,
                    'stage': stage_r.stage,
                    'n_adaptation_sessions': stage_r.n_adaptation_sessions,
                }
                row.update(stage_r.metrics)
                rows.append(row)

        if not rows:
            return

        fieldnames = list(rows[0].keys())
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Summary CSV written to %s", path)

    def generate_adaptation_plots(
        self,
        all_results: List[Any],
        metrics: Optional[List[str]] = None,
    ) -> None:
        if metrics is None:
            metrics = [
                'valid_accuracy',
                'valid_balanced_accuracy',
                'valid_f1_macro',
                'valid_mcc',
            ]
        analyser = AdaptationCurveAnalyser()
        for metric in metrics:
            summary = analyser.summarise(all_results, metric=metric)
            label = metric.replace('valid_', '').replace('_', ' ').title()
            analyser.plot(
                summary,
                metric_label=label,
                title=f'Progressive Adaptation — {label}',
                save_path=self.curves_dir / f'{metric}.png',
            )

    def generate_subject_confusion_matrices(
        self,
        subject_id: int,
        stage: int,
        cm: np.ndarray,
        class_names: List[str],
    ) -> None:
        analyser = ConfusionMatrixAnalyser(class_names)
        analyser.plot(
            cm,
            title=f'Subject {subject_id} — Stage {stage}',
            save_path=self.cm_dir / f'subject{subject_id}_stage{stage}.png',
        )