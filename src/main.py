import json
import math
import os
import random
from collections import defaultdict

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from torch.utils.data import DataLoader, TensorDataset

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from moabb.paradigms import MotorImagery
from core.genericEEGPTModel import GenericEEGPTModel

PLOTS_DIR = './logs/plots'


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(7)

from moabb.datasets import BNCI2014_001 as DATASET_CLASS
SOURCE_SFREQ = 250
TARGET_SFREQ = 250
TARGET_SAMPLES = 1000
TMIN = 0.0
TMAX = TMIN + TARGET_SAMPLES / TARGET_SFREQ

use_channels_names = [
    'FP1', 'FP2',
    'F7', 'F3', 'FZ', 'F4', 'F8',
    'T7', 'C3', 'CZ', 'C4', 'T8',
    'P7', 'P3', 'PZ', 'P4', 'P8',
    'O1', 'O2',
]

LOAD_PATH = "../checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt"

BATCH_SIZE = 64
MAX_EPOCHS = 100
MAX_LR = 4e-4


def get_native_sfreq(dataset, subject):
    subj_data = dataset.get_data(subjects=[subject])[subject]
    first_session = next(iter(subj_data.values()))
    first_run = next(iter(first_session.values()))
    return first_run.info['sfreq']


def build_paradigm():
    return MotorImagery(
        tmin=TMIN,
        tmax=TMAX,
        resample=TARGET_SFREQ,
        channels=use_channels_names,
    )


def load_all_data(dataset, paradigm):
    subjects = dataset.subject_list

    native_sfreq = get_native_sfreq(dataset, subjects[0])
    if native_sfreq != SOURCE_SFREQ:
        print(f"WARNING: declared SOURCE_SFREQ={SOURCE_SFREQ} but dataset reports "
              f"native sfreq={native_sfreq}. MOABB will still resample correctly "
              f"from the true native rate, but double check SOURCE_SFREQ is set "
              f"the way you intended.")

    X, y_labels, metadata = paradigm.get_data(dataset=dataset, subjects=subjects)

    n_trials, n_chans_found, n_time = X.shape
    if n_time != TARGET_SAMPLES:
        if n_time > TARGET_SAMPLES:
            X = X[:, :, :TARGET_SAMPLES]
        else:
            pad = np.zeros((n_trials, n_chans_found, TARGET_SAMPLES - n_time), dtype=X.dtype)
            X = np.concatenate([X, pad], axis=-1)
        print(f"resampled epoch length was {n_time}, "
              f"{'cropped' if n_time > TARGET_SAMPLES else 'zero-padded'} to {TARGET_SAMPLES}")

    classes = sorted(set(y_labels))
    label_to_int = {c: i for i, c in enumerate(classes)}
    y = np.array([label_to_int[label] for label in y_labels], dtype=np.int64)

    return X.astype(np.float32), y, metadata, classes


def make_loso_loaders(X, y, metadata, held_out_subject, batch_size):
    subject_col = metadata['subject'].to_numpy()
    train_mask = subject_col != held_out_subject
    valid_mask = subject_col == held_out_subject

    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    train_ds = TensorDataset(X_t[train_mask], y_t[train_mask])
    valid_ds = TensorDataset(X_t[valid_mask], y_t[valid_mask])

    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=0, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, num_workers=0, shuffle=False)
    return train_loader, valid_loader


class LiveMetricsPlotter(pl.Callback):
    def __init__(self, held_out_subject, out_dir=PLOTS_DIR):
        self.held_out_subject = held_out_subject
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.history = defaultdict(list)

    def on_train_epoch_end(self, trainer, pl_module):
        self._record_and_plot(trainer)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._record_and_plot(trainer)

    def _record_and_plot(self, trainer):
        epoch = trainer.current_epoch
        current = {}
        for k, v in trainer.callback_metrics.items():
            if k == 'epoch':
                continue
            try:
                current[k] = float(v)
            except (TypeError, ValueError):
                continue
        if not current:
            return

        epochs = self.history['_epoch']
        if epochs and epochs[-1] == epoch:
            # train and val callbacks both fire within the same epoch; merge
            for k, v in current.items():
                self.history[k][-1] = v
        else:
            epochs.append(epoch)
            for k, v in current.items():
                self.history[k].append(v)

        self._save_json()
        self._plot()

    def _save_json(self):
        path = os.path.join(self.out_dir, f'subject_{self.held_out_subject}_history.json')
        with open(path, 'w') as f:
            json.dump(dict(self.history), f, indent=2)

    def _plot(self):
        epochs = self.history.get('_epoch', [])
        if not epochs:
            return

        loss_keys = sorted(k for k in self.history if k != '_epoch' and 'loss' in k.lower())
        acc_keys = sorted(k for k in self.history if k != '_epoch' and 'acc' in k.lower())

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for k in loss_keys:
            vals = self.history[k]
            axes[0].plot(epochs[:len(vals)], vals, marker='o', markersize=3, label=k)
        axes[0].set_title(f'Loss — held-out subject {self.held_out_subject}')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        if loss_keys:
            axes[0].legend()
        axes[0].grid(alpha=0.3)

        for k in acc_keys:
            vals = self.history[k]
            axes[1].plot(epochs[:len(vals)], vals, marker='o', markersize=3, label=k)
        axes[1].set_title(f'Accuracy — held-out subject {self.held_out_subject}')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy')
        if acc_keys:
            axes[1].legend()
        axes[1].grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(self.out_dir, f'subject_{self.held_out_subject}_curves.png'), dpi=130)
        plt.close(fig)

    def final_accuracy(self):
        """Best-effort pick of the held-out/validation accuracy's final value."""
        acc_keys = [k for k in self.history if k != '_epoch' and 'acc' in k.lower()]
        if not acc_keys:
            return None
        # Prefer a key that clearly refers to the held-out/validation split.
        preferred = [k for k in acc_keys if 'val' in k.lower() or 'test' in k.lower()]
        key = preferred[0] if preferred else acc_keys[0]
        vals = self.history[key]
        return vals[-1] if vals else None


def plot_subject_accuracy_bar(subject_accuracies, out_dir=PLOTS_DIR):
    """Bar chart of final held-out accuracy per LOSO subject."""
    subjects = list(subject_accuracies.keys())
    accs = [subject_accuracies[s] for s in subjects]

    fig, ax = plt.subplots(figsize=(max(6, len(subjects) * 0.8), 5))
    bars = ax.bar([str(s) for s in subjects], accs, color='#4C72B0')
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{acc:.3f}', ha='center', va='bottom', fontsize=8)

    if accs:
        mean_acc = float(np.mean(accs))
        ax.axhline(mean_acc, color='gray', linestyle='--', linewidth=1,
                   label=f'mean = {mean_acc:.3f}')
        ax.legend()

    ax.set_xlabel('Held-out subject')
    ax.set_ylabel('Accuracy')
    ax.set_title('LOSO held-out accuracy by subject')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, 'loso_subject_accuracy_bar.png'), dpi=130)
    plt.close(fig)

    with open(os.path.join(out_dir, 'loso_subject_accuracy.json'), 'w') as f:
        json.dump({str(k): v for k, v in subject_accuracies.items()}, f, indent=2)


if __name__ == "__main__":
    dataset = DATASET_CLASS()
    paradigm = build_paradigm()

    X, y, metadata, classes = load_all_data(dataset, paradigm)
    output_classes = len(classes)
    print(f"All subjects: X: {X.shape}, classes: {classes}")

    subject_accuracies = {}

    for held_out_subject in dataset.subject_list:
        print(f"=== LOSO held-out subject {held_out_subject} ===")

        train_loader, valid_loader = make_loso_loaders(
            X, y, metadata, held_out_subject, BATCH_SIZE
        )

        steps_per_epoch = math.ceil(len(train_loader))

        model = GenericEEGPTModel(
            load_path=LOAD_PATH,
            use_channels_names=use_channels_names,
            output_classes=output_classes,
            max_lr=MAX_LR,
            steps_per_epoch=steps_per_epoch,
            max_epochs=MAX_EPOCHS,
        )

        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
        metrics_plotter = LiveMetricsPlotter(held_out_subject=held_out_subject)
        callbacks = [lr_monitor, metrics_plotter]

        trainer = pl.Trainer(
            accelerator='auto',
            devices=[0],
            max_epochs=MAX_EPOCHS,
            callbacks=callbacks,
            enable_checkpointing=False,
            logger=[
                pl_loggers.TensorBoardLogger(
                    './logs/', name=f"EEGPT_{dataset.__class__.__name__}_tb",
                    version=f"loso_subject{held_out_subject}"
                ),
                pl_loggers.CSVLogger(
                    './logs/', name=f"EEGPT_{dataset.__class__.__name__}_csv",
                    version=f"loso_subject{held_out_subject}"
                ),
            ],
        )

        trainer.fit(model, train_loader, valid_loader)

        final_acc = metrics_plotter.final_accuracy()
        if final_acc is not None:
            subject_accuracies[held_out_subject] = final_acc
            print(f"Held-out subject {held_out_subject} final accuracy: {final_acc:.4f}")
        else:
            print(f"Held-out subject {held_out_subject}: no accuracy metric found in "
                  f"trainer.callback_metrics — check the metric names GenericEEGPTModel logs.")

    if subject_accuracies:
        plot_subject_accuracy_bar(subject_accuracies)
        print(f"Per-subject accuracy bar chart written to "
              f"{os.path.join(PLOTS_DIR, 'loso_subject_accuracy_bar.png')}")