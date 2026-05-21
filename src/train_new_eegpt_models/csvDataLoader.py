import math
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.core.eegDataset import EegDataset


def temporal_interpolation(x, desired_sequence_length, mode='nearest', use_avg=True):
    if use_avg:
        x = x - torch.mean(x, dim=-2, keepdim=True)
    if len(x.shape) == 2:
        return torch.nn.functional.interpolate(x.unsqueeze(0), desired_sequence_length, mode=mode).squeeze(0)
    elif len(x.shape) == 3:
        return torch.nn.functional.interpolate(x, desired_sequence_length, mode=mode)
    else:
        raise ValueError("TemporalInterpolation only support sequence of single dim channels with optional batch")


def get_data_single_subject(X, y, target_sample=1024):
    # X shape: [n_trials, n_channels, n_timepoints]
    # y shape: [n_trials]

    x = torch.FloatTensor(X)
    y = torch.LongTensor(y)

    if target_sample > 0:
        x = temporal_interpolation(x, target_sample)

    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.2, stratify=y
    )
    train_x, valid_x, train_y, valid_y = train_test_split(
        train_x, train_y, test_size=0.1, stratify=train_y
    )

    return EegDataset(train_x, train_y), \
           EegDataset(valid_x, valid_y), \
           EegDataset(test_x, test_y)


class CsvEegDataLoader:
    def __init__(
        self,
        csv_path: str,
        class_names: list,         # e.g. ["rest", "left_hand", "right_hand"]
        trial_length: int,         # number of rows per trial
        batch_size: int = 32,
        target_sample: int = 1024,
        timestamp_col: str = "timestamp",
        label_col: str = "y",
    ):
        """
        Args:
            csv_path:      Path to the CSV file.
            class_names:   Ordered list of class labels. Index 0 → y=0, index 1 → y=1, etc.
            trial_length:  Number of consecutive rows that form one trial/epoch.
            batch_size:    DataLoader batch size.
            target_sample: Timepoints to interpolate each trial to (0 = no interpolation).
            timestamp_col: Name of the timestamp column to drop (set to None to skip).
            label_col:     Name of the label column.
        """
        # Build class_names dict: {0: "rest", 1: "left_hand", ...}
        self.class_names = {i: name for i, name in enumerate(class_names)}

        # ------------------------------------------------------------------
        # 1. Load CSV
        # ------------------------------------------------------------------
        df = pd.read_csv(csv_path)

        # Drop timestamp column if present
        if timestamp_col and timestamp_col in df.columns:
            df = df.drop(columns=[timestamp_col])

        # Separate features and labels
        y_raw = df[label_col].values.astype(np.int64)
        X_raw = df.drop(columns=[label_col]).values.astype(np.float32)
        # X_raw shape: [n_rows, n_channels]

        # ------------------------------------------------------------------
        # 2. Segment rows into trials of shape [n_trials, n_channels, trial_length]
        # ------------------------------------------------------------------
        n_rows, n_channels = X_raw.shape
        n_trials = n_rows // trial_length

        # Trim any leftover rows that don't fill a complete trial
        X_raw = X_raw[: n_trials * trial_length]
        y_raw = y_raw[: n_trials * trial_length]

        # Reshape to [n_trials, trial_length, n_channels] then transpose
        X_trials = X_raw.reshape(n_trials, trial_length, n_channels)
        X_trials = X_trials.transpose(0, 2, 1)          # → [n_trials, n_channels, trial_length]

        # Take the label from the first row of each trial
        y_trials = y_raw.reshape(n_trials, trial_length)[:, 0]

        # ------------------------------------------------------------------
        # 3. Build datasets & loaders
        # ------------------------------------------------------------------
        train_dataset, valid_dataset, test_dataset = get_data_single_subject(
            X=X_trials,
            y=y_trials,
            target_sample=target_sample,
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=True,
        )

        self.valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=False,
        )

        self.test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=0,
            shuffle=False,
        )

        self.steps_per_epoch = math.ceil(len(self.train_loader))

    # ------------------------------------------------------------------
    # Public API (matches MoabbMotorImageryDataLoader)
    # ------------------------------------------------------------------
    def get_class_names(self):
        return self.class_names

    def get_train_loader(self):
        return self.train_loader

    def get_valid_loader(self):
        return self.valid_loader

    def get_test_loader(self):
        return self.test_loader

    def get_steps_per_epoch(self):
        return self.steps_per_epoch