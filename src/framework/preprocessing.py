"""
preprocessing.py
----------------
Reusable, composable preprocessing stages that convert raw EEG data
from arbitrary datasets into tensors compatible with the unmodified
genericEEGPTModel:

    Input  : [n_channels, n_timepoints]   (raw, arbitrary sample-rate)
    Output : [n_mapped_channels, 1024]    (4 s @ 256 Hz, mean-centred)

Pipeline stages (each is a standalone callable):
    1. ChannelSelector      — pick + reorder channels from canonical map
    2. Resampler            — resample to 256 Hz via linear interpolation
    3. Segmenter            — cut continuous recordings into 4-second windows
    4. BaselineCorrector    — subtract per-channel temporal mean
    5. AmplitudeClipper     — clip extreme artefacts (optional)
    6. EEGPTPreprocessor    — convenience wrapper for the full pipeline
"""

from __future__ import annotations
from typing import List, Tuple, Optional, Generator
import logging

import torch
import torch.nn.functional as F
import numpy as np

from .canonical_channels import resolve_channel_subset

logger = logging.getLogger(__name__)

# Model fixed constants — do NOT change
TARGET_SAMPLE_RATE: int = 256   # Hz
TARGET_WINDOW_SAMPLES: int = 1024  # 4 s × 256 Hz


# ---------------------------------------------------------------------------
# Stage 1 — Channel selection
# ---------------------------------------------------------------------------

class ChannelSelector:
    """
    Select, reorder, and rename channels from a raw dataset array.

    After construction, ``dataset_indices`` holds the row indices to keep
    from any [n_channels, T] array, and ``canonical_names`` holds the
    corresponding canonical names in the same order.
    """

    def __init__(self, dataset_channel_names: List[str]) -> None:
        (
            self.canonical_names,
            self.dataset_indices,
            self.canonical_indices,
        ) = resolve_channel_subset(dataset_channel_names)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [n_raw_channels, T]
        Returns : [n_mapped_channels, T]
        """
        return x[self.dataset_indices, :]

    def n_channels(self) -> int:
        return len(self.canonical_names)


# ---------------------------------------------------------------------------
# Stage 2 — Resampler
# ---------------------------------------------------------------------------

class Resampler:
    """
    Resample EEG from source_sfreq to TARGET_SAMPLE_RATE (256 Hz).

    Uses torch bilinear (nearest / linear) interpolation; works on
    [n_channels, T] tensors without requiring scipy.
    """

    def __init__(self, source_sfreq: int, mode: str = 'linear') -> None:
        self.source_sfreq = source_sfreq
        self.mode = mode
        self._ratio = TARGET_SAMPLE_RATE / source_sfreq

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [n_channels, T]
        Returns : [n_channels, T_resampled]
        """
        if self.source_sfreq == TARGET_SAMPLE_RATE:
            return x

        target_len = round(x.shape[-1] * self._ratio)
        # interpolate expects [batch, channels, length]
        out = F.interpolate(
            x.unsqueeze(0).float(),
            size=target_len,
            mode=self.mode,
            align_corners=False if self.mode == 'linear' else None,
        ).squeeze(0)
        return out


# ---------------------------------------------------------------------------
# Stage 3 — Segmenter
# ---------------------------------------------------------------------------

class Segmenter:
    """
    Slice a continuous [n_channels, T] recording into non-overlapping
    TARGET_WINDOW_SAMPLES windows.

    Optionally applies a fixed sample offset to align windows with
    stimulus onset (e.g. skip a pre-stimulus baseline period).
    """

    def __init__(
        self,
        window_samples: int = TARGET_WINDOW_SAMPLES,
        stride_samples: Optional[int] = None,
        onset_offset: int = 0,
    ) -> None:
        self.window = window_samples
        self.stride = stride_samples if stride_samples is not None else window_samples
        self.onset = onset_offset

    def __call__(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        x : [n_channels, T]
        Returns : list of [n_channels, window_samples] tensors
        """
        x = x[:, self.onset:]
        T = x.shape[-1]
        segments = []
        start = 0
        while start + self.window <= T:
            segments.append(x[:, start:start + self.window])
            start += self.stride
        return segments

    def segment_epochs(
        self,
        epochs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convenience method when the input is already epoched:
        epochs : [n_trials, n_channels, T]
        Returns : [n_trials, n_channels, window_samples]
        (pads or crops each trial to exactly window_samples)
        """
        T = epochs.shape[-1]
        if T == self.window:
            return epochs
        elif T > self.window:
            return epochs[:, :, :self.window]
        else:
            pad = self.window - T
            return F.pad(epochs, (0, pad))


# ---------------------------------------------------------------------------
# Stage 4 — Baseline corrector (mean-centre per channel per window)
# ---------------------------------------------------------------------------

class BaselineCorrector:
    """Subtract the per-channel temporal mean from each window."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [..., n_channels, T]
        """
        return x - x.mean(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------
# Stage 5 — Amplitude clipper (artefact rejection)
# ---------------------------------------------------------------------------

class AmplitudeClipper:
    """
    Clip signal amplitude to ±threshold µV (or arbitrary units).
    Default 800 µV covers even noisy EEG without removing genuine signal.
    """

    def __init__(self, threshold: float = 800.0) -> None:
        self.threshold = threshold

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(-self.threshold, self.threshold)


# ---------------------------------------------------------------------------
# Combined convenience wrapper
# ---------------------------------------------------------------------------

class EEGPTPreprocessor:
    """
    Full preprocessing pipeline for a single dataset.

    Converts raw trials from:
        [n_trials, n_raw_channels, n_raw_timepoints]
    to:
        [n_trials, n_mapped_channels, 1024]

    ready for the unmodified genericEEGPTModel.

    Parameters
    ----------
    dataset_channel_names : List[str]
        Raw channel names as provided by the dataset.
    source_sfreq : int
        Original sampling frequency of the dataset.
    clip_amplitude : Optional[float]
        If given, clip each sample to ±clip_amplitude after baseline correction.
    resample_mode : str
        Interpolation mode for resampling ('linear' or 'nearest').
    """

    def __init__(
        self,
        dataset_channel_names: List[str],
        source_sfreq: int,
        clip_amplitude: Optional[float] = 800.0,
        resample_mode: str = 'linear',
    ) -> None:
        self.channel_selector = ChannelSelector(dataset_channel_names)
        self.resampler = Resampler(source_sfreq, mode=resample_mode)
        self.segmenter = Segmenter()
        self.baseline = BaselineCorrector()
        self.clipper = AmplitudeClipper(clip_amplitude) if clip_amplitude else None

        self.channel_names: List[str] = self.channel_selector.canonical_names
        self.n_channels: int = self.channel_selector.n_channels()

        logger.info(
            "EEGPTPreprocessor: %d channels, source_sfreq=%d Hz → %d Hz",
            self.n_channels, source_sfreq, TARGET_SAMPLE_RATE,
        )

    def process_epochs(self, X: np.ndarray) -> torch.Tensor:
        """
        Process a batch of pre-segmented epochs.

        Parameters
        ----------
        X : np.ndarray, shape [n_trials, n_raw_channels, n_timepoints]

        Returns
        -------
        torch.Tensor, shape [n_trials, n_mapped_channels, 1024]
        """
        x = torch.FloatTensor(X)                         # [N, C_raw, T]
        x = x[:, self.channel_selector.dataset_indices, :]  # [N, C_map, T]
        x = self.resampler(x.view(-1, x.shape[-1]))      # treats as one long batch row
        # resampler works on [C, T]; apply per-trial
        x_raw = torch.FloatTensor(X)[:, self.channel_selector.dataset_indices, :]
        processed = []
        for trial in x_raw:                              # trial: [C, T]
            t = self.resampler(trial)                    # [C, T_256]
            t = self.segmenter.segment_epochs(t.unsqueeze(0)).squeeze(0)  # [C, 1024]
            t = self.baseline(t)
            if self.clipper:
                t = self.clipper(t)
            processed.append(t)
        return torch.stack(processed, dim=0)             # [N, C, 1024]

    def process_continuous(
        self, x: np.ndarray
    ) -> List[torch.Tensor]:
        """
        Segment a continuous recording into 4-second windows.

        Parameters
        ----------
        x : np.ndarray, shape [n_raw_channels, n_timepoints]

        Returns
        -------
        List of tensors each shaped [n_mapped_channels, 1024]
        """
        t = torch.FloatTensor(x)
        t = self.channel_selector(t)
        t = self.resampler(t)
        windows = self.segmenter(t)
        result = []
        for w in windows:
            w = self.baseline(w)
            if self.clipper:
                w = self.clipper(w)
            result.append(w)
        return result
