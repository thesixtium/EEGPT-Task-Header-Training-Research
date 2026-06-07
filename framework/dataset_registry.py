"""
dataset_registry.py
-------------------
Abstract base class for dataset integrations plus a concrete MOABB adapter
that replaces and generalises the existing moabbMotorImageryDataLoader.

The MOABB adapter:
 * loads all subjects and sessions from any MOABB-compatible dataset
 * applies the EEGPTPreprocessor pipeline
 * maps labels through the unified LabelSchema
 * returns fully annotated EEGSampleDataset objects

Adding a new data source requires only subclassing BaseDatasetLoader and
implementing `load_all_subjects`.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import torch

from .eeg_dataset import EEGSample, EEGSampleDataset
from .preprocessing import EEGPTPreprocessor, TARGET_SAMPLE_RATE
from .label_schema import LabelSchema, DATASET_SAMPLE_RATES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseDatasetLoader(ABC):
    """
    Contract that every dataset integration must fulfil.

    Subclasses are responsible for:
    - fetching raw data (numpy arrays)
    - constructing EEGSampleDataset with full metadata
    - exposing channel names and class count
    """

    @abstractmethod
    def load_all_subjects(self) -> EEGSampleDataset:
        """Return all samples across all subjects and sessions."""
        ...

    @abstractmethod
    def get_channel_names(self) -> List[str]:
        """Return the canonical channel names this dataset exposes."""
        ...

    @abstractmethod
    def get_n_classes(self) -> int:
        ...

    @abstractmethod
    def get_dataset_id(self) -> str:
        ...


# ---------------------------------------------------------------------------
# Channel-name helper (avoids a second get_data call)
# ---------------------------------------------------------------------------

def _read_channel_names_from_raw(moabb_dataset, subject_id: int) -> List[str]:
    """
    Extract channel names by opening one raw MNE file directly, bypassing the
    MOABB paradigm entirely so we never trigger the sfreq-mismatch error.
    Falls back gracefully if the dataset does not support get_data(return_epochs=True)
    per-session.
    """
    try:
        raw_dict = moabb_dataset.get_data(subjects=[subject_id])
        # raw_dict shape: {subject: {session: {run: Raw}}}
        for subj_data in raw_dict.values():
            for sess_data in subj_data.values():
                for run_raw in sess_data.values():
                    if hasattr(run_raw, 'ch_names'):
                        return list(run_raw.ch_names)
                    if hasattr(run_raw, 'info'):
                        return list(run_raw.info['ch_names'])
    except Exception as exc:
        logger.warning("Could not read channel names from raw: %s", exc)
    raise RuntimeError(
        f"Cannot determine channel names for {type(moabb_dataset).__name__}. "
        "Add a channel_names attribute to the dataset object."
    )


# ---------------------------------------------------------------------------
# MOABB adapter — generalises the existing moabbMotorImageryDataLoader
# ---------------------------------------------------------------------------

class MoabbDatasetLoader(BaseDatasetLoader):
    """
    Load any MOABB MotorImagery-compatible dataset for all subjects.

    Parameters
    ----------
    moabb_dataset : moabb dataset object (e.g. BNCI2014_004())
    dataset_id : str — matches a key in DATASET_LABEL_MAPS
    label_schema : LabelSchema — pre-constructed for this dataset
    n_classes : int — number of MI classes
    fmin, fmax : bandpass limits
    tmin, tmax : epoch window in seconds
    resample : target sampling frequency passed to MOABB paradigm
    subjects : optional list of subject numbers; None = all subjects
    """

    def __init__(
        self,
        moabb_dataset,
        dataset_id: str,
        label_schema: LabelSchema,
        n_classes: int,
        fmin: float = 0.5,
        fmax: float = 40.0,
        tmin: float = 0.0,
        tmax: float = 4.0,
        resample: Optional[float] = None,
        subjects: Optional[List[int]] = None,
    ) -> None:
        self._moabb_dataset = moabb_dataset
        self._dataset_id = dataset_id
        self._label_schema = label_schema
        self._n_classes = n_classes
        self._fmin = fmin
        self._fmax = fmax
        self._tmin = tmin
        self._tmax = tmax
        self._resample = resample
        self._subjects = subjects

        self._channel_names: Optional[List[str]] = None   # set after first load

    # ------------------------------------------------------------------
    def load_all_subjects(self) -> EEGSampleDataset:
        """
        Fetch data from MOABB for all (or specified) subjects and sessions.

        Returns a fully annotated EEGSampleDataset.
        """
        from moabb.paradigms import P300

        # Always supply an explicit resample rate so MOABB/MNE resamples every
        # session to the same sfreq before concatenating epochs.  Without this,
        # datasets whose sessions were recorded at different native rates (e.g.
        # Mainsah2025_A) cause: ValueError: epochs[N].info['sfreq'] must match
        _resample = int(
            self._resample
            or DATASET_SAMPLE_RATES.get(self._dataset_id)
            or getattr(self._moabb_dataset, 'sfreq', None)
            or TARGET_SAMPLE_RATE
        )

        paradigm = P300(
            tmin=self._tmin,
            tmax=self._tmax,
            resample=_resample,
        )

        all_subjects = self._subjects or self._moabb_dataset.subject_list

        all_samples: List[EEGSample] = []
        preprocessor: Optional[EEGPTPreprocessor] = None

        for subject_id in all_subjects:
            logger.info("Loading subject %d from %s", subject_id, self._dataset_id)
            try:
                X, y_str, metadata = paradigm.get_data(
                    dataset=self._moabb_dataset,
                    subjects=[subject_id],
                    return_epochs=False,
                )
            except Exception as exc:
                logger.warning("Failed to load subject %d: %s", subject_id, exc)
                continue

            # Build preprocessor once — resolve channel names without a second
            # get_data() call (which would crash on datasets with mixed sfreq).
            # Priority: dataset attribute > raw MNE file > paradigm channels attr.
            if preprocessor is None:
                if hasattr(self._moabb_dataset, 'channel_names'):
                    raw_channel_names = list(self._moabb_dataset.channel_names)
                elif hasattr(paradigm, 'channels') and paradigm.channels:
                    raw_channel_names = list(paradigm.channels)
                else:
                    # Read one raw file directly — avoids concatenation entirely.
                    raw_channel_names = _read_channel_names_from_raw(
                        self._moabb_dataset, subject_id
                    )
                # The paradigm already resampled to _resample, so tell the
                # preprocessor the data is already at that rate.
                source_sfreq = _resample
                preprocessor = EEGPTPreprocessor(
                    dataset_channel_names=raw_channel_names,
                    source_sfreq=source_sfreq,
                    target_window_samples=TARGET_SAMPLE_RATE,  # 1 s at 256 Hz = 256 samples
                )
                self._channel_names = preprocessor.channel_names

            # Process epochs → [N, C, 1024]
            X_proc = preprocessor.process_epochs(X)   # Tensor[N, C, 1024]

            # Extract session ids from metadata
            sessions = metadata['session'].values if hasattr(metadata, 'values') else (
                metadata['session'] if 'session' in metadata.columns else ['0'] * len(y_str)
            )
            session_int_map: Dict[str, int] = {}

            for trial_idx in range(len(y_str)):
                raw_label = y_str[trial_idx]
                compact_label = self._label_schema.translate(raw_label)
                if compact_label is None:
                    continue  # label not in active set

                sess_str = str(sessions[trial_idx])
                if sess_str not in session_int_map:
                    session_int_map[sess_str] = len(session_int_map)
                session_id = session_int_map[sess_str]

                all_samples.append(EEGSample(
                    eeg=X_proc[trial_idx],
                    channel_names=preprocessor.channel_names,
                    label=compact_label,
                    subject_id=int(subject_id),
                    session_id=session_id,
                    dataset_id=self._dataset_id,
                ))

        logger.info(
            "%s: loaded %d samples across %d subjects",
            self._dataset_id, len(all_samples), len(all_subjects),
        )
        return EEGSampleDataset(all_samples)

    def load_single_subject(self, subject_id: int) -> EEGSampleDataset:
        """Convenience: load only one subject."""
        prev = self._subjects
        self._subjects = [subject_id]
        ds = self.load_all_subjects()
        self._subjects = prev
        return ds

    # ------------------------------------------------------------------
    def get_channel_names(self) -> List[str]:
        if self._channel_names is None:
            raise RuntimeError("Call load_all_subjects() first to resolve channels.")
        return self._channel_names

    def get_n_classes(self) -> int:
        return self._label_schema.n_classes

    def get_dataset_id(self) -> str:
        return self._dataset_id


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _infer_channels(moabb_dataset) -> List[str]:
    """
    Fallback: retrieve channel names directly from the MOABB dataset object
    by inspecting common attribute names.
    """
    for attr in ('channel_names', 'channels', 'ch_names'):
        if hasattr(moabb_dataset, attr):
            return list(getattr(moabb_dataset, attr))
    raise AttributeError(
        f"Cannot infer channel names from {type(moabb_dataset).__name__}. "
        "Set channel_names attribute on the dataset object manually."
    )