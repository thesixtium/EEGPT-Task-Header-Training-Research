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

import mne
import numpy as np
import torch

# ---------------------------------------------------------------------------
# Silence MNE and MOABB's chatty internal loggers.
# "Adding metadata with N columns" and "X matching events found" are MNE
# internal bookkeeping messages that appear during paradigm.get_data().
# They indicate normal operation and add no useful information here.
# ---------------------------------------------------------------------------
mne.set_log_level("WARNING")
logging.getLogger("moabb").setLevel(logging.WARNING)

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
    dataset_index : 1-based position of this dataset in the full run
                    (used in progress log lines, e.g. [3/47])
    dataset_total : total number of datasets in the run
    cache_dir : if provided, processed .pt tensors are read/written here
                so subjects that have already been preprocessed are not
                re-downloaded or re-processed on subsequent runs
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
        dataset_index: int = 1,
        dataset_total: int = 1,
        cache_dir: Optional[str] = None,
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
        self._dataset_index = dataset_index
        self._dataset_total = dataset_total
        self._cache_dir = cache_dir

        self._channel_names: Optional[List[str]] = None   # set after first load

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dataset_prefix(self) -> str:
        """Short string used at the start of every log line for this dataset."""
        return f"[{self._dataset_index}/{self._dataset_total}] {self._dataset_id}"

    def _subject_cache_path(self, subject_id: int):
        """
        Return a Path for the cached .pt file for this subject, or None if
        caching is disabled.
        """
        if self._cache_dir is None:
            return None
        from pathlib import Path
        cache_root = Path(self._cache_dir) / self._dataset_id
        cache_root.mkdir(parents=True, exist_ok=True)
        return cache_root / f"subject_{subject_id:04d}.pt"

    def _load_subject_from_cache(self, subject_id: int) -> Optional[List[EEGSample]]:
        """
        Attempt to load pre-processed samples for subject_id from disk cache.
        Returns a list of EEGSample on hit, or None on miss.
        """
        path = self._subject_cache_path(subject_id)
        if path is None or not path.exists():
            return None
        try:
            samples = torch.load(path, weights_only=False)
            logger.info(
                "%s — subject %s  →  loaded %d samples from cache  (%s)",
                self._dataset_prefix(), subject_id, len(samples), path,
            )
            return samples
        except Exception as exc:
            logger.warning(
                "%s — subject %s  cache read failed (%s), will re-process",
                self._dataset_prefix(), subject_id, exc,
            )
            return None

    def _save_subject_to_cache(self, subject_id: int, samples: List[EEGSample]) -> None:
        """Persist processed samples to disk cache (silently skips if disabled)."""
        path = self._subject_cache_path(subject_id)
        if path is None:
            return
        try:
            torch.save(samples, path)
            logger.info(
                "%s — subject %s  →  cached %d samples  →  %s",
                self._dataset_prefix(), subject_id, len(samples), path,
            )
        except Exception as exc:
            logger.warning(
                "%s — subject %s  cache write failed: %s",
                self._dataset_prefix(), subject_id, exc,
            )

    # ------------------------------------------------------------------
    def load_all_subjects(self) -> EEGSampleDataset:
        """
        Fetch data from MOABB for all (or specified) subjects and sessions.
        Subjects whose .pt cache file already exists are loaded from disk
        rather than re-downloaded and re-processed.

        Returns a fully annotated EEGSampleDataset.
        """
        from moabb.paradigms import P300

        paradigm = P300(
            tmin=self._tmin,
            tmax=self._tmax,
            resample=self._resample,
        )

        all_subjects = self._subjects or self._moabb_dataset.subject_list
        n_subjects = len(all_subjects)

        logger.info(
            "%s — starting load  (%d subject%s)",
            self._dataset_prefix(),
            n_subjects,
            "s" if n_subjects != 1 else "",
        )

        all_samples: List[EEGSample] = []
        preprocessor: Optional[EEGPTPreprocessor] = None

        for subj_idx, subject_id in enumerate(all_subjects, start=1):

            # ---------------------------------------------------------------
            # Progress header — printed before anything else for this subject
            # ---------------------------------------------------------------
            logger.info(
                "%s — subject %d/%d  (id=%s)",
                self._dataset_prefix(), subj_idx, n_subjects, subject_id,
            )

            # ---------------------------------------------------------------
            # Cache check — skip download + preprocessing if already on disk
            # ---------------------------------------------------------------
            cached = self._load_subject_from_cache(subject_id)
            if cached is not None:
                all_samples.extend(cached)
                # Still need channel names for the dataset object; grab from
                # the first cached sample if preprocessor not yet built.
                if self._channel_names is None and cached:
                    self._channel_names = list(cached[0].channel_names)
                continue

            logger.info(
                "%s — subject %d/%d  →  not cached, downloading / preprocessing",
                self._dataset_prefix(), subj_idx, n_subjects,
            )

            # ------------------------------------------------------------------
            # Resolve channel names on the first subject.
            # If the dataset exposes them directly, use that and load normally.
            # Otherwise we must call get_data with return_epochs=True once so
            # we can read ch_names — then derive X from the epochs object itself
            # rather than issuing a second get_data call (which triggers MNE's
            # concatenate_epochs and fails when sessions have different sfreqs).
            # ------------------------------------------------------------------
            if preprocessor is None and not hasattr(self._moabb_dataset, 'channel_names'):
                try:
                    epochs_obj, y_str, metadata = paradigm.get_data(
                        dataset=self._moabb_dataset,
                        subjects=[subject_id],
                        return_epochs=True,
                    )
                except Exception as exc:
                    logger.warning(
                        "%s — subject %d/%d  FAILED: %s",
                        self._dataset_prefix(), subj_idx, n_subjects, exc,
                    )
                    continue

                raw_channel_names = epochs_obj.ch_names
                # Convert epochs → numpy array [N, C, T] ourselves, same layout
                # as return_epochs=False.
                X = epochs_obj.get_data()  # shape (N, C, T)
                # y_str and metadata are already set from the return above.
            else:
                try:
                    X, y_str, metadata = paradigm.get_data(
                        dataset=self._moabb_dataset,
                        subjects=[subject_id],
                        return_epochs=False,
                    )
                except Exception as exc:
                    logger.warning(
                        "%s — subject %d/%d  FAILED: %s",
                        self._dataset_prefix(), subj_idx, n_subjects, exc,
                    )
                    continue

            # Build the preprocessor once we have channel names
            if preprocessor is None:
                if hasattr(self._moabb_dataset, 'channel_names'):
                    raw_channel_names = self._moabb_dataset.channel_names
                # else: raw_channel_names already set from the epochs branch above

                source_sfreq = int(
                    self._resample
                    or DATASET_SAMPLE_RATES.get(self._dataset_id)
                    or getattr(self._moabb_dataset, 'sfreq', None)
                    or 256
                )
                preprocessor = EEGPTPreprocessor(
                    dataset_channel_names=raw_channel_names,
                    source_sfreq=source_sfreq,
                )
                self._channel_names = preprocessor.channel_names

            # Process epochs → [N, C, 1024]
            X_proc = preprocessor.process_epochs(X)   # Tensor[N, C, 1024]

            # Extract session ids from metadata
            sessions = metadata['session'].values if hasattr(metadata, 'values') else (
                metadata['session'] if 'session' in metadata.columns else ['0'] * len(y_str)
            )
            session_int_map: Dict[str, int] = {}

            subject_samples: List[EEGSample] = []
            for trial_idx in range(len(y_str)):
                raw_label = y_str[trial_idx]
                compact_label = self._label_schema.translate(raw_label)
                if compact_label is None:
                    continue  # label not in active set

                sess_str = str(sessions[trial_idx])
                if sess_str not in session_int_map:
                    session_int_map[sess_str] = len(session_int_map)
                session_id = session_int_map[sess_str]

                subject_samples.append(EEGSample(
                    eeg=X_proc[trial_idx],
                    channel_names=preprocessor.channel_names,
                    label=compact_label,
                    subject_id=int(subject_id),
                    session_id=session_id,
                    dataset_id=self._dataset_id,
                ))

            logger.info(
                "%s — subject %d/%d  →  %d trials, %d session%s",
                self._dataset_prefix(), subj_idx, n_subjects,
                len(subject_samples),
                len(session_int_map),
                "s" if len(session_int_map) != 1 else "",
            )

            self._save_subject_to_cache(subject_id, subject_samples)
            all_samples.extend(subject_samples)

        logger.info(
            "%s — DONE  %d total samples",
            self._dataset_prefix(), len(all_samples),
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