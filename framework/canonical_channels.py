"""
canonical_channels.py
---------------------
Defines the pretrained EEGPT canonical 58-channel vocabulary and provides
utilities for mapping arbitrary dataset electrode subsets into it.

The canonical space is fixed to the pretrained model — do NOT modify the
CHANNEL_DICT without retraining the backbone.
"""

from __future__ import annotations
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical channel dictionary (order = position index in backbone embedding)
# ---------------------------------------------------------------------------
CANONICAL_CHANNELS: List[str] = [
    'FP1', 'FPZ', 'FP2',
    'AF7', 'AF3', 'AF4', 'AF8',
    'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8',
    'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'FT8',
    'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8',
    'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'TP8',
    'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8',
    'O1', 'OZ', 'O2',
]

CHANNEL_DICT: Dict[str, int] = {ch.upper(): idx for idx, ch in enumerate(CANONICAL_CHANNELS)}

N_CANONICAL = len(CANONICAL_CHANNELS)  # 58


# ---------------------------------------------------------------------------
# Alias normalisation — common label variants found in public EEG datasets
# ---------------------------------------------------------------------------
_ALIASES: Dict[str, str] = {
    # BCI Competition / BNCI style aliases
    'EEG-FP1': 'FP1', 'EEG-FP2': 'FP2',
    'EEG-F3': 'F3', 'EEG-F4': 'F4', 'EEG-FZ': 'FZ',
    'EEG-C3': 'C3', 'EEG-C4': 'C4', 'EEG-CZ': 'CZ',
    'EEG-P3': 'P3', 'EEG-P4': 'P4', 'EEG-PZ': 'PZ',
    'EEG-O1': 'O1', 'EEG-O2': 'O2',
    'EEG-F7': 'F7', 'EEG-F8': 'F8',
    'EEG-T7': 'T7', 'EEG-T8': 'T8',
    'EEG-P7': 'P7', 'EEG-P8': 'P8',
    'EEG-FC1': 'FC1', 'EEG-FC2': 'FC2', 'EEG-FC5': 'FC5', 'EEG-FC6': 'FC6',
    'EEG-CP1': 'CP1', 'EEG-CP2': 'CP2', 'EEG-CP5': 'CP5', 'EEG-CP6': 'CP6',
    'EEG-AF3': 'AF3', 'EEG-AF4': 'AF4',
    'EEG-FC3': 'FC3', 'EEG-FC4': 'FC4',
    'EEG-CP3': 'CP3', 'EEG-CP4': 'CP4',
    'T3': 'T7', 'T4': 'T8', 'T5': 'P7', 'T6': 'P8',  # old 10-20 names
    'A1': None, 'A2': None,  # reference electrodes — drop
}


def normalise_channel_name(raw_name: str) -> Optional[str]:
    """
    Normalise a raw channel label to the uppercase canonical form.

    Returns None if the channel should be discarded (reference/ground
    electrode or completely unknown name).
    """
    upper = raw_name.strip().upper()
    # Direct hit
    if upper in CHANNEL_DICT:
        return upper
    # Alias lookup
    if upper in _ALIASES:
        alias = _ALIASES[upper]
        return alias  # may be None → caller should filter
    # Strip leading 'EEG ' or 'EEG-' prefix dynamically
    for prefix in ('EEG-', 'EEG '):
        if upper.startswith(prefix):
            stripped = upper[len(prefix):]
            if stripped in CHANNEL_DICT:
                return stripped
    # Unknown
    return None


def resolve_channel_subset(
    dataset_channels: List[str],
) -> Tuple[List[str], List[int], List[int]]:
    """
    Map a dataset's channel list onto the canonical channel space.

    Returns
    -------
    valid_canonical : List[str]
        Canonical names of channels that could be resolved, in the same
        order as the *input* dataset channel list (unknown channels dropped).
    dataset_indices : List[int]
        Indices into *dataset_channels* for each valid channel.
    canonical_indices : List[int]
        Corresponding indices into CANONICAL_CHANNELS for each valid channel.
    """
    valid_canonical: List[str] = []
    dataset_indices: List[int] = []
    canonical_indices: List[int] = []

    seen: set = set()
    for d_idx, raw in enumerate(dataset_channels):
        canon = normalise_channel_name(raw)
        if canon is None:
            logger.debug("Dropping channel '%s' (not in canonical space)", raw)
            continue
        if canon in seen:
            logger.warning("Duplicate canonical channel '%s' from '%s' — skipping", canon, raw)
            continue
        seen.add(canon)
        valid_canonical.append(canon)
        dataset_indices.append(d_idx)
        canonical_indices.append(CHANNEL_DICT[canon])

    if not valid_canonical:
        raise ValueError(
            f"No channels from the dataset could be mapped to the canonical "
            f"EEGPT space. Dataset channels were: {dataset_channels}"
        )

    logger.info(
        "Channel mapping: %d/%d dataset channels resolved to canonical space",
        len(valid_canonical), len(dataset_channels),
    )
    return valid_canonical, dataset_indices, canonical_indices


def validate_channel_subset(channel_names: List[str]) -> None:
    """
    Raise ValueError if any channel_names entry is not in the canonical space.
    Intended for internal consistency checks.
    """
    unknown = [ch for ch in channel_names if ch.upper() not in CHANNEL_DICT]
    if unknown:
        raise ValueError(f"Unknown canonical channels: {unknown}")
