"""
label_schema.py
---------------
Defines a global motor-imagery label vocabulary and provides per-dataset
label mapping into that unified space.

Datasets may expose only subsets of the global vocabulary. The schema
assigns stable integer indices so that multi-dataset batches are
consistently labelled.

Adding a new dataset label mapping requires only registering a new entry
in DATASET_LABEL_MAPS below.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global vocabulary
# ---------------------------------------------------------------------------
GLOBAL_LABELS: List[str] = [
    'Target',    # 0
    'NonTarget',   # 1
]

GLOBAL_LABEL_TO_IDX: Dict[str, int] = {lbl: idx for idx, lbl in enumerate(GLOBAL_LABELS)}
N_GLOBAL_CLASSES = len(GLOBAL_LABELS)


# ---------------------------------------------------------------------------
# Per-dataset string → global-label mappings
# Keys are lowercase dataset string labels as returned by MOABB paradigm.
# ---------------------------------------------------------------------------
DATASET_LABEL_MAPS: Dict[str, Dict[str, str]] = {
    'BNCI2014_009': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'BI2014a': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'BI2013a': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'BI2014b': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'BI2015a': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'BI2015b': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'BNCI2014_008': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'Chailloux2020': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'Cattan2019_VR': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'EPFLP300': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'ErpCore2021_ERN': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'ErpCore2021_LRP': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'ErpCore2021_MMN': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'ErpCore2021_N170': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'ErpCore2021_N2pc': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'ErpCore2021_N400': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'ErpCore2021_P3': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'GuttmannFlury2025_P300': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'Lee2019_ERP': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'Lee2024_AC': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'Lee2024_BS': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'Lee2024_DL': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'Lee2024_EL': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'Lee2024_TV': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'Mainsah2025_A': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'Mainsah2025_B': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'Mainsah2025_C': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'Mainsah2025_D': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'Mainsah2025_E': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'Mainsah2025_F': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'Mainsah2025_G': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'Mainsah2025_H': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'Mainsah2025_I': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'Mainsah2025_J': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'Mainsah2025_K': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'Mainsah2025_L': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'Mainsah2025_M': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'Mainsah2025_P': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'Mainsah2025_N': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'Mainsah2025_Q': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'Mainsah2025_S1': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'Mainsah2025_R': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'Mainsah2025_S2': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'Speier2017': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'Zheng2020': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'Mainsah2025_O': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
    'Zhang2025': {
        'Target': 'Target',
        'NonTarget': 'NonTarget',
    },
}


# ---------------------------------------------------------------------------
# Per-dataset native sampling frequencies (Hz)
#
# These are the rates at which each dataset was recorded.  The preprocessor
# will resample to TARGET_SAMPLE_RATE (1024 samples / 4 s = 256 Hz) as
# needed.  Edit this table when adding a new dataset rather than relying on
# the MOABB dataset object's `.sfreq` attribute, which is not consistently
# available across all MOABB versions.
# ---------------------------------------------------------------------------
DATASET_SAMPLE_RATES: Dict[str, int] = {
    'BNCI2014_009':  256,
    'BI2014a': 512,
    'BI2013a': 512,
    'BI2014b': 512,
    'BI2015a': 512,
    'BI2015b': 512,
    'BNCI2014_008': 256,
    'Chailloux2020': 256,
    'Cattan2019_VR': 512,
    'EPFLP300': 2048,
    'ErpCore2021_ERN': 1024,
    'ErpCore2021_LRP': 1024,
    'ErpCore2021_MMN': 1024,
    'ErpCore2021_N170': 1024,
    'ErpCore2021_N2pc': 1024,
    'ErpCore2021_N400': 1024,
    'ErpCore2021_P3': 1024,
    'GuttmannFlury2025_P300': 1000,
    'Lee2019_ERP': 1000,
    'Lee2024_AC': 500,
    'Lee2024_BS': 500,
    'Lee2024_DL': 500,
    'Lee2024_EL': 500,
    'Lee2024_TV': 500,
    'Mainsah2025_A': 256,
    'Mainsah2025_B': 256,
    'Mainsah2025_C': 256,
    'Mainsah2025_D': 256,
    'Mainsah2025_E': 256,
    'Mainsah2025_F': 256,
    'Mainsah2025_G': 256,
    'Mainsah2025_H': 256,
    'Mainsah2025_I': 256,
    'Mainsah2025_J': 256,
    'Mainsah2025_K': 256,
    'Mainsah2025_L': 256,
    'Mainsah2025_M': 256,
    'Mainsah2025_P': 256,
    'Mainsah2025_N': 256,
    'Mainsah2025_Q': 256,
    'Mainsah2025_S1': 256,
    'Mainsah2025_R': 256,
    'Mainsah2025_S2': 256,
    'Speier2017': 256,
    'Zheng2020': 1000,
    'Mainsah2025_O': 256,
    'Zhang2025': 1000

}


class LabelSchema:
    """
    Handles label translation for a single dataset.

    Parameters
    ----------
    dataset_id : str
        Must match a key in DATASET_LABEL_MAPS.
    active_global_labels : Optional[List[str]]
        If given, restrict the output space to this ordered subset of
        GLOBAL_LABELS. Useful when a model head is trained for a fixed
        subset rather than the full vocabulary.
    """

    def __init__(
        self,
        dataset_id: str,
        active_global_labels: Optional[List[str]] = None,
    ) -> None:
        if dataset_id not in DATASET_LABEL_MAPS:
            raise KeyError(
                f"No label map registered for dataset '{dataset_id}'. "
                f"Available: {list(DATASET_LABEL_MAPS.keys())}"
            )
        self.dataset_id = dataset_id
        self._raw_to_global = DATASET_LABEL_MAPS[dataset_id]

        if active_global_labels is None:
            # Use all global labels that this dataset actually emits
            active = sorted(
                {v for v in self._raw_to_global.values()},
                key=lambda lbl: GLOBAL_LABEL_TO_IDX[lbl],
            )
            self.active_global_labels = active
        else:
            self.active_global_labels = active_global_labels

        # Compact index within the active subset (0, 1, 2, ...)
        self._global_to_compact: Dict[str, int] = {
            lbl: idx for idx, lbl in enumerate(self.active_global_labels)
        }
        self.n_classes = len(self.active_global_labels)

    # ------------------------------------------------------------------
    def translate(self, raw_label: str) -> Optional[int]:
        """
        Translate a raw dataset label string to a compact integer index.

        Returns None if the label is not part of the active label set
        (caller should skip that sample).
        """
        raw_lower = raw_label.lower()
        global_lbl = self._raw_to_global.get(raw_lower)
        if global_lbl is None:
            # Try direct match (dataset already uses canonical names)
            global_lbl = self._raw_to_global.get(raw_label)
        if global_lbl is None:
            logger.debug(
                "Dataset '%s': raw label '%s' not in map — sample will be skipped",
                self.dataset_id, raw_label,
            )
            return None
        return self._global_to_compact.get(global_lbl)

    def class_names(self) -> List[str]:
        """Return the ordered list of active class names (compact indices)."""
        return list(self.active_global_labels)

    def __repr__(self) -> str:
        return (
            f"LabelSchema(dataset_id='{self.dataset_id}', "
            f"n_classes={self.n_classes}, "
            f"labels={self.active_global_labels})"
        )


# ---------------------------------------------------------------------------
# Helpers for building a shared schema across multiple datasets
# ---------------------------------------------------------------------------

def infer_shared_label_schema(dataset_ids: List[str]) -> Tuple[List[str], Dict[str, LabelSchema]]:
    """
    Given a list of dataset IDs, compute the union of all labels present
    across those datasets and return:

    * shared_labels : List[str] — ordered global label list for the union
    * schemas : Dict[str, LabelSchema] — per-dataset schema objects all
                sharing the same active_global_labels list so that
                compact indices are consistent across datasets.
    """
    union: set = set()
    for did in dataset_ids:
        if did not in DATASET_LABEL_MAPS:
            raise KeyError(f"No label map for '{did}'")
        union.update(DATASET_LABEL_MAPS[did].values())

    shared_labels = sorted(union, key=lambda lbl: GLOBAL_LABEL_TO_IDX.get(lbl, 999))

    schemas: Dict[str, LabelSchema] = {
        did: LabelSchema(did, active_global_labels=shared_labels)
        for did in dataset_ids
    }
    return shared_labels, schemas