"""
eeg_dataset.py
--------------
PyTorch Dataset implementations that carry the full per-sample metadata
schema required by the framework:

    {
        "eeg":          Tensor[n_channels, 1024],
        "channel_names": List[str],
        "label":        int,
        "subject_id":   int,
        "session_id":   int,
        "dataset_id":   str,
    }

Two classes are provided:

* EEGSampleDataset — stores every sample in RAM as a list of dicts.
  Suitable for small-to-medium datasets.

* LazyEEGDataset — stores raw arrays and processes on demand.
  Better for large datasets where preprocessing fits in a single pass.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import torch
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Sample dataclass (optional convenience; Dataset stores plain dicts too)
# ---------------------------------------------------------------------------

@dataclass
class EEGSample:
    eeg: torch.Tensor          # [n_channels, 1024]
    channel_names: List[str]   # len == n_channels, canonical names
    label: int
    subject_id: int
    session_id: int
    dataset_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            'eeg': self.eeg,
            'channel_names': self.channel_names,
            'label': self.label,
            'subject_id': self.subject_id,
            'session_id': self.session_id,
            'dataset_id': self.dataset_id,
        }


# ---------------------------------------------------------------------------
# In-memory dataset
# ---------------------------------------------------------------------------

class EEGSampleDataset(Dataset):
    """
    A simple in-memory dataset of EEGSample objects.

    Supports optional per-sample transforms applied at __getitem__ time.
    """

    def __init__(
        self,
        samples: List[EEGSample],
        transform=None,
    ) -> None:
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx].to_dict()
        if self.transform:
            sample = self.transform(sample)
        return sample

    def filter_by_subject(self, subject_id: int) -> 'EEGSampleDataset':
        kept = [s for s in self.samples if s.subject_id == subject_id]
        return EEGSampleDataset(kept, self.transform)

    def filter_by_session(self, session_id: int) -> 'EEGSampleDataset':
        kept = [s for s in self.samples if s.session_id == session_id]
        return EEGSampleDataset(kept, self.transform)

    def exclude_subject(self, subject_id: int) -> 'EEGSampleDataset':
        kept = [s for s in self.samples if s.subject_id != subject_id]
        return EEGSampleDataset(kept, self.transform)

    def exclude_session(self, session_id: int) -> 'EEGSampleDataset':
        kept = [s for s in self.samples if s.session_id != session_id]
        return EEGSampleDataset(kept, self.transform)

    def get_subject_ids(self) -> List[int]:
        return sorted({s.subject_id for s in self.samples})

    def get_session_ids(self) -> List[int]:
        return sorted({s.session_id for s in self.samples})

    def get_channel_names(self) -> List[str]:
        """
        Return channel names.  All samples in the dataset must share the
        same channel layout (enforced at construction time by the loaders).
        """
        if not self.samples:
            return []
        return self.samples[0].channel_names

    def class_distribution(self) -> Dict[int, int]:
        dist: Dict[int, int] = {}
        for s in self.samples:
            dist[s.label] = dist.get(s.label, 0) + 1
        return dist

    @classmethod
    def concat(cls, datasets: List['EEGSampleDataset']) -> 'EEGSampleDataset':
        all_samples = []
        for d in datasets:
            all_samples.extend(d.samples)
        return cls(all_samples)


# ---------------------------------------------------------------------------
# Collate function for variable-channel batches
# ---------------------------------------------------------------------------

def collate_same_channels(batch: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Default collate: assumes all samples in the batch share the same
    channel layout (guaranteed when each DataLoader draws from a
    single-dataset EEGSampleDataset).

    Returns
    -------
    x : Tensor[batch, n_channels, 1024]
    y : Tensor[batch]  (long)
    """
    eeg = torch.stack([item['eeg'] for item in batch], dim=0)
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    return eeg, labels


def collate_with_metadata(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate that preserves all metadata fields.
    Useful for evaluation and analytics passes.
    """
    eeg = torch.stack([item['eeg'] for item in batch], dim=0)
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.long)
    return {
        'eeg': eeg,
        'label': labels,
        'channel_names': batch[0]['channel_names'],   # shared within batch
        'subject_id': [item['subject_id'] for item in batch],
        'session_id': [item['session_id'] for item in batch],
        'dataset_id': [item['dataset_id'] for item in batch],
    }
