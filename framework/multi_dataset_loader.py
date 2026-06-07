"""
multi_dataset_loader.py
-----------------------
Builds DataLoader objects that can draw from multiple datasets
simultaneously, handling the fact that different datasets expose
different channel subsets.

Two strategies are offered:

1. ConcatLoader (recommended)
   Uses a standard ConcatDataset. All datasets must have been preprocessed
   with the SAME channel layout (or a channel-selection pass is applied
   so every sample has the same canonical channels in the same order).
   The simplest and most efficient approach when channel sets overlap.

2. MultiChannelLoader
   Handles truly heterogeneous channel subsets by keeping each dataset in
   its own DataLoader and interleaving batches. Each batch is still
   homogeneous within itself (all samples share a channel layout) but
   consecutive batches may come from different channel layouts.
   The model sees [batch, n_channels_i, 1024] per step and the caller
   must pass the appropriate channel_names per batch.
"""

from __future__ import annotations
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
import math
import itertools
import logging

import torch
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler

from .eeg_dataset import EEGSampleDataset, collate_same_channels, collate_with_metadata

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utility: build a class-balanced sampler
# ---------------------------------------------------------------------------

def make_balanced_sampler(dataset: EEGSampleDataset) -> WeightedRandomSampler:
    """
    Return a WeightedRandomSampler that up-samples minority classes so
    that every class is seen equally often per epoch.
    """
    labels = [s.label for s in dataset.samples]
    class_counts = torch.bincount(torch.tensor(labels, dtype=torch.long))
    weights_per_class = 1.0 / class_counts.float()
    sample_weights = weights_per_class[torch.tensor(labels)]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# Strategy 1 — ConcatLoader (shared channel layout)
# ---------------------------------------------------------------------------

class ConcatDataLoader:
    """
    Merge multiple EEGSampleDatasets that share the same channel layout
    into a single DataLoader.

    Use this when all datasets have been mapped to the same canonical
    channel subset (e.g. the intersection of their channel sets).

    Parameters
    ----------
    datasets : List[EEGSampleDataset]
    batch_size : int
    shuffle : bool
    balanced : bool — if True, use class-balanced sampling
    num_workers : int
    with_metadata : bool — if True use collate_with_metadata, else collate_same_channels
    """

    def __init__(
        self,
        datasets: List[EEGSampleDataset],
        batch_size: int = 32,
        shuffle: bool = True,
        balanced: bool = False,
        num_workers: int = 0,
        with_metadata: bool = False,
    ) -> None:
        combined = EEGSampleDataset.concat(datasets)
        collate_fn = collate_with_metadata if with_metadata else collate_same_channels

        sampler = make_balanced_sampler(combined) if balanced else None
        self._loader = DataLoader(
            combined,
            batch_size=batch_size,
            shuffle=(shuffle and sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=False,
        )
        self.steps_per_epoch = math.ceil(len(combined) / batch_size)
        self.n_samples = len(combined)
        self.channel_names: List[str] = combined.get_channel_names()

    def get_loader(self) -> DataLoader:
        return self._loader

    def __len__(self) -> int:
        return self.steps_per_epoch


# ---------------------------------------------------------------------------
# Strategy 2 — MultiChannelLoader (heterogeneous channel sets)
# ---------------------------------------------------------------------------

class MultiChannelDataLoader:
    """
    Interleave batches from multiple per-dataset DataLoaders where each
    dataset may use a different channel subset.

    At each training step the caller receives a batch alongside its
    channel_names list; it must forward these to genericEEGPTModel which
    handles variable channel subsets natively via prepare_chan_ids.

    Parameters
    ----------
    dataset_loaders : Dict[str, EEGSampleDataset]
        Maps dataset_id → EEGSampleDataset.
    batch_size : int
    shuffle : bool
    balanced : bool — class-balanced sampling per dataset
    num_workers : int
    """

    def __init__(
        self,
        dataset_loaders: Dict[str, EEGSampleDataset],
        batch_size: int = 32,
        shuffle: bool = True,
        balanced: bool = True,
        num_workers: int = 0,
    ) -> None:
        self._loaders: Dict[str, DataLoader] = {}
        self._channel_names: Dict[str, List[str]] = {}

        for did, ds in dataset_loaders.items():
            sampler = make_balanced_sampler(ds) if balanced else None
            self._loaders[did] = DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=(shuffle and sampler is None),
                sampler=sampler,
                num_workers=num_workers,
                collate_fn=collate_with_metadata,
                drop_last=False,
            )
            self._channel_names[did] = ds.get_channel_names()
            logger.info(
                "MultiChannelDataLoader: %s — %d samples, %d channels",
                did, len(ds), len(self._channel_names[did]),
            )

        # Steps-per-epoch = number of batches across all loaders
        self.steps_per_epoch = sum(len(ldr) for ldr in self._loaders.values())

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, List[str]]]:
        """
        Yield (eeg_batch, label_batch, channel_names) tuples by
        round-robining across dataset loaders.
        """
        iters = [iter(ldr) for ldr in self._loaders.values()]
        chan_lists = list(self._channel_names.values())

        for batch, chans in itertools.chain.from_iterable(
            zip(it, itertools.repeat(ch)) for it, ch in zip(iters, chan_lists)
        ):
            yield batch['eeg'], batch['label'], chans

    def __len__(self) -> int:
        return self.steps_per_epoch

    def get_loader(self, dataset_id: str) -> DataLoader:
        return self._loaders[dataset_id]

    def get_channel_names(self, dataset_id: str) -> List[str]:
        return self._channel_names[dataset_id]


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def train_val_test_split(
    dataset: EEGSampleDataset,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[EEGSampleDataset, EEGSampleDataset, EEGSampleDataset]:
    """
    Random stratified split of an EEGSampleDataset.
    Preserves class balance in each split.
    """
    from sklearn.model_selection import train_test_split as sk_split

    indices = list(range(len(dataset.samples)))
    labels = [s.label for s in dataset.samples]

    if test_ratio > 0:
        train_idx, test_idx = sk_split(
            indices, test_size=test_ratio, stratify=labels, random_state=seed
        )
    else:
        train_idx, test_idx = indices, []

    train_labels = [labels[i] for i in train_idx]
    val_effective_ratio = val_ratio / (1 - test_ratio) if test_ratio < 1 else 0.0
    if val_effective_ratio > 0:
        train_idx, val_idx = sk_split(
            train_idx,
            test_size=val_effective_ratio,
            stratify=train_labels,
            random_state=seed,
        )
    else:
        train_idx, val_idx = train_idx, []

    def subset(idxs):
        return EEGSampleDataset([dataset.samples[i] for i in idxs])

    return subset(train_idx), subset(val_idx), subset(test_idx)


def subject_split(
    dataset: EEGSampleDataset,
    held_out_subject: int,
) -> Tuple[EEGSampleDataset, EEGSampleDataset]:
    """
    Return (train_set, test_set) where test_set contains only
    held_out_subject samples and train_set contains all others.
    """
    train = dataset.exclude_subject(held_out_subject)
    test = dataset.filter_by_subject(held_out_subject)
    return train, test