"""
training/trainer.py
--------------------
Training loop for EEGNet.

Kept deliberately simple: plain PyTorch, no Lightning, no framework magic.
A researcher can read this top to bottom in a few minutes and modify any
part of it.

Responsibilities:
  - One training run (train_model)
  - Epoch loop with optional progress logging
  - Returns the trained model and a loss history for later plotting
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    n_epochs: int,
    learning_rate: float,
    device: torch.device,
    val_loader: Optional[DataLoader] = None,
    checkpoint_path: Optional[Path] = None,
) -> Dict[str, List[float]]:
    """
    Train a binary EEGNet classifier.

    Parameters
    ----------
    model : nn.Module
        The EEGNet model.  Must already be moved to device.
    train_loader : DataLoader
        Yields (x, y) batches where x is [batch, 1, C, T] and y is float.
    n_epochs : int
    learning_rate : float
    device : torch.device
    val_loader : DataLoader, optional
        If provided, validation loss is computed at the end of every epoch.
    checkpoint_path : Path, optional
        If provided, the model state dict is saved here after the final epoch.

    Returns
    -------
    history : dict with keys 'train_loss' and (optionally) 'val_loss',
              each a list of per-epoch floats.
    """
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history: Dict[str, List[float]] = {"train_loss": []}
    if val_loader is not None:
        history["val_loss"] = []

    for epoch in range(1, n_epochs + 1):
        train_loss = _run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        history["train_loss"].append(train_loss)

        if val_loader is not None:
            val_loss = _run_epoch(model, val_loader, criterion, optimizer=None, device=device, train=False)
            history["val_loss"].append(val_loss)
            logger.info(
                "Epoch %d/%d — train_loss: %.4f  val_loss: %.4f",
                epoch, n_epochs, train_loss, val_loss,
            )
        else:
            logger.info("Epoch %d/%d — train_loss: %.4f", epoch, n_epochs, train_loss)

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        logger.info("Checkpoint saved to %s", checkpoint_path)

    return history


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    train: bool,
) -> float:
    """Run one full pass over the loader. Returns mean loss."""
    model.train(train)
    total_loss = 0.0
    n_batches = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            preds = model(x)          # [batch] in (0, 1)
            loss = criterion(preds, y)

            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)
