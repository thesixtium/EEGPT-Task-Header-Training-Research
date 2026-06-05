"""
models/eegnet.py
----------------
EEGNet for binary P300 classification.

Faithful batched implementation of the EEGNet architecture from the reference
code, with two documented changes needed for a research baseline:

  1. Batch support via explicit reshape  (see "Changes" note below)
  2. n_channels / n_timepoints as constructor arguments

Original paper:
  Lawhern et al., "EEGNet: A Compact Convolutional Neural Network for
  EEG-based Brain-Computer Interfaces", J. Neural Eng. 2018.

Input  : Tensor[batch, 1, n_channels, n_timepoints]
Output : Tensor[batch], probabilities in (0, 1)
Loss   : BCELoss

Changes from the reference implementation
------------------------------------------
1. Batch support.

   The reference uses `x.permute(0, 3, 1, 2)` after layer 1 and passes
   the result directly into conv2 (in_channels=1).  This is valid only for
   batch_size=1: the batch and time axes accidentally merge into the "batch"
   dimension that conv2 sees.  For batch_size > 1 it silently uses wrong
   weights.

   Fix: after the permute, explicitly reshape [batch, T', 16, C] →
   [batch×T', 1, 16, C] so conv2 always receives single-channel inputs.
   The batch axis is restored before the FC layer.  The computation is
   mathematically identical to the original for batch_size=1.

2. AdaptiveMaxPool2d for the final pool.

   The original pooling3 = MaxPool2d((2, 4)) requires the spatial
   dimension entering it to be at least 4 wide.  For datasets with fewer
   than ~32 channels the input collapses too narrow and PyTorch raises an
   error.

   Fix: replace pooling3 with AdaptiveMaxPool2d((2, 2)), which always
   outputs a 2×2 map regardless of input size.  For the original 64-channel
   configuration the output is the same size as the original (2×2 vs 2×4 —
   the FC weights simply adjust to the new size, which is computed
   dynamically anyway).  The architecture is otherwise identical.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):

    def __init__(self, n_channels: int = 64, n_timepoints: int = 120) -> None:
        super().__init__()

        # ── Layer 1: temporal convolution ─────────────────────────────────
        # Convolves across time; each of the 16 filters spans 64 samples.
        # Input:  [batch, 1,  n_channels, n_timepoints]
        # Output: [batch, 16, n_channels, n_timepoints - 63]
        self.conv1      = nn.Conv2d(1, 16, (1, 64), padding=0)
        self.batchnorm1 = nn.BatchNorm2d(16, affine=False)

        # ── Layer 2: spatial + temporal convolution ────────────────────────
        # Operates on [batch×T', 1, 16, n_channels] after reshape (see forward).
        self.padding1   = nn.ZeroPad2d((16, 17, 0, 1))
        self.conv2      = nn.Conv2d(1, 4, (2, 32))
        self.batchnorm2 = nn.BatchNorm2d(4, affine=False)
        self.pooling2   = nn.MaxPool2d(2, 4)

        # ── Layer 3: separable convolution ────────────────────────────────
        self.padding2   = nn.ZeroPad2d((2, 1, 4, 3))
        self.conv3      = nn.Conv2d(4, 4, (8, 4))
        self.batchnorm3 = nn.BatchNorm2d(4, affine=False)
        # AdaptiveMaxPool2d replaces MaxPool2d((2, 4)) — see module docstring.
        self.pooling3   = nn.AdaptiveMaxPool2d((2, 2))

        # ── FC layer — built lazily on first forward call ─────────────────
        # The flattened feature size depends on n_channels and n_timepoints.
        # We compute it from a dummy pass rather than hardcoding "4 * 2 * 7"
        # which only holds for the original 64-channel / 120-timepoint config.
        self.fc1: nn.Linear | None = None
        self._fc_built = False

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor[batch, 1, n_channels, n_timepoints]

        Returns
        -------
        Tensor[batch] — Target probability in (0, 1).
        """
        # ── Layer 1 ───────────────────────────────────────────────────────
        x = F.elu(self.conv1(x))
        x = self.batchnorm1(x)
        x = F.dropout(x, p=0.25, training=self.training)
        # x: [batch, 16, n_channels, T']   where T' = n_timepoints - 63

        bs, c16, n_ch, n_t = x.shape

        # Permute → explicit batch reshape for layer 2 (see module docstring)
        x = x.permute(0, 3, 1, 2)              # [batch, T', 16, n_channels]
        x = x.reshape(bs * n_t, 1, c16, n_ch)  # [batch×T', 1,  16, n_channels]

        # ── Layer 2 ───────────────────────────────────────────────────────
        x = self.padding1(x)
        x = F.elu(self.conv2(x))
        x = self.batchnorm2(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.pooling2(x)

        # ── Layer 3 ───────────────────────────────────────────────────────
        x = self.padding2(x)
        x = F.elu(self.conv3(x))
        x = self.batchnorm3(x)
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.pooling3(x)                    # [batch×T', 4, 2, 2]

        # ── Restore batch, flatten, classify ──────────────────────────────
        x = x.reshape(bs, -1)                   # [batch, fc_input_size]

        if not self._fc_built:
            self.fc1 = nn.Linear(x.shape[1], 1).to(x.device)
            self._fc_built = True

        x = torch.sigmoid(self.fc1(x))          # [batch, 1]
        return x.squeeze(1)                     # [batch]
