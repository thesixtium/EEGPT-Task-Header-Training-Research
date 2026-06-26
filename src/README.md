# run_mi_experiment — Pipeline Overview

## What It Does

Runs a Leave-Subjects-Out (LSO) P300/ERP classification experiment across multiple MOABB datasets using a pretrained EEGPT backbone. One subject per dataset is held out, a shared base model is trained on all remaining subjects, then progressive session-based adaptation is applied to each held-out subject.

---

## Preprocessing

Raw EEG is pulled via MOABB and passed through `EEGPTPreprocessor`:

1. **Channel selection** — dataset channels are mapped to the 58-electrode canonical EEGPT vocabulary; unrecognised or reference channels are dropped.
2. **Resampling** — signals are resampled to 256 Hz using linear interpolation.
3. **Segmentation** — continuous recordings are cut into non-overlapping 4-second (1024-sample) windows.
4. **Baseline correction** — per-channel temporal mean is subtracted from each window.
5. **Amplitude clipping** — values outside ±800 µV are clipped to reject artefacts.

Processed samples are cached to `data/dataset_cache/` as `.pt` files to avoid redundant reprocessing.

---

## Model Training

**Frozen:** The entire EEGPT transformer backbone (`target_encoder`) is frozen — weights are loaded from the pretrained checkpoint and `requires_grad=False` is set throughout. It runs in `eval()` mode even during training.

**Trained:** A lightweight channel projection conv (`chan_conv`, Conv1d) and a three-layer FC classification head (`fc1 → fc2 → fc3`) are trained. The backbone produces `[B, 16, 4, 512]` which is flattened to `[B, 32768]` and passed through the head.

**Optimiser:** AdamW with a OneCycleLR scheduler (`max_lr=4e-4`, `pct_start=0.2`). Training runs for 100 base epochs, then 10 adaptation epochs per progressive session stage.

---

## Task Head

**Currently used — three-layer FC:**
```
fc1: 32768 → 512  (GELU + Dropout 0.5 + max_norm=1)
fc2:   512 → 128  (GELU + Dropout 0.5 + max_norm=1)
fc3:   128 → n_classes              (max_norm=0.25)
```

**Original head (commented out):**
```
linear_probe1: 2048 → 16   (per patch, max_norm=1)
linear_probe2: 16×16 → n_classes   (max_norm=0.25)
```
The original head operated on per-patch features before flattening all patch dims, giving a much smaller parameter count. The current head flattens all spatial/temporal patch structure first and uses a deeper MLP, trading compactness for representational capacity.

---

## Graphs

Training metrics are logged to CSV via PyTorch Lightning's `CSVLogger`. After training, `metrics_display()` in `metricMethods.py` reads the latest version folder and generates an 11-panel dashboard saved as a `.png`, covering loss, accuracy, balanced accuracy, Cohen's kappa, F1 (macro/micro/weighted), learning rate, and data statistics.