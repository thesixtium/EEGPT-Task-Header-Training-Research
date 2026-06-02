# EEG Motor Imagery Framework

A generalized training and evaluation framework built around the **unmodified** pretrained `genericEEGPTModel` backbone.

---

## Design principle

> The framework adapts **around** the model. The model never changes.

`genericEEGPTModel` expects `[n_channels, 1024]` tensors (4 s × 256 Hz). Every other concern — dataset loading, channel mapping, resampling, label unification, cross-subject splitting, progressive adaptation — is handled by the framework layers below the model interface.

---

## File structure

```
framework/
├── __init__.py                 Clean public API
├── canonical_channels.py       58-channel EEGPT vocabulary + mapping utilities
├── label_schema.py             Global label vocabulary + per-dataset translation
├── preprocessing.py            Resample → segment → normalise pipeline
├── eeg_dataset.py              EEGSample / EEGSampleDataset with full metadata
├── dataset_registry.py         Abstract loader + MOABB adapter
├── multi_dataset_loader.py     ConcatDataLoader, MultiChannelDataLoader, splits
├── leave_subjects_out.py       LSO evaluation + progressive adaptation protocol
├── analytics.py                Metrics, confusion matrices, adaptation curves
├── experiment_runner.py        Top-level orchestration
└── run_mi_experiment.py        Usage examples (original / LSO / joint / custom)
```

---

## Module responsibilities

### `canonical_channels.py`
Defines the 58-electrode canonical space the pretrained backbone was trained on.

```python
from framework import CANONICAL_CHANNELS, resolve_channel_subset

# Map dataset channels to canonical space
valid_names, ds_indices, canon_indices = resolve_channel_subset(['C3', 'Cz', 'C4', 'EEG-P3'])
# → (['C3', 'CZ', 'C4', 'P3'], [0, 1, 2, 3], [28, 29, 30, 46])
```

Handles:
- Case normalisation (`cz` → `CZ`)
- Legacy electrode name aliases (`T3` → `T7`, `T4` → `T8`)
- `EEG-` prefix stripping
- Unknown / reference electrode rejection

---

### `label_schema.py`
Defines a 5-class global vocabulary (`left_hand`, `right_hand`, `feet`, `rest`, `tongue`) and maps each dataset's raw string labels into compact integer indices.

```python
from framework import LabelSchema, infer_shared_label_schema

# Single dataset
schema = LabelSchema('BNCI2014_004')
compact_idx = schema.translate('left_hand')  # → 0

# Multi-dataset — compute union label space
shared_labels, schemas = infer_shared_label_schema(['BNCI2014_004', 'BNCI2015_001'])
# shared_labels = ['left_hand', 'right_hand', 'feet']
# schemas['BNCI2014_004'].translate('right_hand') → 1 (consistent across datasets)
```

Adding a new dataset requires only one entry in `DATASET_LABEL_MAPS`.

---

### `preprocessing.py`
Composable pipeline stages that convert arbitrary EEG into `[n_channels, 1024]` tensors.

```python
from framework import EEGPTPreprocessor

proc = EEGPTPreprocessor(
    dataset_channel_names=['C3', 'Cz', 'C4', 'Pz'],
    source_sfreq=512,           # dataset native rate; resampled → 256 Hz
    clip_amplitude=800.0,       # optional artefact clipping
)
# X: np.ndarray [n_trials, n_raw_channels, n_timepoints]
X_proc = proc.process_epochs(X)  # Tensor [n_trials, 4, 1024]
```

Individual stages (`ChannelSelector`, `Resampler`, `Segmenter`, `BaselineCorrector`, `AmplitudeClipper`) can be composed independently.

---

### `eeg_dataset.py`
Every sample carries the full metadata schema:

```python
{
    "eeg":           Tensor[n_channels, 1024],
    "channel_names": List[str],          # canonical names
    "label":         int,                # compact index
    "subject_id":    int,
    "session_id":    int,
    "dataset_id":    str,
}
```

`EEGSampleDataset` exposes filtering helpers:

```python
ds.filter_by_subject(3)      # all trials for subject 3
ds.exclude_subject(3)        # all trials except subject 3
ds.filter_by_session(0)      # first session only
ds.get_subject_ids()         # sorted list of subject IDs
EEGSampleDataset.concat([d1, d2])  # merge datasets
```

---

### `dataset_registry.py`
`MoabbDatasetLoader` is the generalised replacement for the original `moabbMotorImageryDataLoader`. It loads **all subjects and sessions** in one call:

```python
from moabb.datasets import BNCI2014_004
from framework import MoabbDatasetLoader, LabelSchema

schema = LabelSchema('BNCI2014_004')
loader = MoabbDatasetLoader(
    moabb_dataset=BNCI2014_004(),
    dataset_id='BNCI2014_004',
    label_schema=schema,
    n_classes=schema.n_classes,
    tmax=4.0,
    resample=256,
)
ds = loader.load_all_subjects()   # EEGSampleDataset, all subjects × sessions
```

Custom non-MOABB datasets subclass `BaseDatasetLoader` and implement `load_all_subjects()`.

---

### `multi_dataset_loader.py`

**Shared channel layout** (most common):

```python
from framework import ConcatDataLoader, train_val_test_split

train_set, val_set, test_set = train_val_test_split(combined_dataset)
loader = ConcatDataLoader([train_set], batch_size=32, balanced=True)
trainer.fit(model, loader.get_loader(), ...)
```

**Heterogeneous channel layouts** (each dataset has different electrodes):

```python
from framework import MultiChannelDataLoader

loader = MultiChannelDataLoader({'BNCI2014_004': ds1, 'BNCI2015_001': ds2})
for eeg_batch, label_batch, channel_names in loader:
    # channel_names changes per batch — pass to model's chan_id preparation
    ...
```

---

### `leave_subjects_out.py`
The full LSO + progressive adaptation protocol.

```
For each held-out subject S:
  Stage 0:   Train on all other subjects   → evaluate on S_test (zero-shot)
  Stage 1:   Adapt on S_session_0          → evaluate on S_test
  Stage 2:   Adapt on S_session_0+1        → evaluate on S_test
  ...
  Stage K:   Adapt on all but S_test       → evaluate on S_test

Key: S_test is FIXED throughout. Adaptation is WARM-START (continues from prior stage).
```

```python
from framework import LeaveSubjectsOutEvaluator

evaluator = LeaveSubjectsOutEvaluator(
    dataset=ds,
    model_factory=my_factory,
    channel_names=channel_names,
    n_classes=n_classes,
    base_epochs=20,
    adapt_epochs=10,
    batch_size=32,
    checkpoints_dir=Path('checkpoints'),
    logs_dir=Path('logs'),
)
results = evaluator.run()   # List[SubjectResult]
```

---

### `analytics.py`
All metrics logged per stage:

| Metric | Key |
|--------|-----|
| Accuracy | `valid_accuracy` |
| Balanced accuracy | `valid_balanced_accuracy` |
| F1 weighted | `valid_f1_weighted` |
| F1 macro | `valid_f1_macro` |
| F1 micro | `valid_f1_micro` |
| Matthews CC | `valid_mcc` |
| Cohen's κ | `valid_cohen_kappa` |

```python
from framework import ExperimentLogger, ConfusionMatrixAnalyser, AdaptationCurveAnalyser

logger = ExperimentLogger('results', 'my_experiment')
logger.save_summary_csv(all_subject_results)
logger.generate_adaptation_plots(all_subject_results)
```

**Output structure:**
```
results/my_experiment/
├── experiment_config.json
├── per_subject/
│   ├── subject_1.json
│   └── subject_2.json
└── summary/
    ├── all_results.csv
    ├── adaptation_curves/
    │   ├── valid_accuracy.png
    │   ├── valid_balanced_accuracy.png
    │   └── valid_mcc.png
    └── confusion_matrices/
        ├── subject1_stage0.png
        └── subject1_stage1.png
```

---

## Running experiments

```bash
# Reproduce original per-dataset training (unchanged)
python -m framework.run_mi_experiment original

# Full LSO evaluation with progressive adaptation
python -m framework.run_mi_experiment lso

# Multi-dataset joint baseline (no LSO)
python -m framework.run_mi_experiment joint

# Custom dataset skeleton
python -m framework.run_mi_experiment custom
```

Or programmatically:

```python
from framework import ExperimentConfig, run_experiment
from moabb.datasets import BNCI2014_004, BNCI2015_001

cfg = ExperimentConfig(
    experiment_name='mi_lso',
    base_model_path='checkpoints/eegpt_mcae_58chs_4s_large4E.ckpt',
    datasets=[BNCI2014_004(), BNCI2015_001()],
    dataset_ids=['BNCI2014_004', 'BNCI2015_001'],
    max_lr=4e-4,
    base_epochs=20,
    adapt_epochs=10,
    batch_size=32,
    lso_enabled=True,
)
run_experiment(cfg)
```

---

## Adding a new dataset

1. Add an entry to `DATASET_LABEL_MAPS` in `label_schema.py`:

```python
DATASET_LABEL_MAPS['MyDataset'] = {
    'left':  'left_hand',
    'right': 'right_hand',
    'feet':  'feet',
}
```

2. If it is MOABB-compatible, use `MoabbDatasetLoader` directly.

3. If custom, subclass `BaseDatasetLoader`:

```python
class MyLoader(BaseDatasetLoader):
    def load_all_subjects(self) -> EEGSampleDataset: ...
    def get_channel_names(self) -> List[str]: ...
    def get_n_classes(self) -> int: ...
    def get_dataset_id(self) -> str: ...
```

4. Add to `ExperimentConfig.datasets` and `dataset_ids`. Done.

---

## Compatibility guarantee

`genericEEGPTModel` is **never modified**. The framework's preprocessing pipeline always produces tensors shaped `[n_channels, 1024]` at 256 Hz regardless of the source dataset's native sampling rate or channel count. The model's `prepare_chan_ids` method receives the resolved canonical channel names for each dataset subset, preserving full compatibility with the pretrained backbone.
