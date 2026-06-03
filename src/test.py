"""
test_datasets.py
----------------
Validates every dataset in your experiment config BEFORE running the full
experiment.  Catches the most common failure modes early:

  1. Dataset ID missing from label_schema.py → DATASET_LABEL_MAPS
  2. MOABB download / data-access failure
  3. No labels survive the LabelSchema translation (all samples skipped)
  4. Channel mapping: zero channels resolve to the canonical EEGPT space
  5. Preprocessing failure (resampling, segmentation, shape check)
  6. Empty dataset after full pipeline

Run from your project root:
    python test_datasets.py

Each dataset is tested on subject 1 only to keep the check fast.
A final summary table is printed at the end.
"""

import logging
import sys
import traceback
from dataclasses import dataclass, field
from typing import List, Optional

# ── configure logging ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,          # suppress MOABB noise during tests
    format='%(levelname)-8s  %(name)s  %(message)s',
)
# Keep our own logger at INFO so test progress is visible
log = logging.getLogger('test_datasets')
log.setLevel(logging.INFO)

# ── imports ──────────────────────────────────────────────────────────────────
from moabb.datasets import (
    BNCI2014_009
)

# ---------------------------------------------------------------------------
# Edit this list to match your run_mi_experiment.py config exactly.
# Each tuple is (moabb_dataset_object, dataset_id_string).
# ---------------------------------------------------------------------------
DATASETS_TO_TEST = [
    (BNCI2014_009(),  'BNCI2014_009')
]

# Test subject — just the first available subject per dataset to keep it fast.
TEST_SUBJECT_INDEX = 0   # index into dataset.subject_list, NOT the subject number


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DatasetTestResult:
    dataset_id: str
    passed: bool = False
    skipped: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Populated on success
    n_samples: int = 0
    n_channels: int = 0
    n_classes: int = 0
    channel_names: List[str] = field(default_factory=list)
    label_distribution: dict = field(default_factory=dict)
    source_sfreq: Optional[int] = None
    eeg_shape: Optional[tuple] = None


# ---------------------------------------------------------------------------
# Helper: resolve the native sample rate for a dataset
# ---------------------------------------------------------------------------

def _get_source_sfreq(dataset_id: str, moabb_dataset) -> int:
    """
    Return the native recording sample rate for *dataset_id*.

    Resolution order:
      1. DATASET_SAMPLE_RATES table in label_schema.py  (preferred — always correct)
      2. moabb_dataset.sfreq attribute                  (not always present)
      3. Hard-coded fallback of 256 Hz                  (last resort)

    If the rate came from the fallback a warning is emitted so it is visible
    in the test output.
    """
    from framework.label_schema import DATASET_SAMPLE_RATES

    rate = DATASET_SAMPLE_RATES.get(dataset_id)
    if rate is not None:
        return int(rate)

    rate = getattr(moabb_dataset, 'sfreq', None)
    if rate is not None:
        return int(rate)

    log.warning(
        "  [!] %s — no entry in DATASET_SAMPLE_RATES and no .sfreq attribute; "
        "falling back to 256 Hz. Add the correct rate to DATASET_SAMPLE_RATES "
        "in label_schema.py.",
        dataset_id,
    )
    return 256


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_label_schema(dataset_id: str, result: DatasetTestResult) -> bool:
    """Check 1 — dataset_id registered in DATASET_LABEL_MAPS."""
    try:
        from framework.label_schema import DATASET_LABEL_MAPS, LabelSchema
        if dataset_id not in DATASET_LABEL_MAPS:
            result.errors.append(
                f"'{dataset_id}' not found in label_schema.py → DATASET_LABEL_MAPS. "
                f"Available keys: {sorted(DATASET_LABEL_MAPS.keys())}"
            )
            return False
        schema = LabelSchema(dataset_id)
        result.n_classes = schema.n_classes
        log.info("  [✓] Label schema — %d classes: %s", schema.n_classes, schema.class_names())
        return True
    except Exception as exc:
        result.errors.append(f"LabelSchema error: {exc}")
        return False


def check_moabb_download(moabb_dataset, dataset_id: str, result: DatasetTestResult) -> Optional[int]:
    """
    Check 2 — MOABB can download / access data and return the test subject number.
    Returns the actual subject number (not index) or None on failure.
    """
    try:
        subject_list = moabb_dataset.subject_list
        if not subject_list:
            result.errors.append("subject_list is empty.")
            return None

        subject = subject_list[min(TEST_SUBJECT_INDEX, len(subject_list) - 1)]

        from moabb.paradigms import P300
        from framework.label_schema import DATASET_LABEL_MAPS

        # Infer n_classes for the paradigm from the label map
        n_classes = len(DATASET_LABEL_MAPS[dataset_id])

        paradigm = P300(
            tmin=0.0,
            tmax=1.0
        )
        X, y, metadata = paradigm.get_data(
            dataset=moabb_dataset,
            subjects=[subject],
            return_epochs=False,
        )

        # Use the registry table rather than moabb_dataset.sfreq
        result.source_sfreq = _get_source_sfreq(dataset_id, moabb_dataset)

        log.info(
            "  [✓] MOABB download — subject %s, raw shape %s, "
            "%d trials, sfreq=%s Hz",
            subject, X.shape, len(y), result.source_sfreq,
        )
        return subject

    except Exception as exc:
        result.errors.append(f"MOABB data access failed: {exc}\n{traceback.format_exc()}")
        return None


def check_label_translation(moabb_dataset, dataset_id: str, subject: int,
                             result: DatasetTestResult) -> bool:
    """Check 3 — at least some labels translate through LabelSchema."""
    try:
        from moabb.paradigms import P300
        from framework.label_schema import DATASET_LABEL_MAPS, LabelSchema

        n_classes = len(DATASET_LABEL_MAPS[dataset_id])
        paradigm = P300(tmin=0.0, tmax=1.0)
        _, y, _ = paradigm.get_data(dataset=moabb_dataset, subjects=[subject],
                                     return_epochs=False)

        schema = LabelSchema(dataset_id)
        raw_labels = set(y)
        translated = {lbl: schema.translate(lbl) for lbl in raw_labels}
        surviving = {k: v for k, v in translated.items() if v is not None}
        dropped   = {k: v for k, v in translated.items() if v is None}

        if dropped:
            result.warnings.append(
                f"Labels not in schema (will be skipped): {list(dropped.keys())}"
            )

        if not surviving:
            result.errors.append(
                f"ALL labels were dropped by LabelSchema! "
                f"Raw labels from MOABB: {list(raw_labels)}. "
                f"Schema map: {schema._raw_to_global}"
            )
            return False

        result.label_distribution = {k: int((y == k).sum()) for k in raw_labels}
        log.info(
            "  [✓] Label translation — %d/%d label types survive: %s",
            len(surviving), len(raw_labels), surviving,
        )
        return True

    except Exception as exc:
        result.errors.append(f"Label translation check failed: {exc}")
        return False


def check_channel_mapping(moabb_dataset, dataset_id: str, subject: int,
                           result: DatasetTestResult) -> bool:
    """Check 4 — at least some channels map to the EEGPT canonical space."""
    try:
        from moabb.paradigms import P300
        from framework.label_schema import DATASET_LABEL_MAPS
        from framework.canonical_channels import resolve_channel_subset

        n_classes = len(DATASET_LABEL_MAPS[dataset_id])
        paradigm = P300(tmin=0.0, tmax=1.0)

        # Get channel names
        if hasattr(moabb_dataset, 'channel_names'):
            raw_ch = moabb_dataset.channel_names
        else:
            epochs, _, _ = paradigm.get_data(dataset=moabb_dataset,
                                              subjects=[subject],
                                              return_epochs=True)
            raw_ch = epochs.ch_names

        valid_canonical, dataset_indices, _ = resolve_channel_subset(raw_ch)

        if not valid_canonical:
            result.errors.append(
                f"No channels from the dataset could be mapped to the canonical "
                f"EEGPT space. Dataset channels were: {list(raw_ch)}"
            )
            return False

        dropped_count = len(raw_ch) - len(valid_canonical)
        if dropped_count:
            result.warnings.append(
                f"{dropped_count}/{len(raw_ch)} channels could not be mapped "
                f"to canonical space and will be dropped."
            )

        result.n_channels = len(valid_canonical)
        result.channel_names = valid_canonical
        log.info(
            "  [✓] Channel mapping — %d/%d channels resolve to canonical space",
            len(valid_canonical), len(raw_ch),
        )
        return True

    except Exception as exc:
        result.errors.append(f"Channel mapping failed: {exc}")
        return False


def check_preprocessing(moabb_dataset, dataset_id: str, subject: int,
                         result: DatasetTestResult) -> bool:
    """Check 5 — EEGPTPreprocessor produces correct output shape [N, C, 1024]."""
    try:
        from moabb.paradigms import P300
        from framework.label_schema import DATASET_LABEL_MAPS
        from framework.preprocessing import EEGPTPreprocessor

        n_classes = len(DATASET_LABEL_MAPS[dataset_id])
        paradigm = P300(tmin=0.0, tmax=1.0)
        X, _, _ = paradigm.get_data(dataset=moabb_dataset, subjects=[subject],
                                     return_epochs=False)

        # Get channel names
        if hasattr(moabb_dataset, 'channel_names'):
            raw_ch = moabb_dataset.channel_names
        else:
            epochs, _, _ = paradigm.get_data(dataset=moabb_dataset,
                                              subjects=[subject],
                                              return_epochs=True)
            raw_ch = epochs.ch_names

        # Use the registry table rather than moabb_dataset.sfreq
        source_sfreq = _get_source_sfreq(dataset_id, moabb_dataset)

        preprocessor = EEGPTPreprocessor(
            dataset_channel_names=raw_ch,
            source_sfreq=source_sfreq,
        )
        X_proc = preprocessor.process_epochs(X)

        expected_time = 1024
        if X_proc.shape[-1] != expected_time:
            result.errors.append(
                f"Preprocessed time dimension is {X_proc.shape[-1]}, expected {expected_time}."
            )
            return False

        if X_proc.shape[0] == 0:
            result.errors.append("Preprocessing produced zero trials.")
            return False

        result.n_samples = X_proc.shape[0]
        result.n_channels = X_proc.shape[1]
        result.eeg_shape = tuple(X_proc.shape)
        log.info(
            "  [✓] Preprocessing — output shape %s (N×C×T)", X_proc.shape,
        )
        return True

    except Exception as exc:
        result.errors.append(f"Preprocessing failed: {exc}\n{traceback.format_exc()}")
        return False


def check_full_loader(moabb_dataset, dataset_id: str, subject: int,
                      result: DatasetTestResult) -> bool:
    """
    Check 6 — MoabbDatasetLoader produces a non-empty EEGSampleDataset
    with correct label integers.
    """
    try:
        from framework.dataset_registry import MoabbDatasetLoader
        from framework.label_schema import LabelSchema, DATASET_LABEL_MAPS

        n_classes = len(DATASET_LABEL_MAPS[dataset_id])
        schema = LabelSchema(dataset_id)

        loader = MoabbDatasetLoader(
            moabb_dataset=moabb_dataset,
            dataset_id=dataset_id,
            label_schema=schema,
            n_classes=n_classes,
            subjects=[subject],
        )
        ds = loader.load_all_subjects()

        if len(ds) == 0:
            result.errors.append(
                "MoabbDatasetLoader returned an empty dataset after full pipeline. "
                "All samples were likely dropped during label translation."
            )
            return False

        # Spot-check first sample
        sample = ds[0]
        eeg = sample['eeg']
        if eeg.shape[-1] != 1024:
            result.errors.append(
                f"Sample EEG time dimension is {eeg.shape[-1]}, expected 1024."
            )
            return False

        labels = [s.label for s in ds.samples]
        unique_labels = set(labels)
        if len(unique_labels) < 2:
            result.warnings.append(
                f"Only {len(unique_labels)} unique label(s) in the loaded data: "
                f"{unique_labels}. Model may not train usefully."
            )

        result.n_samples = len(ds)
        log.info(
            "  [✓] Full loader — %d samples, labels present: %s",
            len(ds), sorted(unique_labels),
        )
        return True

    except Exception as exc:
        result.errors.append(f"Full loader check failed: {exc}\n{traceback.format_exc()}")
        return False


# ---------------------------------------------------------------------------
# Per-dataset test runner
# ---------------------------------------------------------------------------

def validate_dataset(moabb_dataset, dataset_id: str) -> DatasetTestResult:
    result = DatasetTestResult(dataset_id=dataset_id)
    log.info("=" * 60)
    log.info("Testing: %s", dataset_id)

    # Check 1
    if not check_label_schema(dataset_id, result):
        return result

    # Check 2
    subject = check_moabb_download(moabb_dataset, dataset_id, result)
    if subject is None:
        return result

    # Check 3
    if not check_label_translation(moabb_dataset, dataset_id, subject, result):
        return result

    # Check 4
    if not check_channel_mapping(moabb_dataset, dataset_id, subject, result):
        return result

    # Check 5
    if not check_preprocessing(moabb_dataset, dataset_id, subject, result):
        return result

    # Check 6
    if not check_full_loader(moabb_dataset, dataset_id, subject, result):
        return result

    result.passed = True
    return result


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(results: List[DatasetTestResult]) -> None:
    passed  = [r for r in results if r.passed]
    failed  = [r for r in results if not r.passed and not r.skipped]
    skipped = [r for r in results if r.skipped]

    print()
    print("=" * 70)
    print("  DATASET VALIDATION SUMMARY")
    print("=" * 70)
    print(f"  Total:   {len(results)}")
    print(f"  Passed:  {len(passed)}")
    print(f"  Failed:  {len(failed)}")
    print(f"  Skipped: {len(skipped)}")
    print()

    if passed:
        print("── PASSED ─────────────────────────────────────────────────────")
        for r in passed:
            warn_str = f"  ({len(r.warnings)} warning(s))" if r.warnings else ""
            print(f"  ✓  {r.dataset_id:<25}  "
                  f"{r.n_samples:>5} samples  "
                  f"{r.n_channels:>3} ch  "
                  f"{r.n_classes} classes"
                  f"{warn_str}")
            for w in r.warnings:
                print(f"       ⚠  {w}")

    if failed:
        print()
        print("── FAILED ─────────────────────────────────────────────────────")
        for r in failed:
            print(f"  ✗  {r.dataset_id}")
            for err in r.errors:
                # Truncate long tracebacks in the summary
                short = err.split('\n')[0]
                print(f"       ERROR: {short}")
            for w in r.warnings:
                print(f"       WARN:  {w}")

    if skipped:
        print()
        print("── SKIPPED ────────────────────────────────────────────────────")
        for r in skipped:
            print(f"  –  {r.dataset_id}")

    print()
    if failed:
        print("  ✗ Fix the errors above before running the full experiment.")
    else:
        print("  ✓ All datasets passed — safe to run the full experiment.")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    results: List[DatasetTestResult] = []

    for moabb_ds, ds_id in DATASETS_TO_TEST:
        try:
            result = validate_dataset(moabb_ds, ds_id)
        except Exception as exc:
            # Catch anything that slipped through the per-check try/excepts
            result = DatasetTestResult(dataset_id=ds_id)
            result.errors.append(f"Unexpected error: {exc}\n{traceback.format_exc()}")

        results.append(result)

    print_summary(results)

    # Exit with non-zero code if any dataset failed, useful for CI
    if any(not r.passed and not r.skipped for r in results):
        sys.exit(1)