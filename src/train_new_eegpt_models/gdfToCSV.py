import mne
import numpy as np
import pandas as pd
from pathlib import Path


SFREQ = 250
MI_EVENT_ID = {
    '769': 0,  # left hand
    '770': 1,  # right hand
}
DEFAULT_TMIN = 0.0
DEFAULT_TMAX = 4.0


def _load_single_gdf(
    gdf_path: str,
    tmin: float,
    tmax: float,
    include_eog: bool,
    skip_artifacts: bool,
    verbose: bool,
) -> tuple[pd.DataFrame, int]:
    """
    Load one T file and return a flat DataFrame plus the number of
    samples per trial.
    """
    raw = mne.io.read_raw_gdf(gdf_path, stim_channel='auto', preload=True, verbose=verbose)

    # The GDF header has highpass/lowpass swapped (100.0 / 0.5), which causes
    # MNE to refuse direct raw.filter() calls and zero the signal.
    # Solution: filter the underlying numpy array directly, bypassing info checks.
    sfreq = raw.info['sfreq']
    raw._data[:] = mne.filter.filter_data(
        raw._data,
        sfreq=sfreq,
        l_freq=0.5,
        h_freq=100.0,
        method='fir',
        verbose=verbose,
    )

    events, _ = mne.events_from_annotations(raw, event_id=MI_EVENT_ID, verbose=verbose)

    if len(events) == 0:
        raise ValueError(f"No MI events found in {gdf_path}")

    EEG_CHANNEL_NAMES = ['EEG:C3', 'EEG:Cz', 'EEG:C4']
    picks = mne.pick_channels(raw.info['ch_names'], include=EEG_CHANNEL_NAMES)

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id={k: v for k, v in MI_EVENT_ID.items() if v in np.unique(events[:, 2])},
        tmin=tmin,
        tmax=tmax - (1.0 / SFREQ),
        picks=picks,
        baseline=None,
        preload=True,
        reject_by_annotation=skip_artifacts,
        verbose=verbose,
    )

    if len(epochs) == 0:
        raise ValueError(f"All trials rejected in {gdf_path}")

    data   = epochs.get_data()       # [n_epochs, n_channels, n_times]
    labels = epochs.events[:, 2]    # 0 or 1
    n_epochs, n_ch, n_times = data.shape

    # Z-score each channel across all epochs before saving.
    # Raw MNE EEG is in volts (~1e-6). Without normalization the CSV loader
    # reads everything as ~0.0000. The MOABB loader z-scores internally.
    for ch in range(n_ch):
        ch_data = data[:, ch, :]
        mu  = ch_data.mean()
        std = ch_data.std()
        if std > 1e-10:
            data[:, ch, :] = (ch_data - mu) / std

    rows  = data.transpose(0, 2, 1).reshape(-1, n_ch)
    y_col = np.repeat(labels, n_times)

    df = pd.DataFrame(rows, columns=epochs.ch_names)
    df['y'] = y_col.astype(np.int64)
    df.columns = [c.replace('EEG:', '') for c in df.columns if c != 'y'] + ['y']

    return df, n_times


def convert_bciciv2b_directory_to_single_csv(
    gdf_dir: str,
    csv_path: str,
    tmin: float = DEFAULT_TMIN,
    tmax: float = DEFAULT_TMAX,
    include_eog: bool = False,
    skip_artifacts: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, int]:
    """
    Convert all T files in gdf_dir into a single merged CSV.

    Trims every session to the same number of trials as the smallest
    session so each subject/session contributes equally.

    Parameters
    ----------
    gdf_dir   : folder containing the .gdf files
    csv_path  : destination CSV path
    tmin/tmax : epoch window in seconds relative to MI cue
    include_eog : include EOG channels (default False)
    skip_artifacts : drop artifact-flagged trials (default True)
    verbose   : print per-file summary

    Returns
    -------
    df           : the merged DataFrame written to csv_path
    trial_length : samples per trial — pass directly to CsvEegDataLoader
    """
    gdf_files = sorted(Path(gdf_dir).glob("B*T.gdf"))
    if not gdf_files:
        raise FileNotFoundError(f"No B*T.gdf files found in {gdf_dir}")

    # ------------------------------------------------------------------
    # 1. Load all files
    # ------------------------------------------------------------------
    loaded     = []   # list of (filename, df, n_times, n_trials)
    trial_length = None

    for gdf_path in gdf_files:
        try:
            df, n_times = _load_single_gdf(
                str(gdf_path), tmin, tmax, include_eog, skip_artifacts, verbose=False
            )
            n_trials = len(df) // n_times
            loaded.append((gdf_path.name, df, n_times, n_trials))

            if trial_length is None:
                trial_length = n_times
            elif n_times != trial_length:
                raise ValueError(
                    f"Mismatched trial length in {gdf_path.name}: "
                    f"expected {trial_length} samples, got {n_times}"
                )

        except Exception as exc:
            print(f"[WARN] Skipped {gdf_path.name}: {exc}")

    if not loaded:
        raise RuntimeError("No files were successfully loaded.")

    # ------------------------------------------------------------------
    # 2. Trim to the minimum trial count so every file contributes equally
    # ------------------------------------------------------------------
    min_trials = min(n_trials for _, _, _, n_trials in loaded)

    if verbose:
        print(f"\n{'File':<20} {'Trials':>8} {'Rows':>10} {'Used':>8}")
        print("-" * 50)

    trimmed = []
    for filename, df, n_times, n_trials in loaded:
        rows_to_keep = min_trials * n_times
        df_trimmed   = df.iloc[:rows_to_keep].copy()
        trimmed.append(df_trimmed)

        if verbose:
            print(f"{filename:<20} {n_trials:>8} {len(df):>10} {min_trials:>8}")

    # ------------------------------------------------------------------
    # 3. Merge and save
    # ------------------------------------------------------------------
    merged = pd.concat(trimmed, ignore_index=True)
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(csv_path, index=False)

    if verbose:
        total_trials = min_trials * len(loaded)
        print(f"\n{'─'*50}")
        print(f"Files loaded    : {len(loaded)}")
        print(f"Trials per file : {min_trials}  (trimmed to smallest)")
        print(f"Total trials    : {total_trials}")
        print(f"Trial length    : {trial_length} samples  ← use this for CsvEegDataLoader")
        print(f"Total CSV rows  : {len(merged)}")
        print(f"Saved to        : {csv_path}")

    return merged, trial_length