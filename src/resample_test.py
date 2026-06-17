"""
resample_test_.py
-----------------
Visual sanity-check for the resampling step used in preprocessing.py.

Generates a synthetic multi-component sine wave at a user-defined source
frequency, resamples it to 256 Hz using the same linear interpolation that
EEGPTPreprocessor uses (torch.nn.functional.interpolate, mode='linear'),
then plots both signals side-by-side so you can verify that:

  * the waveform shape is preserved after resampling
  * the time axis aligns correctly (both cover the same duration)
  * no obvious aliasing or amplitude distortion

Usage
-----
    python resample_test_.py                  # default: 1000 Hz -> 256 Hz
    python resample_test_.py --src 512        # 512 Hz -> 256 Hz
    python resample_test_.py --src 128        # upsampling: 128 Hz -> 256 Hz
    python resample_test_.py --src 1000 --duration 0.5   # 0.5 s window

The script has NO dependency on the rest of the EEGPT framework; it only
uses numpy, matplotlib, and torch so it can be run in isolation.
"""

from __future__ import annotations
import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ---------------------------------------------------------------------------
# Constants — must match preprocessing.py
# ---------------------------------------------------------------------------
TARGET_SAMPLE_RATE: int = 256   # Hz


# ---------------------------------------------------------------------------
# Resampler (copy of the logic in preprocessing.py — kept inline so this
# file has zero dependency on the rest of the project)
# ---------------------------------------------------------------------------

def resample_linear(signal: np.ndarray, source_sfreq: int, target_sfreq: int) -> np.ndarray:
    """
    Resample a 1-D or 2-D signal [channels, T] from source_sfreq to target_sfreq
    using torch linear interpolation — identical to preprocessing.Resampler.

    Parameters
    ----------
    signal : np.ndarray  shape [T] or [C, T]
    source_sfreq : int
    target_sfreq : int

    Returns
    -------
    np.ndarray  same number of dims as input, resampled along last axis
    """
    squeeze = signal.ndim == 1
    if squeeze:
        signal = signal[np.newaxis, :]          # [1, T]

    x = torch.FloatTensor(signal)               # [C, T]
    target_len = round(x.shape[-1] * target_sfreq / source_sfreq)
    out = F.interpolate(
        x.unsqueeze(0),                         # [1, C, T]
        size=target_len,
        mode='linear',
        align_corners=False,
    ).squeeze(0)                                # [C, T_resampled]

    result = out.numpy()
    return result[0] if squeeze else result


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def make_signal(sfreq: int, duration: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a realistic multi-component EEG-like waveform.

    Components:
      * 10 Hz alpha-band    (dominant)
      * 20 Hz beta-band     (half amplitude)
      *  3 Hz theta-band    (quarter amplitude)
      * 50 Hz power-line noise (small, to test high-freq behaviour)

    Returns (time_axis, signal) both as 1-D numpy arrays.
    """
    t = np.linspace(0, duration, int(sfreq * duration), endpoint=False)
    sig = (
          1.0  * np.sin(2 * math.pi * 10 * t)   # alpha
        + 0.5  * np.sin(2 * math.pi * 20 * t)   # beta
        + 0.25 * np.sin(2 * math.pi *  3 * t)   # theta
        + 0.1  * np.sin(2 * math.pi * 50 * t)   # power-line
    )
    return t, sig


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_resampling(
    source_sfreq: int,
    duration: float = 1.0,
    output_path: str = "resample_test_output.png",
) -> None:
    target_sfreq = TARGET_SAMPLE_RATE

    t_src, sig_src = make_signal(source_sfreq, duration)
    sig_dst = resample_linear(sig_src, source_sfreq, target_sfreq)
    t_dst = np.linspace(0, duration, len(sig_dst), endpoint=False)

    # Spectrum via FFT (for the frequency-domain subplot)
    def spectrum(sig, sfreq):
        n = len(sig)
        freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
        amps  = np.abs(np.fft.rfft(sig)) / n * 2
        return freqs, amps

    freqs_src, amps_src = spectrum(sig_src, source_sfreq)
    freqs_dst, amps_dst = spectrum(sig_dst, target_sfreq)

    # ── Layout ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(
        f"Resampling sanity check:  {source_sfreq} Hz  →  {target_sfreq} Hz "
        f"({duration:.2f} s window, {len(sig_src)} → {len(sig_dst)} samples)",
        fontsize=13, fontweight='bold', y=0.98,
    )

    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    ax_src_t  = fig.add_subplot(gs[0, 0])   # top-left:  source time domain
    ax_dst_t  = fig.add_subplot(gs[0, 1])   # top-right: resampled time domain
    ax_overlay = fig.add_subplot(gs[1, 0])  # bottom-left: overlay
    ax_freq   = fig.add_subplot(gs[1, 1])   # bottom-right: frequency domain

    src_colour = '#2196F3'   # blue
    dst_colour = '#F44336'   # red
    alpha_thin = 0.75

    # -- Source time domain ------------------------------------------------
    ax_src_t.plot(t_src * 1000, sig_src, color=src_colour, lw=1.2, alpha=alpha_thin)
    ax_src_t.scatter(t_src * 1000, sig_src, s=6, color=src_colour, zorder=3,
                     label=f"{source_sfreq} Hz ({len(sig_src)} pts)")
    ax_src_t.set_title(f"Original  ({source_sfreq} Hz)", fontsize=11)
    ax_src_t.set_xlabel("Time (ms)")
    ax_src_t.set_ylabel("Amplitude (a.u.)")
    ax_src_t.legend(fontsize=8)
    ax_src_t.grid(True, alpha=0.3)

    # -- Resampled time domain ---------------------------------------------
    ax_dst_t.plot(t_dst * 1000, sig_dst, color=dst_colour, lw=1.2, alpha=alpha_thin)
    ax_dst_t.scatter(t_dst * 1000, sig_dst, s=6, color=dst_colour, zorder=3,
                     label=f"{target_sfreq} Hz ({len(sig_dst)} pts)")
    ax_dst_t.set_title(f"Resampled  ({target_sfreq} Hz)", fontsize=11)
    ax_dst_t.set_xlabel("Time (ms)")
    ax_dst_t.set_ylabel("Amplitude (a.u.)")
    ax_dst_t.legend(fontsize=8)
    ax_dst_t.grid(True, alpha=0.3)

    # -- Overlay (shows alignment) ----------------------------------------
    ax_overlay.plot(t_src * 1000, sig_src, color=src_colour, lw=1.5,
                    alpha=0.7, label=f"Original ({source_sfreq} Hz)")
    ax_overlay.plot(t_dst * 1000, sig_dst, color=dst_colour, lw=1.5,
                    alpha=0.7, linestyle='--', label=f"Resampled ({target_sfreq} Hz)")
    ax_overlay.set_title("Overlay", fontsize=11)
    ax_overlay.set_xlabel("Time (ms)")
    ax_overlay.set_ylabel("Amplitude (a.u.)")
    ax_overlay.legend(fontsize=8)
    ax_overlay.grid(True, alpha=0.3)

    # -- Frequency domain --------------------------------------------------
    nyq_src = source_sfreq / 2
    nyq_dst = target_sfreq / 2
    fmax_plot = min(nyq_src, nyq_dst, 80)   # cap at 80 Hz for readability

    mask_src = freqs_src <= fmax_plot
    mask_dst = freqs_dst <= fmax_plot

    ax_freq.plot(freqs_src[mask_src], amps_src[mask_src],
                 color=src_colour, lw=1.5, alpha=0.8, label=f"Original ({source_sfreq} Hz)")
    ax_freq.plot(freqs_dst[mask_dst], amps_dst[mask_dst],
                 color=dst_colour, lw=1.5, alpha=0.8, linestyle='--',
                 label=f"Resampled ({target_sfreq} Hz)")
    ax_freq.axvline(nyq_dst, color='grey', lw=0.8, linestyle=':', label=f"Nyquist @ {target_sfreq} Hz")
    ax_freq.set_title("Frequency spectrum", fontsize=11)
    ax_freq.set_xlabel("Frequency (Hz)")
    ax_freq.set_ylabel("|Amplitude|")
    ax_freq.legend(fontsize=8)
    ax_freq.grid(True, alpha=0.3)

    # Annotate the known spectral peaks
    for freq, label in [(3, "theta"), (10, "alpha"), (20, "beta"), (50, "50 Hz")]:
        if freq <= fmax_plot:
            ax_freq.axvline(freq, color='green', lw=0.6, linestyle='--', alpha=0.5)
            ax_freq.text(freq + 0.5, ax_freq.get_ylim()[1] * 0.85, label,
                         fontsize=7, color='green', rotation=90, va='top')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Print a quick numeric summary
    print(f"\n{'='*55}")
    print(f"  Source:    {source_sfreq} Hz  |  {len(sig_src)} samples over {duration:.2f} s")
    print(f"  Target:    {target_sfreq} Hz  |  {len(sig_dst)} samples over {duration:.2f} s")
    print(f"  Ratio:     {target_sfreq}/{source_sfreq} = {target_sfreq/source_sfreq:.4f}")
    print(f"  Max abs deviation (overlay): {np.max(np.abs(sig_src - resample_linear(sig_dst, target_sfreq, source_sfreq))):.4f} a.u.")
    print(f"{'='*55}\n")

    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualise the resampling step from preprocessing.py."
    )
    parser.add_argument(
        "--src", type=int, default=1000,
        help="Source sampling frequency in Hz (default: 1000). "
             "Will be resampled to 256 Hz.",
    )
    parser.add_argument(
        "--duration", type=float, default=1.0,
        help="Window duration in seconds (default: 1.0).",
    )
    parser.add_argument(
        "--out", type=str, default="resample_test_output.png",
        help="Output image path (default: resample_test_output.png).",
    )
    args = parser.parse_args()

    plot_resampling(
        source_sfreq=args.src,
        duration=args.duration,
        output_path=args.out,
    )