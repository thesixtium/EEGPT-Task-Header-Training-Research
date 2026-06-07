"""
evaluation/analytics.py
------------------------
Comprehensive reporting and visualisation for LOSO results.

Designed to support the evaluation of a *zero-calibration, generalizable*
P300 binary classifier (Target vs NonTarget) under Leave-One-Subject-Out
cross-validation.

Every plot answers a specific scientific question about cross-subject
generalization — the core claim being:
    "Can this model be deployed to a completely new user without any
     subject-specific fine-tuning or calibration?"

Plots produced
--------------
1.  loso_metrics_overview.png   — Bar chart: Acc / Bal-Acc / F1 / MCC / AUC per subject
2.  balanced_accuracy_detail.png— Dedicated BA plot with chance baseline (50%)
3.  confusion_matrices.png      — Grid of per-subject normalised confusion matrices
4.  aggregate_radar.png         — Radar chart of mean metrics (quick one-glance summary)
5.  metric_distributions.png    — Violin + swarm: distribution of each metric across subjects
6.  subject_metric_heatmap.png  — Heatmap: subjects × metrics
7.  roc_curves.png              — Per-subject ROC curves + mean ± std band
8.  precision_recall_curves.png — Per-subject PR curves + mean ± std band (P300 is imbalanced)
9.  calibration_curves.png      — Reliability diagrams (are predicted probabilities trustworthy?)
10. training_curves.png         — Loss & accuracy per epoch for every fold (requires history)
11. training_curves_summary.png — Mean ± std training curves across all folds
12. generalization_gap.png      — Train vs val metric per fold (overfitting/underfitting probe)
13. error_analysis.png          — FP / FN / TP / TN rates per subject (class confusion detail)
14. cumulative_performance.png  — Sorted subjects: how many pass clinical / research thresholds?

All figures use a consistent dark-on-light scientific theme.
All matplotlib calls are guarded so the module works in headless environments.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional deps
# ---------------------------------------------------------------------------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.lines import Line2D
    import matplotlib.ticker as mticker
    _MPL = True
except ImportError:
    _MPL = False
    logger.warning("matplotlib not available — plots will be skipped")

try:
    import seaborn as sns
    _SNS = True
except ImportError:
    _SNS = False

try:
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    from sklearn.calibration import calibration_curve
    _SKL = True
except ImportError:
    _SKL = False

try:
    from scipy.stats import mannwhitneyu
    _SCIPY = True
except ImportError:
    _SCIPY = False


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
# Palette inspired by neuroimaging / EEG toolbox aesthetics.
# Clean white background, slate accents, and a distinct highlight per metric.

PALETTE = {
    "accuracy":          "#2E86AB",   # steel blue
    "balanced_accuracy": "#E84855",   # crimson — the primary metric, most prominent
    "f1_macro":          "#3BB273",   # emerald
    "mcc":               "#F4A261",   # amber
    "roc_auc":           "#9B5DE5",   # violet
    "background":        "#FFFFFF",
    "grid":              "#E8EDF2",
    "text":              "#1A1D23",
    "subtext":           "#6B7280",
    "chance":            "#C0C0C0",
}

METRIC_LABELS = {
    "accuracy":          "Accuracy",
    "balanced_accuracy": "Balanced Accuracy",
    "f1_macro":          "F1 (macro)",
    "mcc":               "MCC",
    "roc_auc":           "ROC-AUC",
}

CLINICAL_THRESHOLD = 0.70   # "acceptable" generalisation bar
RESEARCH_THRESHOLD = 0.80   # "strong" generalisation bar


def _apply_base_style(ax, title: str = "", xlabel: str = "", ylabel: str = "") -> None:
    """Apply consistent styling to an axes."""
    ax.set_facecolor(PALETTE["background"])
    ax.grid(True, color=PALETTE["grid"], linewidth=0.8, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CBD5E1")
    ax.tick_params(colors=PALETTE["text"], labelsize=9)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold",
                     color=PALETTE["text"], pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=PALETTE["subtext"])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=PALETTE["subtext"])


def _fig_style(fig) -> None:
    fig.patch.set_facecolor(PALETTE["background"])


def _save(fig, path: Path, dpi: int = 150) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight",
                facecolor=PALETTE["background"])
    logger.info("Saved %s", path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# MCC helper (not always in sklearn for this import pattern)
# ---------------------------------------------------------------------------

def _mcc(cm: List[List[int]]) -> float:
    """Matthews Correlation Coefficient from a 2×2 confusion matrix."""
    if cm is None or len(cm) != 2:
        return float("nan")
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    if denom == 0:
        return float("nan")
    return (tp * tn - fp * fn) / denom


def _enrich_metrics(result: Dict) -> Dict:
    """Add MCC to a result dict if not already present."""
    m = result.get("metrics", {})
    if m and "mcc" not in m:
        m["mcc"] = _mcc(m.get("confusion_matrix"))
    return result


def _get(results: List[Dict], key: str) -> List[float]:
    return [r.get("metrics", {}).get(key, float("nan")) for r in results]


def _subject_ids(results: List[Dict]) -> List[str]:
    return [str(r["subject_id"]) for r in results]


# ---------------------------------------------------------------------------
# 1. LOSO Metrics Overview (multi-metric bar chart)
# ---------------------------------------------------------------------------

def plot_loso_metrics_overview(
    all_results: List[Dict],
    output_dir: Path,
) -> None:
    """Bar chart showing all 5 key metrics for every subject side-by-side."""
    if not _MPL:
        return
    results = [_enrich_metrics(r) for r in all_results]
    sids = _subject_ids(results)
    n = len(sids)

    metrics = ["accuracy", "balanced_accuracy", "f1_macro", "mcc", "roc_auc"]
    colors  = [PALETTE[m] for m in metrics]
    labels  = [METRIC_LABELS[m] for m in metrics]

    x = np.arange(n)
    width = 0.15
    offsets = np.linspace(-(len(metrics)-1)/2, (len(metrics)-1)/2, len(metrics)) * width

    fig, ax = plt.subplots(figsize=(max(10, n * 1.2), 5))
    _fig_style(fig)

    for i, (metric, color, label) in enumerate(zip(metrics, colors, labels)):
        vals = _get(results, metric)
        ax.bar(x + offsets[i], vals, width, label=label,
               color=color, alpha=0.85, zorder=3)

    # Chance lines
    ax.axhline(0.5, color=PALETTE["chance"], linestyle="--",
               linewidth=1.2, zorder=2, label="Chance (0.5)")
    ax.axhline(CLINICAL_THRESHOLD, color="#94A3B8", linestyle=":",
               linewidth=1.0, zorder=2, label=f"Clinical threshold ({CLINICAL_THRESHOLD})")
    ax.axhline(RESEARCH_THRESHOLD, color="#64748B", linestyle=":",
               linewidth=1.0, zorder=2, label=f"Research threshold ({RESEARCH_THRESHOLD})")

    _apply_base_style(ax,
        title="Per-Subject LOSO Performance — All Metrics",
        xlabel="Subject ID",
        ylabel="Score")
    ax.set_xticks(x)
    ax.set_xticklabels(sids, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9,
              ncol=2, edgecolor="#E2E8F0")

    # Annotate mean balanced accuracy
    ba_vals = [v for v in _get(results, "balanced_accuracy") if not np.isnan(v)]
    if ba_vals:
        mean_ba = np.mean(ba_vals)
        ax.text(0.01, 0.97, f"Mean Balanced Accuracy: {mean_ba:.3f}",
                transform=ax.transAxes, fontsize=9, color=PALETTE["balanced_accuracy"],
                fontweight="bold", va="top")

    fig.tight_layout()
    _save(fig, output_dir / "loso_metrics_overview.png")


# ---------------------------------------------------------------------------
# 2. Balanced Accuracy Detail
# ---------------------------------------------------------------------------

def plot_balanced_accuracy_detail(
    all_results: List[Dict],
    output_dir: Path,
) -> None:
    """
    Focused plot on Balanced Accuracy — the primary generalization metric
    for an imbalanced P300 paradigm.
    """
    if not _MPL:
        return
    results = [_enrich_metrics(r) for r in all_results]
    sids = _subject_ids(results)
    ba_vals = _get(results, "balanced_accuracy")
    x = np.arange(len(sids))

    fig, ax = plt.subplots(figsize=(max(8, len(sids) * 0.9), 4.5))
    _fig_style(fig)

    # Color bars by performance tier
    def bar_color(v):
        if np.isnan(v):      return PALETTE["chance"]
        if v >= RESEARCH_THRESHOLD:  return "#22C55E"
        if v >= CLINICAL_THRESHOLD:  return "#F59E0B"
        return "#EF4444"

    bar_colors = [bar_color(v) for v in ba_vals]
    bars = ax.bar(x, ba_vals, color=bar_colors, alpha=0.85, zorder=3, edgecolor="white", linewidth=0.5)

    ax.axhline(0.5,  color=PALETTE["chance"],   linestyle="--", linewidth=1.5, zorder=2, label="Chance (0.5)")
    ax.axhline(CLINICAL_THRESHOLD, color="#F59E0B", linestyle=":", linewidth=1.5, zorder=2,
               label=f"Clinical bar ({CLINICAL_THRESHOLD})")
    ax.axhline(RESEARCH_THRESHOLD, color="#22C55E", linestyle=":", linewidth=1.5, zorder=2,
               label=f"Research bar ({RESEARCH_THRESHOLD})")

    valid = [v for v in ba_vals if not np.isnan(v)]
    if valid:
        mean_ba = np.mean(valid)
        std_ba  = np.std(valid)
        ax.axhline(mean_ba, color=PALETTE["balanced_accuracy"], linewidth=2, zorder=4,
                   label=f"Mean = {mean_ba:.3f} ± {std_ba:.3f}")
        ax.fill_between([-0.5, len(sids)-0.5],
                        mean_ba - std_ba, mean_ba + std_ba,
                        color=PALETTE["balanced_accuracy"], alpha=0.08, zorder=1)

    # Value labels on bars
    for bar, v in zip(bars, ba_vals):
        if not np.isnan(v):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.008, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7, color=PALETTE["text"])

    _apply_base_style(ax,
        title="Balanced Accuracy per Subject (LOSO)\nPrimary Generalization Metric for P300",
        xlabel="Subject ID",
        ylabel="Balanced Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(sids, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9, edgecolor="#E2E8F0")

    # Tier legend annotation
    legend_patches = [
        plt.Rectangle((0,0),1,1, color="#22C55E", label=f"≥ {RESEARCH_THRESHOLD} (research)"),
        plt.Rectangle((0,0),1,1, color="#F59E0B", label=f"≥ {CLINICAL_THRESHOLD} (clinical)"),
        plt.Rectangle((0,0),1,1, color="#EF4444", label=f"< {CLINICAL_THRESHOLD} (below threshold)"),
    ]
    ax.legend(handles=legend_patches + ax.get_legend_handles_labels()[0][
                  [l.startswith("Chance") or l.startswith("Mean") for l in ax.get_legend_handles_labels()[1]].index(True)
                  if any(l.startswith("Chance") for l in ax.get_legend_handles_labels()[1]) else 0:],
              loc="lower right", fontsize=8, framealpha=0.9, edgecolor="#E2E8F0")

    fig.tight_layout()
    _save(fig, output_dir / "balanced_accuracy_detail.png")


# ---------------------------------------------------------------------------
# 3. Confusion Matrices Grid
# ---------------------------------------------------------------------------

def plot_confusion_matrices(
    all_results: List[Dict],
    class_names: List[str],
    output_dir: Path,
) -> None:
    """Grid of normalized confusion matrices, one per subject."""
    if not _MPL:
        return
    results = [r for r in all_results if r.get("metrics", {}).get("confusion_matrix")]
    if not results:
        return

    n = len(results)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(ncols * 3.2, nrows * 3.2))
    _fig_style(fig)
    axes_flat = np.array(axes).flatten()

    for ax, result in zip(axes_flat, results):
        cm = np.array(result["metrics"]["confusion_matrix"], dtype=float)
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_norm = cm / row_sums

        ba = result["metrics"].get("balanced_accuracy", float("nan"))
        mcc_val = _mcc(result["metrics"]["confusion_matrix"])

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        nc = len(class_names)
        ax.set_xticks(range(nc))
        ax.set_yticks(range(nc))
        ax.set_xticklabels(class_names, fontsize=8, rotation=30, ha="right")
        ax.set_yticklabels(class_names, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=7, color=PALETTE["subtext"])
        ax.set_ylabel("True", fontsize=7, color=PALETTE["subtext"])
        ax.spines[["top","right","left","bottom"]].set_color("#CBD5E1")

        for i in range(nc):
            for j in range(nc):
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{cm_norm[i,j]:.2f}\n({int(cm[i,j])})",
                        ha="center", va="center", fontsize=8, color=color)

        title_color = "#22C55E" if ba >= RESEARCH_THRESHOLD else (
                      "#F59E0B" if ba >= CLINICAL_THRESHOLD else "#EF4444") \
                      if not np.isnan(ba) else PALETTE["text"]
        ax.set_title(
            f"S{result['subject_id']}  BA={ba:.3f}  MCC={mcc_val:.3f}",
            fontsize=8, color=title_color, fontweight="bold"
        )

    # Hide unused axes
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle("Per-Subject Normalised Confusion Matrices (LOSO)\nTop-left: True Negative (NonTarget), Bottom-right: True Positive (Target)",
                 fontsize=11, fontweight="bold", color=PALETTE["text"], y=1.01)
    fig.tight_layout()
    _save(fig, output_dir / "confusion_matrices.png")


# Also keep the legacy single-matrix function
def plot_confusion_matrix(
    cm: List[List[int]],
    class_names: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
) -> None:
    if not _MPL:
        return
    cm_arr = np.array(cm, dtype=float)
    row_sums = cm_arr.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm_arr / row_sums
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(4, n), max(4, n)))
    _fig_style(fig)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set(xticks=range(n), yticks=range(n),
           xticklabels=class_names, yticklabels=class_names,
           xlabel="Predicted", ylabel="True", title=title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                    color="white" if cm_norm[i,j] > 0.5 else "black")
    fig.tight_layout()
    if save_path:
        _save(fig, save_path)
    else:
        plt.close(fig)


def save_all_confusion_matrices(
    all_results: List[Dict],
    class_names: List[str],
    output_dir: Path,
) -> None:
    for result in all_results:
        sid = result["subject_id"]
        cm  = result.get("metrics", {}).get("confusion_matrix")
        if cm is None:
            continue
        plot_confusion_matrix(
            cm, class_names,
            title=f"Subject {sid}",
            save_path=output_dir / f"confusion_matrix_subject_{sid}.png",
        )


# ---------------------------------------------------------------------------
# 4. Radar Chart (aggregate overview)
# ---------------------------------------------------------------------------

def plot_aggregate_radar(
    aggregate: Dict,
    output_dir: Path,
) -> None:
    """Spider / radar chart of mean metrics — the single-glance summary."""
    if not _MPL:
        return

    metrics = ["accuracy", "balanced_accuracy", "f1_macro", "mcc", "roc_auc"]
    labels  = [METRIC_LABELS[m] for m in metrics]
    values  = []
    stds    = []
    for m in metrics:
        stats = aggregate.get(m, {})
        raw_mean = stats.get("mean", float("nan"))
        raw_std  = stats.get("std",  0.0)
        # MCC is in [-1, 1]; normalise to [0, 1] for radar display
        if m == "mcc":
            raw_mean = (raw_mean + 1) / 2 if not np.isnan(raw_mean) else float("nan")
        values.append(raw_mean)
        stds.append(raw_std / 2 if m == "mcc" else raw_std)

    n = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles_closed = angles + angles[:1]
    values_plot = values + values[:1]
    std_hi = [min(1, v + s) for v, s in zip(values, stds)] + [min(1, values[0] + stds[0])]
    std_lo = [max(0, v - s) for v, s in zip(values, stds)] + [max(0, values[0] - stds[0])]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    _fig_style(fig)
    ax.set_facecolor(PALETTE["background"])

    # Grid rings
    for r in [0.25, 0.5, 0.75, 1.0]:
        ax.plot(angles_closed, [r] * (n+1), color=PALETTE["grid"], linewidth=0.8, zorder=1)
        ax.text(0, r, f"{r:.2f}", ha="center", va="center", fontsize=7,
                color=PALETTE["subtext"])

    # Chance ring
    ax.plot(angles_closed, [0.5] * (n+1),
            color=PALETTE["chance"], linewidth=1.5, linestyle="--", zorder=2, label="Chance")

    # Std band
    ax.fill_between(angles_closed, std_lo, std_hi,
                    color=PALETTE["balanced_accuracy"], alpha=0.12, zorder=3)

    # Mean values
    ax.plot(angles_closed, values_plot,
            color=PALETTE["balanced_accuracy"], linewidth=2.5, zorder=4)
    ax.fill(angles_closed, values_plot,
            color=PALETTE["balanced_accuracy"], alpha=0.25, zorder=3)

    # Dots at each vertex
    ax.scatter(angles, values,
               color=[PALETTE[m] for m in metrics], s=60, zorder=5)

    # Axis labels
    ax.set_thetagrids(np.degrees(angles), labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)

    title_lines = ["Aggregate LOSO Performance — Radar Summary",
                   f"(MCC axis normalised to [0,1];  mean ± std across subjects)"]
    ax.set_title("\n".join(title_lines), fontsize=10, fontweight="bold",
                 color=PALETTE["text"], pad=20)

    fig.tight_layout()
    _save(fig, output_dir / "aggregate_radar.png")


# ---------------------------------------------------------------------------
# 5. Metric Distributions (violin + strip)
# ---------------------------------------------------------------------------

def plot_metric_distributions(
    all_results: List[Dict],
    output_dir: Path,
) -> None:
    """Violin + strip plot of each metric across subjects."""
    if not _MPL:
        return
    results = [_enrich_metrics(r) for r in all_results]
    metrics = ["accuracy", "balanced_accuracy", "f1_macro", "mcc", "roc_auc"]
    data = {m: [v for v in _get(results, m) if not np.isnan(v)] for m in metrics}

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 5), sharey=False)
    _fig_style(fig)

    for ax, metric in zip(axes, metrics):
        vals = data[metric]
        if not vals:
            ax.set_visible(False)
            continue
        color = PALETTE[metric]
        # Violin
        if len(vals) >= 4:
            parts = ax.violinplot(vals, positions=[0], showmedians=True,
                                   showextrema=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.4)
            for part in ["cmedians", "cmins", "cmaxes", "cbars"]:
                if part in parts:
                    parts[part].set_color(color)
                    parts[part].set_linewidth(1.5)

        # Strip
        jitter = np.random.uniform(-0.05, 0.05, len(vals))
        ax.scatter(jitter, vals, color=color, alpha=0.7, s=30, zorder=4)

        # Chance line
        ax.axhline(0.5, color=PALETTE["chance"], linestyle="--", linewidth=1.2)
        ax.axhline(CLINICAL_THRESHOLD, color="#94A3B8", linestyle=":", linewidth=1.0)

        mean_val = np.mean(vals)
        ax.axhline(mean_val, color=color, linewidth=2, zorder=5,
                   label=f"μ={mean_val:.3f}")

        _apply_base_style(ax, title=METRIC_LABELS[metric], ylabel="Score")
        ax.set_xlim(-0.3, 0.3)
        ax.set_xticks([])
        lo = -1 if metric == "mcc" else 0
        ax.set_ylim(lo, 1.05)
        ax.text(0, mean_val + 0.02, f"{mean_val:.3f}", ha="center",
                fontsize=9, color=color, fontweight="bold")

    fig.suptitle("Distribution of Metrics Across LOSO Subjects",
                 fontsize=12, fontweight="bold", color=PALETTE["text"])
    fig.tight_layout()
    _save(fig, output_dir / "metric_distributions.png")


# ---------------------------------------------------------------------------
# 6. Subject × Metric Heatmap
# ---------------------------------------------------------------------------

def plot_subject_metric_heatmap(
    all_results: List[Dict],
    output_dir: Path,
) -> None:
    """Heatmap: rows = subjects, columns = metrics."""
    if not _MPL:
        return
    results = [_enrich_metrics(r) for r in all_results]
    metrics = ["accuracy", "balanced_accuracy", "f1_macro", "mcc", "roc_auc"]
    sids = _subject_ids(results)

    matrix = np.array([
        [r.get("metrics", {}).get(m, float("nan")) for m in metrics]
        for r in results
    ])

    # Normalise MCC from [-1,1] to [0,1] for consistent heatmap scale
    mcc_col = metrics.index("mcc")
    matrix[:, mcc_col] = (matrix[:, mcc_col] + 1) / 2

    fig, ax = plt.subplots(figsize=(8, max(4, len(sids) * 0.5 + 1.5)))
    _fig_style(fig)

    if _SNS:
        sns.heatmap(
            matrix,
            ax=ax,
            xticklabels=[METRIC_LABELS[m] for m in metrics],
            yticklabels=sids,
            cmap="RdYlGn",
            vmin=0, vmax=1,
            annot=True, fmt=".3f",
            linewidths=0.5, linecolor=PALETTE["grid"],
            cbar_kws={"label": "Score (MCC normalised to [0,1])"},
            annot_kws={"size": 8},
        )
    else:
        im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([METRIC_LABELS[m] for m in metrics], rotation=30, ha="right")
        ax.set_yticks(range(len(sids)))
        ax.set_yticklabels(sids)
        fig.colorbar(im, ax=ax)
        for i in range(len(sids)):
            for j in range(len(metrics)):
                v = matrix[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                            fontsize=8, color="black")

    # Threshold lines (horizontal rows)
    for i, row in enumerate(matrix):
        ba = row[metrics.index("balanced_accuracy")]
        if ba < CLINICAL_THRESHOLD and not np.isnan(ba):
            ax.add_patch(plt.Rectangle((0, i-0.01 if _SNS else i-0.5),
                                        len(metrics), 0.02 if _SNS else 1,
                                        fill=False, edgecolor="#EF4444",
                                        linewidth=1.5, clip_on=False))

    ax.set_title(
        "Subject × Metric Performance Heatmap (LOSO)\n"
        "Red highlight: subjects below clinical threshold on Balanced Accuracy",
        fontsize=10, fontweight="bold", color=PALETTE["text"]
    )
    fig.tight_layout()
    _save(fig, output_dir / "subject_metric_heatmap.png")


# ---------------------------------------------------------------------------
# 7. ROC Curves
# ---------------------------------------------------------------------------

def plot_roc_curves(
    all_results: List[Dict],
    output_dir: Path,
) -> None:
    """
    Per-subject ROC curves.  Requires `y_true` and `y_prob` stored in results.
    Falls back to single AUC dot plot if raw arrays are unavailable.
    """
    if not _MPL:
        return

    has_arrays = any(
        "y_true" in r.get("metrics", {}) and "y_prob" in r.get("metrics", {})
        for r in all_results
    )

    results = [_enrich_metrics(r) for r in all_results]

    if has_arrays and _SKL:
        _plot_roc_full(results, output_dir)
    else:
        _plot_roc_auc_bars(results, output_dir)


def _plot_roc_full(results: List[Dict], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    _fig_style(fig)

    tprs, fprs_base = [], np.linspace(0, 1, 200)
    for r in results:
        m = r.get("metrics", {})
        y_true = np.array(m["y_true"])
        y_prob = np.array(m["y_prob"])
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, alpha=0.3, lw=1, color=PALETTE["roc_auc"])
        tprs.append(np.interp(fprs_base, fpr, tpr))
        tprs[-1][0] = 0.0

    if tprs:
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr  = np.std(tprs, axis=0)
        mean_auc = auc(fprs_base, mean_tpr)
        ax.plot(fprs_base, mean_tpr, lw=2.5, color=PALETTE["roc_auc"],
                label=f"Mean ROC (AUC = {mean_auc:.3f})")
        ax.fill_between(fprs_base, mean_tpr - std_tpr, mean_tpr + std_tpr,
                        color=PALETTE["roc_auc"], alpha=0.15,
                        label="± 1 std dev")

    ax.plot([0,1],[0,1], "--", color=PALETTE["chance"], lw=1.5, label="Chance")
    _apply_base_style(ax, "ROC Curves — LOSO (all subjects)", "False Positive Rate", "True Positive Rate")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    fig.tight_layout()
    _save(fig, output_dir / "roc_curves.png")


def _plot_roc_auc_bars(results: List[Dict], output_dir: Path) -> None:
    """Fallback: simple bar chart of per-subject ROC-AUC."""
    sids = _subject_ids(results)
    vals = _get(results, "roc_auc")
    valid_vals = [v for v in vals if not np.isnan(v)]
    mean_auc = np.mean(valid_vals) if valid_vals else float("nan")

    fig, ax = plt.subplots(figsize=(max(7, len(sids)), 4))
    _fig_style(fig)
    colors = [PALETTE["roc_auc"]] * len(vals)
    ax.bar(range(len(sids)), vals, color=colors, alpha=0.8, zorder=3)
    ax.axhline(0.5, color=PALETTE["chance"], linestyle="--", lw=1.5, label="Chance")
    if not np.isnan(mean_auc):
        ax.axhline(mean_auc, color=PALETTE["roc_auc"], lw=2,
                   label=f"Mean AUC = {mean_auc:.3f}")
    _apply_base_style(ax, "ROC-AUC per Subject (LOSO)\n(full ROC curves require y_prob arrays in results)",
                      "Subject", "ROC-AUC")
    ax.set_xticks(range(len(sids)))
    ax.set_xticklabels(sids)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, output_dir / "roc_curves.png")


# ---------------------------------------------------------------------------
# 8. Precision–Recall Curves
# ---------------------------------------------------------------------------

def plot_precision_recall_curves(
    all_results: List[Dict],
    output_dir: Path,
) -> None:
    """
    PR curves are more informative than ROC for class-imbalanced P300 data.
    Falls back to F1 bar chart if raw arrays unavailable.
    """
    if not _MPL:
        return
    results = [_enrich_metrics(r) for r in all_results]
    has_arrays = any(
        "y_true" in r.get("metrics", {}) and "y_prob" in r.get("metrics", {})
        for r in results
    )

    if has_arrays and _SKL:
        fig, ax = plt.subplots(figsize=(6, 6))
        _fig_style(fig)
        all_precs, all_recs_base = [], np.linspace(0, 1, 200)
        for r in results:
            m = r.get("metrics", {})
            y_true = np.array(m["y_true"])
            y_prob = np.array(m["y_prob"])
            if len(np.unique(y_true)) < 2:
                continue
            prec, rec, _ = precision_recall_curve(y_true, y_prob)
            ax.plot(rec, prec, alpha=0.3, lw=1, color=PALETTE["f1_macro"])
            all_precs.append(np.interp(all_recs_base, rec[::-1], prec[::-1]))

        if all_precs:
            mean_prec = np.mean(all_precs, axis=0)
            std_prec  = np.std(all_precs, axis=0)
            ax.plot(all_recs_base, mean_prec, lw=2.5, color=PALETTE["f1_macro"], label="Mean PR")
            ax.fill_between(all_recs_base,
                            np.clip(mean_prec - std_prec, 0, 1),
                            np.clip(mean_prec + std_prec, 0, 1),
                            color=PALETTE["f1_macro"], alpha=0.15, label="± 1 std dev")

        _apply_base_style(ax, "Precision–Recall Curves — LOSO\n(P300 is class-imbalanced; PR curves more informative than ROC)",
                          "Recall", "Precision")
        ax.legend(fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    else:
        # Fallback: F1 bar chart
        sids = _subject_ids(results)
        vals = _get(results, "f1_macro")
        fig, ax = plt.subplots(figsize=(max(7, len(sids)), 4))
        _fig_style(fig)
        ax.bar(range(len(sids)), vals, color=PALETTE["f1_macro"], alpha=0.8, zorder=3)
        ax.axhline(np.nanmean(vals), color=PALETTE["f1_macro"], lw=2,
                   label=f"Mean F1 = {np.nanmean(vals):.3f}")
        _apply_base_style(ax, "F1 (macro) per Subject — LOSO\n(PR curves require y_prob arrays in results)",
                          "Subject", "F1 (macro)")
        ax.set_xticks(range(len(sids)))
        ax.set_xticklabels(sids)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=9)

    fig.tight_layout()
    _save(fig, output_dir / "precision_recall_curves.png")


# ---------------------------------------------------------------------------
# 9. Calibration Curves
# ---------------------------------------------------------------------------

def plot_calibration_curves(
    all_results: List[Dict],
    output_dir: Path,
) -> None:
    """
    Reliability diagrams.  Well-calibrated probabilities matter for
    BCI systems that threshold or combine evidence across trials.
    Falls back to a note if raw arrays unavailable.
    """
    if not _MPL or not _SKL:
        return
    results = [_enrich_metrics(r) for r in all_results]
    has_arrays = any(
        "y_true" in r.get("metrics", {}) and "y_prob" in r.get("metrics", {})
        for r in results
    )
    if not has_arrays:
        logger.info("Calibration curves skipped — y_prob arrays not stored in results. "
                    "Add y_true/y_prob to each result dict in loso_runner.py to enable.")
        return

    fig, ax = plt.subplots(figsize=(6, 6))
    _fig_style(fig)
    ax.plot([0, 1], [0, 1], "--", color=PALETTE["chance"], lw=1.5, label="Perfect calibration")

    for r in results:
        m = r.get("metrics", {})
        y_true = np.array(m.get("y_true", []))
        y_prob = np.array(m.get("y_prob", []))
        if len(y_true) == 0 or len(np.unique(y_true)) < 2:
            continue
        try:
            frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
            ax.plot(mean_pred, frac_pos, alpha=0.4, lw=1, color=PALETTE["mcc"],
                    marker="o", markersize=3)
        except Exception:
            pass

    _apply_base_style(ax, "Calibration (Reliability) Curves — LOSO\nAre predicted P300 probabilities trustworthy?",
                      "Mean Predicted Probability", "Fraction of Positives")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    fig.tight_layout()
    _save(fig, output_dir / "calibration_curves.png")


# ---------------------------------------------------------------------------
# 10 & 11. Training Curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    all_results: List[Dict],
    output_dir: Path,
) -> None:
    """
    Per-fold and summary training curves.

    Expects each result dict to optionally contain a "training_history" key:
        {
          "train_loss": [float, ...],
          "val_loss":   [float, ...],
          "train_acc":  [float, ...],   # optional
          "val_acc":    [float, ...],   # optional
          "val_bal_acc":[float, ...],   # optional — balanced accuracy per epoch
        }

    If no history is found this function logs a helpful message and returns.
    """
    if not _MPL:
        return
    histories = [r.get("training_history") for r in all_results]
    if not any(h for h in histories):
        logger.info(
            "Training curves skipped — no 'training_history' found in results.\n"
            "  To enable: in trainer.py, collect per-epoch metrics and store them\n"
            "  in the result dict as result['training_history'] = {\n"
            "      'train_loss': [...], 'val_loss': [...],\n"
            "      'train_acc': [...],  'val_acc': [...],\n"
            "      'val_bal_acc': [...]\n"
            "  }"
        )
        return

    valid = [(r, h) for r, h in zip(all_results, histories) if h]
    _plot_training_per_fold(valid, output_dir)
    _plot_training_summary(valid, output_dir)


def _plot_training_per_fold(pairs, output_dir: Path) -> None:
    n = len(pairs)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols * 2,
                              figsize=(ncols * 6, nrows * 3.5))
    _fig_style(fig)
    axes_flat = np.array(axes).flatten()

    for idx, (result, hist) in enumerate(pairs):
        ax_loss = axes_flat[idx * 2]
        ax_acc  = axes_flat[idx * 2 + 1]
        epochs  = range(1, len(hist.get("train_loss", [])) + 1)

        if "train_loss" in hist:
            ax_loss.plot(epochs, hist["train_loss"], color=PALETTE["accuracy"], lw=1.5, label="Train")
        if "val_loss" in hist:
            ax_loss.plot(epochs, hist["val_loss"], color=PALETTE["balanced_accuracy"], lw=1.5,
                         linestyle="--", label="Val")
        _apply_base_style(ax_loss, f"S{result['subject_id']} Loss", "Epoch", "Loss")
        ax_loss.legend(fontsize=7)

        if "val_bal_acc" in hist:
            ax_acc.plot(epochs, hist["val_bal_acc"], color=PALETTE["balanced_accuracy"], lw=1.5,
                        label="Val Bal-Acc")
        if "val_acc" in hist:
            ax_acc.plot(epochs, hist["val_acc"], color=PALETTE["accuracy"], lw=1.5,
                        linestyle="--", label="Val Acc")
        if "train_acc" in hist:
            ax_acc.plot(epochs, hist["train_acc"], color=PALETTE["f1_macro"], lw=1.0,
                        linestyle=":", label="Train Acc")
        ax_acc.axhline(0.5, color=PALETTE["chance"], linestyle="--", lw=0.8)
        _apply_base_style(ax_acc, f"S{result['subject_id']} Accuracy", "Epoch", "Score")
        ax_acc.legend(fontsize=7)
        ax_acc.set_ylim(0, 1.05)

    for ax in axes_flat[n*2:]:
        ax.set_visible(False)

    fig.suptitle("Per-Fold Training Curves", fontsize=12,
                 fontweight="bold", color=PALETTE["text"])
    fig.tight_layout()
    _save(fig, output_dir / "training_curves.png")


def _plot_training_summary(pairs, output_dir: Path) -> None:
    """Mean ± std training curves across all LOSO folds."""
    def _pad_and_stack(lists):
        max_len = max(len(l) for l in lists)
        padded = [l + [l[-1]] * (max_len - len(l)) for l in lists]
        return np.array(padded)

    keys = ["train_loss", "val_loss", "val_bal_acc", "val_acc"]
    arrays = {}
    for key in keys:
        data = [h[key] for _, h in pairs if key in h]
        if data:
            arrays[key] = _pad_and_stack(data)

    if not arrays:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    _fig_style(fig)

    epochs = range(1, arrays.get("train_loss", arrays.get("val_loss", [[0]]))
                   .shape[1] + 1) if arrays else range(1, 2)
    max_e  = max(len(list(epochs)), 1)
    x      = np.arange(1, max_e + 1)

    # Loss
    for key, color, label, ls in [
        ("train_loss", PALETTE["accuracy"], "Train Loss", "-"),
        ("val_loss",   PALETTE["balanced_accuracy"], "Val Loss", "--"),
    ]:
        if key in arrays:
            mean = arrays[key].mean(axis=0)
            std  = arrays[key].std(axis=0)
            ax1.plot(x, mean, color=color, lw=2, linestyle=ls, label=label)
            ax1.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
    _apply_base_style(ax1, "Mean Training Loss (all folds ± 1 std)", "Epoch", "Loss")
    ax1.legend(fontsize=9)

    # Accuracy
    for key, color, label, ls in [
        ("val_bal_acc", PALETTE["balanced_accuracy"], "Val Balanced Accuracy", "-"),
        ("val_acc",     PALETTE["accuracy"], "Val Accuracy", "--"),
        ("train_acc",   PALETTE["f1_macro"], "Train Accuracy", ":"),
    ]:
        if key in arrays:
            mean = arrays[key].mean(axis=0)
            std  = arrays[key].std(axis=0)
            ax2.plot(x, mean, color=color, lw=2, linestyle=ls, label=label)
            ax2.fill_between(x, np.clip(mean-std,0,1), np.clip(mean+std,0,1),
                             color=color, alpha=0.15)
    ax2.axhline(0.5, color=PALETTE["chance"], linestyle="--", lw=1.2, label="Chance")
    _apply_base_style(ax2, "Mean Accuracy Curves (all folds ± 1 std)", "Epoch", "Score")
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=9)

    fig.suptitle("Training Curve Summary — Mean ± Std Across All LOSO Folds",
                 fontsize=12, fontweight="bold", color=PALETTE["text"])
    fig.tight_layout()
    _save(fig, output_dir / "training_curves_summary.png")


# ---------------------------------------------------------------------------
# 12. Generalization Gap
# ---------------------------------------------------------------------------

def plot_generalization_gap(
    all_results: List[Dict],
    output_dir: Path,
) -> None:
    """
    Train vs. val/test performance per fold.
    Large gap → overfitting; val > train → underfitting or lucky splits.

    Requires 'train_metrics' in each result dict (final-epoch train metrics).
    Falls back to a note if unavailable.
    """
    if not _MPL:
        return
    has_train = any("train_metrics" in r for r in all_results)
    if not has_train:
        logger.info(
            "Generalization gap plot skipped — no 'train_metrics' in results.\n"
            "  Add final-epoch train metrics to each result dict in loso_runner.py:\n"
            "  result['train_metrics'] = {'balanced_accuracy': ..., 'accuracy': ...}"
        )
        return

    results = [_enrich_metrics(r) for r in all_results]
    sids = _subject_ids(results)

    metrics_to_show = [
        ("balanced_accuracy", "Balanced Accuracy"),
        ("accuracy", "Accuracy"),
    ]
    fig, axes = plt.subplots(1, len(metrics_to_show), figsize=(12, 4.5))
    _fig_style(fig)

    for ax, (metric, label) in zip(axes, metrics_to_show):
        train_vals = [r.get("train_metrics", {}).get(metric, float("nan")) for r in results]
        test_vals  = [r.get("metrics", {}).get(metric, float("nan")) for r in results]
        x = np.arange(len(sids))

        ax.bar(x - 0.2, train_vals, 0.35, label="Train", color=PALETTE["accuracy"], alpha=0.7)
        ax.bar(x + 0.2, test_vals,  0.35, label="Test (unseen subject)", color=PALETTE["balanced_accuracy"], alpha=0.7)

        for i, (tr, te) in enumerate(zip(train_vals, test_vals)):
            if not (np.isnan(tr) or np.isnan(te)):
                gap = tr - te
                ax.annotate(f"Δ{gap:+.2f}", (x[i], max(tr, te) + 0.015),
                            ha="center", fontsize=7, color="#64748B")

        ax.axhline(0.5, color=PALETTE["chance"], linestyle="--", lw=1)
        _apply_base_style(ax, f"Generalization Gap — {label}", "Subject", label)
        ax.set_xticks(x)
        ax.set_xticklabels(sids)
        ax.set_ylim(0, 1.12)
        ax.legend(fontsize=9)

    fig.suptitle("Train vs Test Performance per Fold (Generalization Gap)\n"
                 "Δ = train − test;  large positive Δ indicates overfitting",
                 fontsize=11, fontweight="bold", color=PALETTE["text"])
    fig.tight_layout()
    _save(fig, output_dir / "generalization_gap.png")


# ---------------------------------------------------------------------------
# 13. Error Analysis (FP / FN rates)
# ---------------------------------------------------------------------------

def plot_error_analysis(
    all_results: List[Dict],
    class_names: List[str],
    output_dir: Path,
) -> None:
    """
    Decompose the confusion matrix into miss rates per class and per subject.
    For P300 this means:
      - Target Miss Rate (FN) — system fails to detect a P300 when it occurred
      - NonTarget False Alarm Rate (FP) — system fires when no P300
    These map directly to BCI usability: FN kills throughput, FP kills accuracy.
    """
    if not _MPL:
        return
    results = [r for r in all_results if r.get("metrics", {}).get("confusion_matrix")]
    if not results:
        return

    sids = [str(r["subject_id"]) for r in results]
    fnrs, fprs_ = [], []  # false-negative-rate, false-positive-rate (per class)
    tprs_, tnrs = [], []

    for r in results:
        cm = np.array(r["metrics"]["confusion_matrix"], dtype=float)
        tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
        total_pos = tp + fn
        total_neg = tn + fp
        fnrs.append(fn / total_pos if total_pos > 0 else float("nan"))
        fprs_.append(fp / total_neg if total_neg > 0 else float("nan"))
        tprs_.append(tp / total_pos if total_pos > 0 else float("nan"))
        tnrs.append(tn / total_neg if total_neg > 0 else float("nan"))

    x = np.arange(len(sids))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _fig_style(fig)

    # Sensitivity / Specificity
    ax = axes[0]
    ax.bar(x - 0.2, tprs_, 0.35, label=f"Sensitivity (TPR) — detect {class_names[1]}",
           color="#22C55E", alpha=0.8)
    ax.bar(x + 0.2, tnrs, 0.35, label=f"Specificity (TNR) — reject {class_names[0]}",
           color="#2E86AB", alpha=0.8)
    ax.axhline(0.5, color=PALETTE["chance"], linestyle="--", lw=1)
    ax.axhline(CLINICAL_THRESHOLD, color="#94A3B8", linestyle=":", lw=1)
    _apply_base_style(ax, "Sensitivity & Specificity per Subject", "Subject", "Rate")
    ax.set_xticks(x); ax.set_xticklabels(sids)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)

    # Miss / False Alarm rates
    ax = axes[1]
    ax.bar(x - 0.2, fnrs, 0.35, label=f"Miss Rate (FNR) — missed {class_names[1]}",
           color="#EF4444", alpha=0.8)
    ax.bar(x + 0.2, fprs_, 0.35, label=f"False Alarm Rate (FPR) — false {class_names[1]}",
           color="#F59E0B", alpha=0.8)
    ax.axhline(1 - CLINICAL_THRESHOLD, color="#94A3B8", linestyle=":", lw=1,
               label=f"Error threshold ({1-CLINICAL_THRESHOLD:.1f})")
    _apply_base_style(ax, "Miss Rate & False Alarm Rate per Subject\n(lower is better)",
                      "Subject", "Rate")
    ax.set_xticks(x); ax.set_xticklabels(sids)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    fig.suptitle(
        "Error Analysis: Per-Class Error Rates per Unseen Subject\n"
        "Miss Rate (FNR) = P300 events the model fails to detect; "
        "False Alarm Rate (FPR) = NonTarget incorrectly classified as Target",
        fontsize=10, fontweight="bold", color=PALETTE["text"]
    )
    fig.tight_layout()
    _save(fig, output_dir / "error_analysis.png")


# ---------------------------------------------------------------------------
# 14. Cumulative Performance Plot
# ---------------------------------------------------------------------------

def plot_cumulative_performance(
    all_results: List[Dict],
    output_dir: Path,
) -> None:
    """
    Sort subjects by balanced accuracy.  Show what fraction of new users
    would exceed clinical / research thresholds — the deployment readiness curve.
    """
    if not _MPL:
        return
    results = [_enrich_metrics(r) for r in all_results]
    ba_vals = sorted(
        [v for v in _get(results, "balanced_accuracy") if not np.isnan(v)]
    )
    if not ba_vals:
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    _fig_style(fig)

    # Step CDF
    x = np.concatenate([[0], ba_vals, [1]])
    y = np.concatenate([[0], np.arange(1, len(ba_vals)+1) / len(ba_vals), [1]])
    ax.step(x, y, where="post", color=PALETTE["balanced_accuracy"], lw=2.5,
            label="Empirical CDF (subjects)")
    ax.fill_between(x, 0, y, step="post",
                    color=PALETTE["balanced_accuracy"], alpha=0.12)

    # Threshold verticals
    for thresh, color, label in [
        (0.5,  PALETTE["chance"], "Chance"),
        (CLINICAL_THRESHOLD, "#F59E0B", f"Clinical ({CLINICAL_THRESHOLD})"),
        (RESEARCH_THRESHOLD, "#22C55E", f"Research ({RESEARCH_THRESHOLD})"),
    ]:
        ax.axvline(thresh, color=color, linestyle="--", lw=1.5, label=label)
        # Fraction above threshold
        frac = np.mean(np.array(ba_vals) >= thresh)
        ax.text(thresh + 0.01, 0.05, f"{frac:.0%}", fontsize=8, color=color,
                fontweight="bold")

    _apply_base_style(ax,
        "Deployment Readiness: Cumulative Distribution of Balanced Accuracy\n"
        "% of new users who would exceed each threshold with zero calibration",
        "Balanced Accuracy",
        "Fraction of Subjects")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc="upper left")

    mean_ba = np.mean(ba_vals)
    ax.text(0.98, 0.5,
            f"n={len(ba_vals)} subjects\nMean BA = {mean_ba:.3f}\n"
            f"Above clinical: {np.mean(np.array(ba_vals) >= CLINICAL_THRESHOLD):.0%}\n"
            f"Above research: {np.mean(np.array(ba_vals) >= RESEARCH_THRESHOLD):.0%}",
            transform=ax.transAxes, ha="right", va="center",
            fontsize=9, color=PALETTE["text"],
            bbox=dict(boxstyle="round,pad=0.4", facecolor=PALETTE["grid"],
                      edgecolor="#CBD5E1"))

    fig.tight_layout()
    _save(fig, output_dir / "cumulative_performance.png")


# ---------------------------------------------------------------------------
# Console reporting (preserved from original)
# ---------------------------------------------------------------------------

def print_loso_table(all_results: List[Dict]) -> None:
    results = [_enrich_metrics(r) for r in all_results]
    header = (f"{'Subject':>8}  {'Acc':>8}  {'Bal Acc':>8}  "
              f"{'F1':>8}  {'MCC':>8}  {'AUC':>8}  {'N':>6}")
    print("\n" + "=" * len(header))
    print("Per-subject LOSO results")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for result in results:
        sid = result["subject_id"]
        m   = result.get("metrics", {})
        if not m:
            print(f"{sid:>8}  {'(no data)':>52}")
            continue
        print(
            f"{sid:>8}  "
            f"{m.get('accuracy',          float('nan')):>8.4f}  "
            f"{m.get('balanced_accuracy', float('nan')):>8.4f}  "
            f"{m.get('f1_macro',          float('nan')):>8.4f}  "
            f"{m.get('mcc',               float('nan')):>8.4f}  "
            f"{m.get('roc_auc',           float('nan')):>8.4f}  "
            f"{m.get('n_samples',         0):>6}"
        )


def print_aggregate_table(aggregate: Dict) -> None:
    metrics_to_print = [
        ("accuracy",          "Accuracy"),
        ("balanced_accuracy", "Balanced Accuracy"),
        ("precision",         "Precision (macro)"),
        ("recall",            "Recall (macro)"),
        ("f1_macro",          "F1 (macro)"),
        ("roc_auc",           "ROC-AUC"),
    ]
    print("\n" + "=" * 65)
    print("Aggregate LOSO results  (mean ± std  [min, max])")
    print("=" * 65)
    for key, label in metrics_to_print:
        stats = aggregate.get(key, {})
        mean  = stats.get("mean", float("nan"))
        std   = stats.get("std",  float("nan"))
        mn    = stats.get("min",  float("nan"))
        mx    = stats.get("max",  float("nan"))
        print(f"  {label:<22}  {mean:.4f} ± {std:.4f}  [{mn:.4f}, {mx:.4f}]")

    # MCC aggregate (not in original aggregate dict — compute here)
    print()


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_results_csv(all_results: List[Dict], path: Path) -> None:
    results = [_enrich_metrics(r) for r in all_results]
    scalar_keys = [
        "accuracy", "balanced_accuracy", "precision",
        "recall", "f1_macro", "mcc", "roc_auc", "n_samples",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject_id"] + scalar_keys)
        writer.writeheader()
        for result in results:
            row = {"subject_id": result["subject_id"]}
            for k in scalar_keys:
                row[k] = result.get("metrics", {}).get(k, "")
            writer.writerow(row)
    logger.info("Results CSV written to %s", path)


# ---------------------------------------------------------------------------
# Master convenience function
# ---------------------------------------------------------------------------

def run_full_analytics(
    all_results: List[Dict],
    aggregate: Dict,
    class_names: List[str],
    output_dir: Path,
) -> None:
    """
    Generate every plot and save the CSV in one call.

    Call from train.py after run_loso() returns:

        from evaluation.analytics import run_full_analytics
        run_full_analytics(all_results, aggregate, CLASS_NAMES, cfg.logs_dir())

    Parameters
    ----------
    all_results : list of per-subject result dicts from loso_runner.run_loso
    aggregate   : dict from evaluation.metrics.aggregate_metrics
    class_names : e.g. ["NonTarget", "Target"]
    output_dir  : directory to write all outputs into
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("Running full analytics suite…")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}\n")

    steps = [
        ("LOSO metrics overview",        lambda: plot_loso_metrics_overview(all_results, output_dir)),
        ("Balanced accuracy detail",     lambda: plot_balanced_accuracy_detail(all_results, output_dir)),
        ("Confusion matrices",           lambda: plot_confusion_matrices(all_results, class_names, output_dir)),
        ("Aggregate radar",              lambda: plot_aggregate_radar(aggregate, output_dir)),
        ("Metric distributions",         lambda: plot_metric_distributions(all_results, output_dir)),
        ("Subject × metric heatmap",     lambda: plot_subject_metric_heatmap(all_results, output_dir)),
        ("ROC curves",                   lambda: plot_roc_curves(all_results, output_dir)),
        ("Precision–Recall curves",      lambda: plot_precision_recall_curves(all_results, output_dir)),
        ("Calibration curves",           lambda: plot_calibration_curves(all_results, output_dir)),
        ("Training curves",              lambda: plot_training_curves(all_results, output_dir)),
        ("Generalization gap",           lambda: plot_generalization_gap(all_results, output_dir)),
        ("Error analysis",               lambda: plot_error_analysis(all_results, class_names, output_dir)),
        ("Cumulative performance",       lambda: plot_cumulative_performance(all_results, output_dir)),
        ("Results CSV",                  lambda: save_results_csv(all_results, output_dir / "loso_results.csv")),
    ]

    for name, fn in steps:
        try:
            fn()
            print(f"  ✓  {name}")
        except Exception as exc:
            print(f"  ✗  {name}: {exc}")
            logger.exception("Analytics step '%s' failed", name)

    print(f"\n{'='*60}")
    print("Analytics complete.")
    print_loso_table(all_results)
    print_aggregate_table(aggregate)