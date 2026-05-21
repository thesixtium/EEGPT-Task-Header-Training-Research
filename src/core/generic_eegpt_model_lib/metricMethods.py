import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pyhealth.metrics import binary_metrics_fn, multiclass_metrics_fn


def get_latest_metrics_csv(save_dir: str, model_name: str) -> str:
    """
    Returns the path to metrics.csv from the highest-numbered version folder
    inside {save_dir}/{model_name}_csv/.
    """
    csv_log_dir = os.path.join(save_dir, f"{model_name}_csv")

    versions = [
        d for d in os.listdir(csv_log_dir)
        if os.path.isdir(os.path.join(csv_log_dir, d)) and d.startswith("version_")
    ]

    if not versions:
        raise FileNotFoundError(f"No version folders found in {csv_log_dir}")

    latest = max(versions, key=lambda v: int(v.split("_")[1]))

    return os.path.join(csv_log_dir, latest, "metrics.csv")

def metrics_display( csv_path, model_name, img_path ):
    df = pd.read_csv(csv_path)

    # --- Separate rows: train rows have train_loss, valid rows have valid_loss ---
    train_df = df[df["train_loss"].notna()].copy()
    valid_df = df[df["valid_loss"].notna()].copy()
    lr_df    = df[df["lr-AdamW"].notna()].copy()

    # Use 'step' as the x-axis throughout
    x_train = train_df["step"]
    x_valid = valid_df["step"]
    x_lr    = lr_df["step"]

    # -----------------------------------------------------------------------
    # Layout: 4 rows × 3 cols  (12 panels, last one unused → spans)
    # -----------------------------------------------------------------------
    fig = plt.figure(figsize=(18, 20))
    fig.patch.set_facecolor("#0f1117")

    gs = gridspec.GridSpec(
        4, 3,
        figure=fig,
        hspace=0.55,
        wspace=0.35,
        left=0.07, right=0.97,
        top=0.93, bottom=0.05,
    )

    ACCENT  = "#7c83fd"   # indigo-blue for training
    ACCENT2 = "#f77f00"   # orange for validation
    GREEN   = "#56cfe1"   # teal for balanced accuracy / kappa
    PINK    = "#ff6b9d"   # pink for weighted
    LIME    = "#a9e34b"   # lime for macro
    YELLOW  = "#ffd166"   # yellow for micro
    LR_COL  = "#c77dff"   # purple for LR

    GRID_KW  = dict(color="#2a2d3e", linewidth=0.6, linestyle="--")
    TEXT_KW  = dict(color="#e0e0e0")
    TICK_KW  = dict(colors="#9a9db0")

    def style_ax(ax, title):
        ax.set_facecolor("#1a1c2e")
        ax.set_title(title, color="#e0e0e0", fontsize=10, fontweight="bold", pad=6)
        ax.tick_params(colors="#9a9db0", labelsize=7.5)
        ax.xaxis.label.set_color("#9a9db0")
        ax.yaxis.label.set_color("#9a9db0")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2d3e")
        ax.grid(**GRID_KW)
        ax.set_xlabel("Step", fontsize=8)

    def plot_line(ax, x, y, color, label=None, lw=1.6, alpha=1.0):
        mask = y.notna()
        ax.plot(x[mask], y[mask], color=color, linewidth=lw,
                label=label, alpha=alpha, solid_capstyle="round")


    # ── 1. Loss (train + valid) ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    plot_line(ax, x_train, train_df["train_loss"], ACCENT,  "Train")
    plot_line(ax, x_valid, valid_df["valid_loss"], ACCENT2, "Valid")
    ax.legend(fontsize=7.5, facecolor="#1a1c2e", labelcolor="#e0e0e0", edgecolor="#2a2d3e")
    style_ax(ax, "Loss")

    # ── 2. Accuracy (train + valid_acc) ─────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    plot_line(ax, x_train, train_df["train_acc"],  ACCENT,  "Train")
    plot_line(ax, x_valid, valid_df["valid_acc"],  ACCENT2, "Valid")
    ax.legend(fontsize=7.5, facecolor="#1a1c2e", labelcolor="#e0e0e0", edgecolor="#2a2d3e")
    style_ax(ax, "Accuracy")

    # ── 3. Learning rate ─────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    plot_line(ax, x_lr, lr_df["lr-AdamW"], LR_COL)
    style_ax(ax, "Learning Rate (AdamW)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2e}"))

    # ── 4. Valid accuracy variants ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    plot_line(ax, x_valid, valid_df["valid_accuracy"],          ACCENT2, "Accuracy")
    plot_line(ax, x_valid, valid_df["valid_balanced_accuracy"], GREEN,   "Balanced")
    ax.legend(fontsize=7.5, facecolor="#1a1c2e", labelcolor="#e0e0e0", edgecolor="#2a2d3e")
    style_ax(ax, "Valid Accuracy vs Balanced Accuracy")

    # ── 5. Cohen's Kappa ─────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    plot_line(ax, x_valid, valid_df["valid_cohen_kappa"], GREEN)
    ax.axhline(0, color="#555", linewidth=0.8, linestyle=":")
    style_ax(ax, "Cohen's Kappa")

    # ── 6. F1 scores ─────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    plot_line(ax, x_valid, valid_df["valid_f1_macro"],    LIME,   "Macro")
    plot_line(ax, x_valid, valid_df["valid_f1_micro"],    YELLOW, "Micro")
    plot_line(ax, x_valid, valid_df["valid_f1_weighted"], PINK,   "Weighted")
    ax.legend(fontsize=7.5, facecolor="#1a1c2e", labelcolor="#e0e0e0", edgecolor="#2a2d3e")
    style_ax(ax, "F1 Scores")

    # ── 7. Data loading avg ───────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    plot_line(ax, x_train, train_df["data_avg"], ACCENT)
    style_ax(ax, "Data Load Time (avg)")

    # ── 8. Data loading max ───────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    plot_line(ax, x_train, train_df["data_max"], ACCENT2)
    style_ax(ax, "Data Load Time (max)")

    # ── 9. Data loading min ───────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 2])
    plot_line(ax, x_train, train_df["data_min"], GREEN)
    style_ax(ax, "Data Load Time (min)")

    # ── 10. Data loading std ──────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[3, 0])
    plot_line(ax, x_train, train_df["data_std"], PINK)
    style_ax(ax, "Data Load Time (std)")

    # ── 11. Train loss vs Valid loss  (wider span) ────────────────────────────────
    ax = fig.add_subplot(gs[3, 1:])
    plot_line(ax, x_train, train_df["train_loss"], ACCENT,  "Train Loss")
    plot_line(ax, x_valid, valid_df["valid_loss"], ACCENT2, "Valid Loss")
    ax.legend(fontsize=8, facecolor="#1a1c2e", labelcolor="#e0e0e0", edgecolor="#2a2d3e")
    style_ax(ax, "Train vs Valid Loss (overview)")

    # ── Title ─────────────────────────────────────────────────────────────────────
    fig.suptitle(
        "Training Metrics Dashboard",
        fontsize=15, fontweight="bold",
        color="#e0e0e0", y=0.965,
    )

    out_path = img_path / f"{model_name}_training_metrics.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"Saved → {out_path}")
    #plt.show()

def get_metrics(output, target, metrics, is_binary, threshold=0.5):
    if is_binary:
        if 'roc_auc' not in metrics or sum(target) * (len(target) - sum(target)) != 0:  # to prevent all 0 or all 1 and raise the AUROC error
            results = binary_metrics_fn(
                target,
                output,
                metrics=metrics,
                threshold=threshold,
            )
        else:
            results = {
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "pr_auc": 0.0,
                "roc_auc": 0.0,
            }
    else:
        results = multiclass_metrics_fn(
            target, output, metrics=metrics
        )
    return results