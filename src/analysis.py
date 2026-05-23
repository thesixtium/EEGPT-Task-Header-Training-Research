import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from pathlib import Path
import sys

# =========================================================
# CONFIG
# =========================================================
CSV_PATH = "ga_eegpt_results.csv"   # change if needed
OUT_DIR  = Path("ga_analysis")
OUT_DIR.mkdir(exist_ok=True)

ACCENT   = "#00C9A7"
ACCENT2  = "#FF6B6B"
ACCENT3  = "#FFD93D"
BG       = "#0F1117"
PANEL    = "#1A1D2E"
TEXT     = "#E8EAF0"
SUBTEXT  = "#8B8FA8"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    SUBTEXT,
    "axes.labelcolor":   TEXT,
    "axes.titlecolor":   TEXT,
    "xtick.color":       SUBTEXT,
    "ytick.color":       SUBTEXT,
    "text.color":        TEXT,
    "grid.color":        "#2A2D3E",
    "grid.linewidth":    0.6,
    "font.family":       "monospace",
    "figure.dpi":        150,
})

PALETTE = [ACCENT, ACCENT2, ACCENT3, "#A78BFA", "#60A5FA", "#F472B6"]


# =========================================================
# LOAD
# =========================================================
def load(path):
    df = pd.read_csv(path)
    required = {"lr_scheduler_name", "target_name", "max_epochs",
                "max_learning_rate", "gamma", "fitness_score", "generation"}
    missing = required - set(df.columns)
    if missing:
        print(f"[WARN] Missing columns: {missing}")
    return df


# =========================================================
# 1. BOXPLOTS – categorical variables vs fitness
# =========================================================
def plot_categorical_boxplots(df, out_dir):
    cat_cols = ["lr_scheduler_name", "target_name"]
    cat_cols = [c for c in cat_cols if c in df.columns]

    fig, axes = plt.subplots(1, len(cat_cols), figsize=(7 * len(cat_cols), 6))
    fig.patch.set_facecolor(BG)
    if len(cat_cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cat_cols):
        categories = df[col].unique()
        data_by_cat = [df.loc[df[col] == c, "fitness_score"].values for c in categories]
        colors = PALETTE[:len(categories)]

        bp = ax.boxplot(
            data_by_cat,
            patch_artist=True,
            medianprops=dict(color=BG, linewidth=2),
            whiskerprops=dict(color=SUBTEXT),
            capprops=dict(color=SUBTEXT),
            flierprops=dict(marker="o", color=ACCENT2, markersize=4, alpha=0.7),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

        # overlay individual points
        for i, (cat_data, color) in enumerate(zip(data_by_cat, colors), start=1):
            jitter = np.random.uniform(-0.15, 0.15, size=len(cat_data))
            ax.scatter(np.full_like(cat_data, i) + jitter, cat_data,
                       color=color, s=22, alpha=0.6, zorder=5)

        ax.set_xticks(range(1, len(categories) + 1))
        ax.set_xticklabels(categories, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Fitness Score (valid_acc)", fontsize=10)
        ax.set_title(f"{col}  vs  Fitness", fontsize=12, pad=10)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)

    fig.suptitle("Categorical Hyperparameters vs Fitness", fontsize=14,
                 fontweight="bold", y=1.02)
    fig.tight_layout()
    out = out_dir / "01_categorical_boxplots.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved → {out}")


# =========================================================
# 2. SCATTER PLOTS – numeric variables vs fitness
# =========================================================
def plot_numeric_scatters(df, out_dir):
    num_cols = ["max_epochs", "max_learning_rate", "gamma", "generation"]
    num_cols = [c for c in num_cols if c in df.columns]

    ncols = 2
    nrows = int(np.ceil(len(num_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    fig.patch.set_facecolor(BG)
    axes = np.array(axes).flatten()

    for ax, col in zip(axes, num_cols):
        x = df[col].values
        y = df["fitness_score"].values

        # colour points by fitness
        sc = ax.scatter(x, y, c=y, cmap="plasma", s=35, alpha=0.85,
                        edgecolors="none", zorder=4)

        # trend line
        if len(x) > 2:
            z = np.polyfit(x, y, 1)
            p = np.poly1d(z)
            xs = np.linspace(x.min(), x.max(), 200)
            ax.plot(xs, p(xs), color=ACCENT, linewidth=1.5,
                    linestyle="--", alpha=0.8, zorder=5)

        # correlation annotation
        corr = np.corrcoef(x, y)[0, 1]
        ax.text(0.97, 0.05, f"r = {corr:+.3f}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9, color=ACCENT3)

        cb = fig.colorbar(sc, ax=ax, pad=0.02)
        cb.ax.tick_params(labelsize=7, colors=SUBTEXT)
        cb.outline.set_edgecolor(SUBTEXT)

        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel("Fitness Score", fontsize=10)
        ax.set_title(f"{col}  vs  Fitness", fontsize=11, pad=8)
        ax.yaxis.grid(True)
        ax.set_axisbelow(True)

        if col == "max_learning_rate":
            ax.set_xscale("log")

    # hide any unused panels
    for ax in axes[len(num_cols):]:
        ax.set_visible(False)

    fig.suptitle("Numeric Hyperparameters vs Fitness", fontsize=14,
                 fontweight="bold")
    fig.tight_layout()
    out = out_dir / "02_numeric_scatters.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved → {out}")


# =========================================================
# 3. CORRELATION MATRIX
# =========================================================
def plot_correlation_matrix(df, out_dir):
    # encode categoricals as integers for correlation
    df_enc = df.copy()
    for col in ["lr_scheduler_name", "target_name"]:
        if col in df_enc.columns:
            df_enc[col] = pd.factorize(df_enc[col])[0]

    keep = ["lr_scheduler_name", "target_name", "max_epochs",
            "max_learning_rate", "gamma", "generation", "fitness_score"]
    keep = [c for c in keep if c in df_enc.columns]
    corr = df_enc[keep].corr()

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(PANEL)

    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmin=-1, vmax=1, center=0,
        annot=True, fmt=".2f", annot_kws={"size": 9, "color": TEXT},
        linewidths=0.5, linecolor="#2A2D3E",
        ax=ax,
        cbar_kws={"shrink": 0.8},
        square=True,
    )
    ax.set_title("Correlation Matrix\n(categorical cols label-encoded)",
                 fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(axis="x", rotation=35, labelsize=9)
    ax.tick_params(axis="y", rotation=0,  labelsize=9)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors=SUBTEXT, labelsize=8)
    cbar.outline.set_edgecolor(SUBTEXT)

    fig.tight_layout()
    out = out_dir / "03_correlation_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved → {out}")


# =========================================================
# 4. FITNESS OVER GENERATIONS
# =========================================================
def plot_fitness_over_generations(df, out_dir):
    if "generation" not in df.columns:
        return

    gen_avg  = df.groupby("generation")["fitness_score"].mean()
    gen_best = df.groupby("generation")["fitness_score"].max()
    gen_best_overall = gen_best.cummax()

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG)

    ax.fill_between(gen_avg.index, gen_avg.values,
                    alpha=0.15, color=ACCENT)
    ax.plot(gen_avg.index, gen_avg.values,
            color=ACCENT, linewidth=1.8, label="Avg fitness")
    ax.plot(gen_best.index, gen_best.values,
            color=ACCENT2, linewidth=1.8, linestyle="--", label="Best per gen")
    ax.plot(gen_best_overall.index, gen_best_overall.values,
            color=ACCENT3, linewidth=2.5, label="Best overall (cummax)")

    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("Fitness Score (valid_acc)", fontsize=11)
    ax.set_title("GA Fitness Over Generations", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=9)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = out_dir / "04_fitness_over_generations.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved → {out}")


# =========================================================
# 5. TOP-N RUNS TABLE (printed + saved as image)
# =========================================================
def plot_top_runs(df, out_dir, n=10):
    top = (df.sort_values("fitness_score", ascending=False)
             .drop_duplicates()
             .head(n)
             .reset_index(drop=True))

    col_order = ["fitness_score", "lr_scheduler_name", "target_name",
                 "max_epochs", "max_learning_rate", "gamma", "generation"]
    col_order = [c for c in col_order if c in top.columns]
    top = top[col_order]

    print(f"\n{'='*60}")
    print(f"  TOP {n} RUNS BY FITNESS")
    print(f"{'='*60}")
    print(top.to_string(index=True))

    fig, ax = plt.subplots(figsize=(max(10, len(col_order) * 1.8), n * 0.5 + 1.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    cell_text = []
    for _, row in top.iterrows():
        formatted = []
        for c in col_order:
            v = row[c]
            if isinstance(v, float):
                formatted.append(f"{v:.5f}")
            else:
                formatted.append(str(v))
        cell_text.append(formatted)

    headers = [c.replace("_", "\n") for c in col_order]
    tbl = ax.table(
        cellText=cell_text,
        colLabels=headers,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.6)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("#2A2D3E")
        if row == 0:
            cell.set_facecolor(ACCENT)
            cell.set_text_props(color=BG, fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#1F2235")
            cell.set_text_props(color=TEXT)
        else:
            cell.set_facecolor(PANEL)
            cell.set_text_props(color=TEXT)

    ax.set_title(f"Top {n} Runs by Fitness Score",
                 fontsize=13, fontweight="bold", color=TEXT, pad=12)

    fig.tight_layout()
    out = out_dir / "05_top_runs_table.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved → {out}")


# =========================================================
# 6. PAIRPLOT (numeric columns coloured by scheduler)
# =========================================================
def plot_pairplot(df, out_dir):
    num_cols = [c for c in ["max_epochs", "max_learning_rate", "gamma", "fitness_score"]
                if c in df.columns]

    if "lr_scheduler_name" not in df.columns or len(num_cols) < 2:
        return

    schedulers = df["lr_scheduler_name"].unique()
    color_map = {s: PALETTE[i % len(PALETTE)] for i, s in enumerate(schedulers)}

    n = len(num_cols)
    fig, axes = plt.subplots(n, n, figsize=(3.5 * n, 3.5 * n))
    fig.patch.set_facecolor(BG)

    for i, col_y in enumerate(num_cols):
        for j, col_x in enumerate(num_cols):
            ax = axes[i][j]
            ax.set_facecolor(PANEL)

            for sched in schedulers:
                sub = df[df["lr_scheduler_name"] == sched]
                if i == j:
                    ax.hist(sub[col_x].values, bins=10,
                            color=color_map[sched], alpha=0.55,
                            edgecolor="none")
                else:
                    ax.scatter(sub[col_x].values, sub[col_y].values,
                               color=color_map[sched], s=18, alpha=0.7,
                               edgecolors="none", label=sched)

            if col_x == "max_learning_rate":
                ax.set_xscale("log")
            if col_y == "max_learning_rate":
                ax.set_yscale("log")

            if i == n - 1:
                ax.set_xlabel(col_x, fontsize=8, labelpad=4)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(col_y, fontsize=8, labelpad=4)
            else:
                ax.set_yticklabels([])

            ax.tick_params(labelsize=6)
            for spine in ax.spines.values():
                spine.set_edgecolor(SUBTEXT)

    # legend
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                           markerfacecolor=color_map[s], markersize=7, label=s)
               for s in schedulers]
    fig.legend(handles=handles, title="lr_scheduler", loc="upper right",
               fontsize=8, title_fontsize=9,
               facecolor=PANEL, edgecolor=SUBTEXT, labelcolor=TEXT)

    fig.suptitle("Pairplot – Numeric Hyperparameters (coloured by scheduler)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = out_dir / "06_pairplot.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"Saved → {out}")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    path = CSV_PATH
    print(f"Loading: {path}")
    df = load(path)
    print(f"Rows: {len(df)}  |  Columns: {list(df.columns)}\n")

    plot_categorical_boxplots(df, OUT_DIR)
    plot_numeric_scatters(df, OUT_DIR)
    plot_correlation_matrix(df, OUT_DIR)
    plot_fitness_over_generations(df, OUT_DIR)
    plot_top_runs(df, OUT_DIR, n=10)
    plot_pairplot(df, OUT_DIR)

    print(f"\nAll plots saved to: {OUT_DIR.resolve()}")