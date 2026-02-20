import json
import matplotlib.pyplot as plt
import numpy as np

# Your log data (50 epochs)
with open(r"C:\PS\Sleep-Staging\results\temporal spectral fusion\concat\training_metrics_fusion.jsonl", 'r') as f:
    log_data = f.read()
logs = [json.loads(line) for line in log_data.strip().split('\n') if line.strip()]
n_epochs = len(logs)
epochs = list(range(1, n_epochs + 1))

# FIXED: Extract metrics for SINGLE run (no averaging needed for 50 entries)
train_loss = [m['train_loss'] for m in logs]
accuracy = [m['accuracy'] for m in logs]
f1_weighted = [m['f1_weighted'] for m in logs]
f1_macro = [m['f1_macro'] for m in logs]
kappa = [m['kappa'] for m in logs]
precision = [m['precision'] for m in logs]
recall = [m['recall'] for m in logs]
specificity = [m['specificity'] for m in logs]

# Individual plots - 7 separate figures
plots = [
    (train_loss, 'Training Loss', 'Loss', min),
    (accuracy, 'Accuracy', 'Accuracy', max),
    (f1_weighted, 'F1-Weighted', 'F1-Weighted', max),
    (kappa, "Cohen's Kappa", 'Kappa', max),
    (precision, 'Precision', 'Precision', max),
    (recall, 'Recall', 'Recall', max),
    (specificity, 'Specificity', 'Specificity', max)
]

for data, title, ylabel, extremum_func in plots:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(epochs, data, linewidth=4, marker='o', markersize=8, color='tab:blue')
    
    ax.set_title(f'{title} Over Training Epochs', fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(labelsize=12)
    
    # Annotate best value
    best_val = extremum_func(data)
    best_epoch = epochs[data.index(best_val)]
    ax.annotate(f'{ "Min" if extremum_func==min else "Peak" }: {best_val:.4f}\nEpoch: {best_epoch}', 
                xy=(best_epoch, best_val), 
                xytext=(15, 15), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8),
                fontsize=13, fontweight='bold', ha='left')
    
    plt.tight_layout()
    plt.show()

# Quick summary
print(f"\n=== FINAL PERFORMANCE (Epoch {n_epochs}) ===")
print(f"Accuracy: {accuracy[-1]:.4f}  |  Peak: {max(accuracy):.4f}")
print(f"F1-Weighted: {f1_weighted[-1]:.4f}  |  Peak: {max(f1_weighted):.4f}")
print(f"Kappa: {kappa[-1]:.4f}  |  Peak: {max(kappa):.4f}")
print(f"Train Loss: {train_loss[-1]:.4f}")

# """
# visualize_training.py
# ─────────────────────
# Full training metrics visualization for sleep staging fusion model.

# Usage:
#     python visualize_training.py path/to/training_metrics_fusion.jsonl
#     python visualize_training.py path/to/training_metrics_fusion.jsonl --out ./results

# Outputs (saved to --out directory, default = same folder as the JSONL):
#     01_loss.png                 Training loss curve with phase annotations
#     02_accuracy.png             Accuracy + rolling mean
#     03_f1_scores.png            Weighted & macro F1 with best-epoch markers
#     04_precision_recall.png     Precision vs recall
#     05_kappa_specificity.png    Cohen's κ & specificity
#     06_f1_gap.png               Macro–weighted F1 gap (class imbalance proxy)
#     07_learning_velocity.png    Per-epoch ΔF1 bar chart
#     08_correlation_heatmap.png  Pearson correlation between all metrics
#     09_radar.png                Radar chart of best-epoch vs final-epoch metrics
#     10_overview_grid.png        All key metrics in one 3×3 summary grid
# """

# import argparse
# import json
# import sys
# from pathlib import Path

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import matplotlib.ticker as ticker
# import matplotlib.gridspec as gridspec
# import numpy as np
# import pandas as pd
# import seaborn as sns
# from matplotlib.lines import Line2D

# # ── Palette ────────────────────────────────────────────────────────────────
# BG        = "#080d14"
# SURFACE   = "#0e1620"
# BORDER    = "#1a2535"
# TEAL      = "#00d4b8"
# CYAN      = "#38bdf8"
# AMBER     = "#f59e0b"
# ROSE      = "#f43f5e"
# VIOLET    = "#a78bfa"
# GREEN     = "#4ade80"
# ORANGE    = "#fb923c"
# PINK      = "#e879f9"
# MUTED     = "#64748b"
# MUTED2    = "#94a3b8"
# TEXT      = "#e2e8f0"

# PHASE_COLORS = {
#     "Rapid descent\n(epochs 0–30)":   "#00d4b820",
#     "Refinement\n(epochs 30–80)":     "#38bdf820",
#     "Convergence\n(epochs 80–149)":   "#a78bfa20",
# }
# PHASE_EDGE = ["#00d4b8", "#38bdf8", "#a78bfa"]
# PHASE_BOUNDS = [(0, 30), (30, 80), (80, 149)]

# METRIC_META = {
#     "train_loss":   dict(label="Train Loss",       color=ROSE,   fmt=".4f", lower_is_better=True),
#     "accuracy":     dict(label="Accuracy",          color=TEAL,   fmt=".4f", lower_is_better=False),
#     "f1_weighted":  dict(label="F1 (weighted)",     color=CYAN,   fmt=".4f", lower_is_better=False),
#     "f1_macro":     dict(label="F1 (macro)",        color=VIOLET, fmt=".4f", lower_is_better=False),
#     "precision":    dict(label="Precision",         color=ORANGE, fmt=".4f", lower_is_better=False),
#     "recall":       dict(label="Recall",            color=PINK,   fmt=".4f", lower_is_better=False),
#     "kappa":        dict(label="Cohen's κ",         color=AMBER,  fmt=".4f", lower_is_better=False),
#     "specificity":  dict(label="Specificity",       color=GREEN,  fmt=".4f", lower_is_better=False),
# }


# # ── Helpers ─────────────────────────────────────────────────────────────────

# def load_data(path: Path) -> pd.DataFrame:
#     rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
#     df = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
#     return df


# def apply_base_style():
#     plt.rcParams.update({
#         "figure.facecolor":   BG,
#         "axes.facecolor":     SURFACE,
#         "axes.edgecolor":     BORDER,
#         "axes.labelcolor":    MUTED2,
#         "axes.titlecolor":    TEXT,
#         "axes.titlesize":     11,
#         "axes.labelsize":     9,
#         "axes.spines.top":    False,
#         "axes.spines.right":  False,
#         "axes.grid":          True,
#         "grid.color":         BORDER,
#         "grid.linewidth":     0.5,
#         "grid.alpha":         0.6,
#         "xtick.color":        MUTED,
#         "ytick.color":        MUTED,
#         "xtick.labelsize":    8,
#         "ytick.labelsize":    8,
#         "legend.facecolor":   SURFACE,
#         "legend.edgecolor":   BORDER,
#         "legend.labelcolor":  MUTED2,
#         "legend.fontsize":    8,
#         "text.color":         TEXT,
#         "font.family":        "monospace",
#         "lines.linewidth":    1.8,
#         "lines.solid_capstyle": "round",
#     })


# def save(fig: plt.Figure, path: Path, name: str):
#     out = path / name
#     fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
#     plt.close(fig)
#     print(f"  ✓  {out.name}")


# def add_phase_bands(ax, alpha=0.08):
#     """Shade the three training phases."""
#     for (lo, hi), color, label in zip(PHASE_BOUNDS, PHASE_EDGE, PHASE_COLORS.keys()):
#         ax.axvspan(lo, hi, color=color, alpha=alpha, zorder=0)


# def mark_best(ax, epochs, values, lower_is_better=False, color=TEAL):
#     """Star the best epoch."""
#     best_idx = int(np.argmin(values) if lower_is_better else np.argmax(values))
#     bx, by = epochs[best_idx], values[best_idx]
#     ax.scatter([bx], [by], marker="*", s=220, color=color,
#                zorder=10, edgecolors="white", linewidths=0.4)
#     ax.annotate(
#         f"  best: {by:.4f}\n  (ep {bx})",
#         xy=(bx, by), xytext=(8, 0), textcoords="offset points",
#         fontsize=7, color=color, va="center",
#     )


# def smooth(values, window=7):
#     """Simple moving average."""
#     s = pd.Series(values)
#     return s.rolling(window, center=True, min_periods=1).mean().to_numpy()


# def subtitle(ax, text):
#     ax.set_title(text, fontsize=9, color=MUTED, pad=4)


# def section_title(ax, text):
#     ax.set_title(text, fontsize=11, color=TEXT, fontweight="bold", pad=8)


# # ── Plot 1 — Loss ───────────────────────────────────────────────────────────

# def plot_loss(df, out):
#     fig, ax = plt.subplots(figsize=(12, 4.5))
#     ep = df["epoch"].to_numpy()
#     loss = df["train_loss"].to_numpy()

#     add_phase_bands(ax)

#     ax.fill_between(ep, loss, alpha=0.12, color=ROSE)
#     ax.plot(ep, loss, color=ROSE, lw=2, label="Focal Loss")
#     ax.plot(ep, smooth(loss, 9), color=ROSE, lw=1, ls="--", alpha=0.5, label="Smoothed (9-ep MA)")
#     mark_best(ax, ep, loss, lower_is_better=True, color=ROSE)

#     # Phase labels at top
#     for (lo, hi), color, label in zip(PHASE_BOUNDS, PHASE_EDGE, PHASE_COLORS.keys()):
#         ax.text((lo + hi) / 2, ax.get_ylim()[1] if ax.get_ylim()[1] else loss.max(),
#                 label.replace("\n", " "), ha="center", va="bottom",
#                 fontsize=7, color=color, style="italic")

#     ax.set_xlabel("Epoch")
#     ax.set_ylabel("Loss")
#     section_title(ax, "Training Loss — Focal Loss (γ=2)")
#     ax.legend(loc="upper right")
#     fig.tight_layout()
#     save(fig, out, "01_loss.png")


# # ── Plot 2 — Accuracy ───────────────────────────────────────────────────────

# def plot_accuracy(df, out):
#     fig, ax = plt.subplots(figsize=(12, 4.5))
#     ep = df["epoch"].to_numpy()
#     acc = df["accuracy"].to_numpy()

#     add_phase_bands(ax)
#     ax.fill_between(ep, acc, alpha=0.10, color=TEAL)
#     ax.plot(ep, acc, color=TEAL, lw=2, label="Accuracy")
#     ax.plot(ep, smooth(acc, 9), color=TEAL, lw=1.2, ls="--", alpha=0.55, label="9-ep MA")
#     mark_best(ax, ep, acc, color=TEAL)

#     ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
#     ax.set_xlabel("Epoch")
#     ax.set_ylabel("Accuracy")
#     section_title(ax, "Validation Accuracy")
#     ax.legend(loc="lower right")
#     fig.tight_layout()
#     save(fig, out, "02_accuracy.png")


# # ── Plot 3 — F1 Scores ──────────────────────────────────────────────────────

# def plot_f1(df, out):
#     fig, ax = plt.subplots(figsize=(12, 4.5))
#     ep = df["epoch"].to_numpy()
#     fw = df["f1_weighted"].to_numpy()
#     fm = df["f1_macro"].to_numpy()

#     add_phase_bands(ax)
#     ax.fill_between(ep, fm, fw, alpha=0.08, color=CYAN, label="_fill")
#     ax.plot(ep, fw, color=CYAN,   lw=2,   label="F1 Weighted")
#     ax.plot(ep, fm, color=VIOLET, lw=2,   label="F1 Macro")
#     ax.plot(ep, smooth(fw, 9), color=CYAN,   lw=1, ls="--", alpha=0.45)
#     ax.plot(ep, smooth(fm, 9), color=VIOLET, lw=1, ls="--", alpha=0.45)

#     best_fw = int(np.argmax(fw))
#     best_fm = int(np.argmax(fm))
#     ax.axvline(best_fw, color=CYAN,   lw=0.8, ls=":", alpha=0.7)
#     ax.axvline(best_fm, color=VIOLET, lw=0.8, ls=":", alpha=0.7)
#     ax.scatter([ep[best_fw]], [fw[best_fw]], marker="*", s=200, color=CYAN,   zorder=10, edgecolors="white", lw=0.4)
#     ax.scatter([ep[best_fm]], [fm[best_fm]], marker="*", s=200, color=VIOLET, zorder=10, edgecolors="white", lw=0.4)
#     ax.annotate(f"best weighted\n{fw[best_fw]:.4f} @ ep{ep[best_fw]}",
#                 xy=(ep[best_fw], fw[best_fw]), xytext=(6, -20), textcoords="offset points",
#                 fontsize=7, color=CYAN)
#     ax.annotate(f"best macro\n{fm[best_fm]:.4f} @ ep{ep[best_fm]}",
#                 xy=(ep[best_fm], fm[best_fm]), xytext=(6, 10), textcoords="offset points",
#                 fontsize=7, color=VIOLET)

#     ax.set_xlabel("Epoch")
#     ax.set_ylabel("F1 Score")
#     section_title(ax, "F1 Scores — Weighted & Macro")
#     ax.legend(loc="lower right")
#     fig.tight_layout()
#     save(fig, out, "03_f1_scores.png")


# # ── Plot 4 — Precision & Recall ─────────────────────────────────────────────

# def plot_precision_recall(df, out):
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))

#     ep = df["epoch"].to_numpy()
#     prec = df["precision"].to_numpy()
#     rec  = df["recall"].to_numpy()

#     # Left: vs epoch
#     add_phase_bands(ax1)
#     ax1.plot(ep, prec, color=ORANGE, lw=2, label="Precision")
#     ax1.plot(ep, rec,  color=PINK,   lw=2, label="Recall")
#     ax1.plot(ep, smooth(prec, 9), color=ORANGE, lw=1, ls="--", alpha=0.45)
#     ax1.plot(ep, smooth(rec,  9), color=PINK,   lw=1, ls="--", alpha=0.45)
#     mark_best(ax1, ep, prec, color=ORANGE)
#     ax1.set_xlabel("Epoch")
#     ax1.set_ylabel("Score")
#     section_title(ax1, "Precision & Recall over Epochs")
#     ax1.legend()

#     # Right: scatter precision vs recall (coloured by epoch)
#     sc = ax2.scatter(rec, prec, c=ep, cmap="plasma", s=18, alpha=0.75, zorder=3)
#     cb = fig.colorbar(sc, ax=ax2, pad=0.02)
#     cb.ax.yaxis.set_tick_params(color=MUTED)
#     cb.set_label("Epoch", color=MUTED2, fontsize=8)
#     plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED)
#     ax2.set_xlabel("Recall")
#     ax2.set_ylabel("Precision")
#     section_title(ax2, "Precision vs Recall (coloured by epoch)")

#     fig.tight_layout()
#     save(fig, out, "04_precision_recall.png")


# # ── Plot 5 — Kappa & Specificity ────────────────────────────────────────────

# def plot_kappa_specificity(df, out):
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
#     ep   = df["epoch"].to_numpy()
#     kap  = df["kappa"].to_numpy()
#     spec = df["specificity"].to_numpy()

#     for ax, vals, color, label, ref in [
#         (ax1, kap,  AMBER, "Cohen's κ",  None),
#         (ax2, spec, GREEN, "Specificity", None),
#     ]:
#         add_phase_bands(ax)
#         ax.fill_between(ep, vals, alpha=0.10, color=color)
#         ax.plot(ep, vals, color=color, lw=2, label=label)
#         ax.plot(ep, smooth(vals, 9), color=color, lw=1, ls="--", alpha=0.5)
#         mark_best(ax, ep, vals, color=color)
#         ax.set_xlabel("Epoch")
#         ax.set_ylabel(label)
#         section_title(ax, label + " over Epochs")
#         ax.legend()

#     # Kappa interpretation bands
#     for lo, hi, lbl, alpha in [(0.4, 0.6, "Moderate", 0.06), (0.6, 0.8, "Substantial", 0.06)]:
#         ax1.axhspan(lo, hi, color=AMBER, alpha=alpha)
#         ax1.text(148, (lo + hi) / 2, lbl, ha="right", va="center", fontsize=6.5, color=AMBER, alpha=0.7)

#     fig.tight_layout()
#     save(fig, out, "05_kappa_specificity.png")


# # ── Plot 6 — F1 Gap ─────────────────────────────────────────────────────────

# def plot_f1_gap(df, out):
#     fig, ax = plt.subplots(figsize=(12, 4.5))
#     ep  = df["epoch"].to_numpy()
#     gap = (df["f1_weighted"] - df["f1_macro"]).to_numpy()

#     add_phase_bands(ax)
#     ax.fill_between(ep, gap, alpha=0.15, color=AMBER)
#     ax.plot(ep, gap, color=AMBER, lw=2, label="F1 weighted − F1 macro")
#     ax.plot(ep, smooth(gap, 9), color=AMBER, lw=1, ls="--", alpha=0.5, label="9-ep MA")

#     # Annotate min gap (most balanced)
#     min_idx = int(np.argmin(gap))
#     ax.scatter([ep[min_idx]], [gap[min_idx]], marker="D", s=100, color=AMBER,
#                zorder=10, edgecolors="white", lw=0.4)
#     ax.annotate(f"min gap {gap[min_idx]:.4f}\n(ep {ep[min_idx]})",
#                 xy=(ep[min_idx], gap[min_idx]), xytext=(6, 6), textcoords="offset points",
#                 fontsize=7, color=AMBER)

#     ax.set_xlabel("Epoch")
#     ax.set_ylabel("F1 gap (weighted − macro)")
#     section_title(ax, "F1 Macro–Weighted Gap — Class Imbalance Proxy\n"
#                        "(smaller = model performance more evenly distributed across sleep stages)")
#     ax.legend()
#     fig.tight_layout()
#     save(fig, out, "06_f1_gap.png")


# # ── Plot 7 — Learning Velocity ──────────────────────────────────────────────

# def plot_learning_velocity(df, out):
#     fig, ax = plt.subplots(figsize=(12, 4.5))
#     ep  = df["epoch"].to_numpy()
#     fw  = df["f1_weighted"].to_numpy()
#     delta = np.diff(fw)
#     ep_d  = ep[1:]

#     colors = [GREEN if d >= 0 else ROSE for d in delta]
#     ax.bar(ep_d, delta, color=colors, width=0.85, alpha=0.8, zorder=3)
#     ax.axhline(0, color=BORDER, lw=1.2)
#     ax.plot(ep_d, smooth(delta, 11), color=CYAN, lw=1.5, ls="--", alpha=0.7, label="11-ep MA")

#     # Phase bands
#     add_phase_bands(ax, alpha=0.06)

#     ax.set_xlabel("Epoch")
#     ax.set_ylabel("ΔF1 (weighted)")
#     section_title(ax, "Per-Epoch Learning Velocity  (ΔF1 weighted)\n"
#                        "Green = improvement · Red = regression")
#     ax.legend()

#     gain_patch = mpatches.Patch(color=GREEN, alpha=0.8, label=f"+gain epochs: {(delta > 0).sum()}")
#     drop_patch  = mpatches.Patch(color=ROSE,  alpha=0.8, label=f"−drop epochs: {(delta < 0).sum()}")
#     ax.legend(handles=[gain_patch, drop_patch, Line2D([0],[0], color=CYAN, ls="--", label="Smoothed")],
#               loc="upper right")
#     fig.tight_layout()
#     save(fig, out, "07_learning_velocity.png")


# # ── Plot 8 — Correlation Heatmap ────────────────────────────────────────────

# def plot_correlation(df, out):
#     cols = ["train_loss", "accuracy", "f1_weighted", "f1_macro",
#             "precision", "recall", "kappa", "specificity"]
#     labels = ["Loss", "Accuracy", "F1 W", "F1 M",
#               "Precision", "Recall", "κ", "Specificity"]

#     corr = df[cols].corr()

#     fig, ax = plt.subplots(figsize=(9, 7))
#     mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

#     cmap = sns.diverging_palette(220, 10, as_cmap=True)
#     sns.heatmap(
#         corr, ax=ax, mask=mask,
#         cmap=cmap, vmin=-1, vmax=1, center=0,
#         annot=True, fmt=".2f", annot_kws={"size": 8, "color": TEXT},
#         linewidths=0.5, linecolor=BORDER,
#         xticklabels=labels, yticklabels=labels,
#         cbar_kws={"shrink": 0.8, "label": "Pearson r"},
#         square=True,
#     )
#     ax.tick_params(colors=MUTED2, labelsize=8)
#     cb = ax.collections[0].colorbar
#     cb.ax.yaxis.set_tick_params(color=MUTED)
#     cb.set_label("Pearson r", color=MUTED2, fontsize=8)
#     plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED)

#     section_title(ax, "Metric Correlation Matrix (lower triangle)")
#     fig.tight_layout()
#     save(fig, out, "08_correlation_heatmap.png")


# # ── Plot 9 — Radar ──────────────────────────────────────────────────────────

# def plot_radar(df, out):
#     metrics = ["accuracy", "f1_weighted", "f1_macro", "precision", "recall", "kappa", "specificity"]
#     labels  = ["Accuracy", "F1 W", "F1 M", "Precision", "Recall", "κ", "Specificity"]
#     N = len(metrics)

#     best_idx  = int(df["f1_weighted"].idxmax())
#     final_idx = len(df) - 1

#     def get_vals(idx):
#         return [df.iloc[idx][m] for m in metrics]

#     best_vals  = get_vals(best_idx)
#     final_vals = get_vals(final_idx)

#     angles = [n / N * 2 * np.pi for n in range(N)]
#     angles += angles[:1]

#     fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
#     ax.set_facecolor(SURFACE)
#     fig.patch.set_facecolor(BG)

#     ax.set_theta_offset(np.pi / 2)
#     ax.set_theta_direction(-1)
#     ax.set_xticks(angles[:-1])
#     ax.set_xticklabels(labels, color=MUTED2, size=9)
#     ax.set_ylim(0.0, 1.0)
#     ax.set_yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
#     ax.set_yticklabels(["0.0","0.1","0.2","0.3","0.4","0.5","0.6","0.7","0.8","0.9","1.0"], color=MUTED, size=7)
#     ax.grid(color=BORDER, linewidth=0.7)
#     ax.spines["polar"].set_color(BORDER)

#     for vals, color, label in [
#         (best_vals,  CYAN,  f"Best epoch (ep {df.iloc[best_idx]['epoch']:.0f})"),
#         (final_vals, AMBER, f"Final epoch (ep {df.iloc[final_idx]['epoch']:.0f})"),
#     ]:
#         v = vals + vals[:1]
#         ax.plot(angles, v, color=color, lw=2, label=label)
#         ax.fill(angles, v, color=color, alpha=0.12)

#     ax.set_title("Best vs Final Epoch — Key Metrics", color=TEXT,
#                  fontsize=11, fontweight="bold", pad=18)
#     ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=8)
#     fig.tight_layout()
#     save(fig, out, "09_radar.png")


# # ── Plot 10 — Overview 3×3 Grid ─────────────────────────────────────────────

# def plot_overview_grid(df, out):
#     ep = df["epoch"].to_numpy()
#     panels = [
#         ("train_loss",  "Training Loss",    ROSE,   True),
#         ("accuracy",    "Accuracy",         TEAL,   False),
#         ("f1_weighted", "F1 Weighted",      CYAN,   False),
#         ("f1_macro",    "F1 Macro",         VIOLET, False),
#         ("precision",   "Precision",        ORANGE, False),
#         ("recall",      "Recall",           PINK,   False),
#         ("kappa",       "Cohen's κ",        AMBER,  False),
#         ("specificity", "Specificity",      GREEN,  False),
#     ]

#     fig = plt.figure(figsize=(18, 13))
#     fig.patch.set_facecolor(BG)

#     # Big title
#     fig.suptitle(
#         "Training Metrics — Full Overview\n"
#         "Temporal–Spectral Fusion · Sleep Staging · 150 Epochs",
#         fontsize=14, fontweight="bold", color=TEXT, y=0.98
#     )

#     gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.32)
#     axes_list = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(8)]

#     for ax, (col, label, color, lib) in zip(axes_list, panels):
#         vals = df[col].to_numpy()
#         ax.set_facecolor(SURFACE)
#         for spine in ax.spines.values():
#             spine.set_edgecolor(BORDER)
#         ax.tick_params(colors=MUTED, labelsize=7)
#         ax.grid(color=BORDER, linewidth=0.4, alpha=0.7)
#         ax.set_xlabel("Epoch", fontsize=7.5, color=MUTED2)
#         ax.set_ylabel(label, fontsize=7.5, color=MUTED2)
#         ax.set_title(label, fontsize=9, color=TEXT, fontweight="bold")

#         ax.fill_between(ep, vals, alpha=0.10, color=color)
#         ax.plot(ep, vals, color=color, lw=1.6)
#         ax.plot(ep, smooth(vals, 9), color=color, lw=0.9, ls="--", alpha=0.5)

#         best_i = int(np.argmin(vals) if lib else np.argmax(vals))
#         ax.scatter([ep[best_i]], [vals[best_i]], marker="*", s=140, color=color,
#                    zorder=10, edgecolors="white", lw=0.3)

#         # Best value annotation (top-right corner text)
#         best_label = f"best {vals[best_i]:.4f}"
#         ax.text(0.97, 0.05 if lib else 0.95, best_label,
#                 transform=ax.transAxes, ha="right", va="bottom" if lib else "top",
#                 fontsize=6.5, color=color)

#     # 9th panel: delta-F1 bar chart (mini)
#     ax9 = fig.add_subplot(gs[2, 2])
#     fw    = df["f1_weighted"].to_numpy()
#     delta = np.diff(fw)
#     ep_d  = ep[1:]
#     bar_colors = [GREEN if d >= 0 else ROSE for d in delta]
#     ax9.bar(ep_d, delta, color=bar_colors, width=0.85, alpha=0.8)
#     ax9.axhline(0, color=BORDER, lw=1)
#     ax9.set_facecolor(SURFACE)
#     for spine in ax9.spines.values():
#         spine.set_edgecolor(BORDER)
#     ax9.tick_params(colors=MUTED, labelsize=7)
#     ax9.grid(color=BORDER, linewidth=0.4, alpha=0.7)
#     ax9.set_xlabel("Epoch", fontsize=7.5, color=MUTED2)
#     ax9.set_ylabel("ΔF1", fontsize=7.5, color=MUTED2)
#     ax9.set_title("ΔF1 Learning Velocity", fontsize=9, color=TEXT, fontweight="bold")

#     fig.tight_layout(rect=[0, 0, 1, 0.96])
#     save(fig, out, "10_overview_grid.png")


# # ── Main ────────────────────────────────────────────────────────────────────

# def main():
#     parser = argparse.ArgumentParser(
#         description="Visualize sleep staging training metrics from a JSONL file."
#     )
#     parser.add_argument("--jsonl_path", type=Path, default=r"C:\PS\Sleep-Staging\results\temporal spectral fusion\Cross Attn\training_metrics_fusion.jsonl",
#                         help="Path to training_metrics_fusion.jsonl")
#     parser.add_argument("--out", type=Path, default=None,
#                         help="Output directory (default: same folder as jsonl_path)")
#     args = parser.parse_args()

#     if not args.jsonl_path.exists():
#         print(f"ERROR: file not found: {args.jsonl_path}", file=sys.stderr)
#         sys.exit(1)

#     out_dir = args.out if args.out else args.jsonl_path.parent / "training_plots"
#     out_dir.mkdir(parents=True, exist_ok=True)

#     print(f"\nLoading  : {args.jsonl_path}")
#     df = load_data(args.jsonl_path)
#     print(f"Epochs   : {len(df)}")
#     print(f"Output   : {out_dir}\n")

#     apply_base_style()

#     best_epoch = int(df.loc[df["f1_weighted"].idxmax(), "epoch"])
#     best_f1    = df["f1_weighted"].max()
#     best_acc   = df["accuracy"].max()
#     best_kappa = df["kappa"].max()
#     print(f"─── Best checkpoint ────────────────────")
#     print(f"  Epoch      : {best_epoch}")
#     print(f"  F1 weighted: {best_f1:.4f}")
#     print(f"  Accuracy   : {best_acc:.4f}")
#     print(f"  Kappa      : {best_kappa:.4f}")
#     print(f"────────────────────────────────────────\n")
#     print("Generating plots:")

#     plot_loss(df, out_dir)
#     plot_accuracy(df, out_dir)
#     plot_f1(df, out_dir)
#     plot_precision_recall(df, out_dir)
#     plot_kappa_specificity(df, out_dir)
#     plot_f1_gap(df, out_dir)
#     plot_learning_velocity(df, out_dir)
#     plot_correlation(df, out_dir)
#     plot_radar(df, out_dir)
#     plot_overview_grid(df, out_dir)

#     print(f"\nDone. {len(list(out_dir.glob('*.png')))} plots saved to:\n  {out_dir.resolve()}\n")


# if __name__ == "__main__":
#     main()