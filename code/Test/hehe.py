import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from collections import defaultdict

# ── load data ──────────────────────────────────────────────────────────────────
with open('C:/Users/pujas/Downloads/search_log (1) (1).json') as f:
    data = json.load(f)

iterations   = [d["iteration"]       for d in data]
performance  = [d["performance"]     for d in data]
reward       = [d["reward"]          for d in data]
params_m     = [d["params_million"]  for d in data]
flops_g      = [d["flops_giga"]      for d in data]
seq_types    = [d["arch"]["sequence"]["type"]        for d in data]
fusion_types = [d["arch"]["fusion"]["fusion_type"]   for d in data]
channels     = [d["arch"]["feature"]["channels"]     for d in data]
layer_ops    = [d["arch"]["feature"]["layer_ops"]    for d in data]

# running best reward
running_best = []
best = -np.inf
for r in reward:
    if r > best:
        best = r
    running_best.append(best)

best_idx  = int(np.argmax(reward))
best_data = data[best_idx]

# ── palette ────────────────────────────────────────────────────────────────────
SEQ_COLORS   = {"LSTM": "#4C9BE8", "BiLSTM": "#6A5ACD", "GRU": "#F4A261", "BiGRU": "#E76F51"}
FUSION_MARKS = {"gated": "o", "concat": "^"}
CHAN_COLORS  = {16: "#a8dadc", 32: "#457b9d", 48: "#1d3557"}

BG   = "#0f1117"
GRID = "#2a2d3a"
FG   = "#e0e0e0"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": FG,
    "xtick.color": FG, "ytick.color": FG,
    "text.color": FG, "grid.color": GRID,
    "grid.linewidth": 0.6, "font.family": "DejaVu Sans",
    "legend.framealpha": 0.15, "legend.edgecolor": GRID,
})

fig = plt.figure(figsize=(18, 13))
fig.patch.set_facecolor(BG)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

# ── 1. Reward over iterations + running best ───────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.scatter(iterations, reward, c=[SEQ_COLORS[s] for s in seq_types],
            alpha=0.65, s=28, zorder=3, label="_nolegend_")
ax1.plot(iterations, running_best, color="#FFD166", lw=2.2, label="Running best", zorder=4)
ax1.axvline(best_idx, color="#FFD166", lw=1.2, ls="--", alpha=0.5)
ax1.scatter([best_idx], [reward[best_idx]], color="#FFD166", s=120,
            zorder=5, edgecolors="white", linewidths=1.2)
ax1.set_title("Search Reward over Iterations", fontsize=13, fontweight="bold", pad=10)
ax1.set_xlabel("Iteration"); ax1.set_ylabel("Reward")
ax1.grid(True, axis="y")
# seq legend
patches = [mpatches.Patch(color=c, label=k) for k, c in SEQ_COLORS.items()]
ax1.legend(handles=patches + [plt.Line2D([0],[0], color="#FFD166", lw=2, label="Running best")],
           loc="upper left", fontsize=8, ncol=5)

# ── 2. Performance vs FLOPs (Pareto scatter) ────────────────────────────────
ax2 = fig.add_subplot(gs[1, :2])
for d, ft, st, ch, p, fl in zip(data, fusion_types, seq_types, channels, performance, flops_g):
    ax2.scatter(fl, p, c=SEQ_COLORS[st], marker=FUSION_MARKS[ft],
                s=55, alpha=0.75, edgecolors="none")

# Pareto front
sorted_pts = sorted(zip(flops_g, performance), key=lambda x: x[0])
pareto = []
max_p = -np.inf
for fl, p in sorted_pts:
    if p > max_p:
        pareto.append((fl, p)); max_p = p
if pareto:
    px, py = zip(*sorted(pareto))
    ax2.step(px, py, where="post", color="#FFD166", lw=1.8, label="Pareto front", zorder=4)
ax2.scatter([flops_g[best_idx]], [performance[best_idx]], color="#FFD166",
            s=140, zorder=5, edgecolors="white", lw=1.2, label="Best arch")
ax2.set_title("Performance vs. FLOPs", fontsize=12, fontweight="bold")
ax2.set_xlabel("FLOPs (GFLOPs)"); ax2.set_ylabel("Performance")
ax2.grid(True)
# markers for fusion
m_patches = [plt.scatter([],[], marker=m, c="#aaa", s=55, label=k)
              for k, m in FUSION_MARKS.items()]
ax2.legend(handles=m_patches + [plt.Line2D([0],[0], color="#FFD166", lw=2, label="Pareto front"),
           plt.scatter([],[], marker="o", c="#FFD166", s=80, edgecolors="white", label="Best arch")],
           fontsize=8)

# ── 3. Performance by sequence type (violin) ────────────────────────────────
ax3 = fig.add_subplot(gs[1, 2])
seq_order = ["LSTM", "BiLSTM", "GRU", "BiGRU"]
perf_by_seq = [[ performance[i] for i,s in enumerate(seq_types) if s == st] for st in seq_order]
parts = ax3.violinplot(perf_by_seq, positions=range(len(seq_order)), showmedians=True, widths=0.7)
for i, (pc, st) in enumerate(zip(parts["bodies"], seq_order)):
    pc.set_facecolor(SEQ_COLORS[st]); pc.set_alpha(0.75)
parts["cmedians"].set_color("#FFD166"); parts["cmedians"].set_linewidth(2)
for key in ("cbars","cmins","cmaxes"):
    parts[key].set_color(FG); parts[key].set_linewidth(0.8)
ax3.set_xticks(range(len(seq_order))); ax3.set_xticklabels(seq_order, fontsize=8)
ax3.set_title("Performance by\nSequence Type", fontsize=12, fontweight="bold")
ax3.set_ylabel("Performance"); ax3.grid(True, axis="y")

# ── 4. Performance by channel width ─────────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
chan_order = [16, 32, 48]
perf_by_ch = [[performance[i] for i,c in enumerate(channels) if c == ch] for ch in chan_order]
bp = ax4.boxplot(perf_by_ch, patch_artist=True, medianprops=dict(color="#FFD166", lw=2),
                 whiskerprops=dict(color=FG), capprops=dict(color=FG), flierprops=dict(markerfacecolor=FG, markersize=4))
for patch, ch in zip(bp["boxes"], chan_order):
    patch.set_facecolor(CHAN_COLORS[ch]); patch.set_alpha(0.8)
ax4.set_xticks([1,2,3]); ax4.set_xticklabels([f"ch={c}" for c in chan_order])
ax4.set_title("Performance by\nChannel Width", fontsize=12, fontweight="bold")
ax4.set_ylabel("Performance"); ax4.grid(True, axis="y")

# ── 5. Layer ops heatmap (op combo vs performance) ──────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
combo_perf = defaultdict(list)
for i, ops in enumerate(layer_ops):
    key = "-".join(ops)
    combo_perf[key].append(performance[i])
combos = sorted(combo_perf.keys())
means  = [np.mean(combo_perf[k]) for k in combos]
colors = [plt.cm.plasma((m - min(means)) / (max(means) - min(means))) for m in means]
bars = ax5.barh(range(len(combos)), means, color=colors, edgecolor=GRID, height=0.6)
ax5.set_yticks(range(len(combos))); ax5.set_yticklabels(combos, fontsize=8)
ax5.set_title("Avg Performance by\nLayer Op Combo", fontsize=12, fontweight="bold")
ax5.set_xlabel("Mean Performance"); ax5.grid(True, axis="x")
for bar, m in zip(bars, means):
    ax5.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
             f"{m:.1f}", va="center", fontsize=7.5, color=FG)

# ── 6. Best architecture summary ─────────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis("off")
best_a = best_data["arch"]
summary = [
    ("Best Iteration",  str(best_idx)),
    ("Reward",          f"{best_data['reward']:.4f}"),
    ("Performance",     f"{best_data['performance']:.4f}"),
    ("FLOPs",           f"{best_data['flops_giga']:.3f} G"),
    ("Params",          f"{best_data['params_million']:.3f} M"),
    ("", ""),
    ("── Architecture ──", ""),
    ("Channels",        str(best_a['feature']['channels'])),
    ("Layer Ops",       " → ".join(best_a['feature']['layer_ops'])),
    ("Sequence",        best_a['sequence']['type']),
    ("Hidden Dim",      str(best_a['sequence']['hidden_dim'])),
    ("Fusion",          best_a['fusion']['fusion_type']),
]
ax6.set_title("Best Architecture", fontsize=12, fontweight="bold")
y = 0.97
for label, val in summary:
    if label.startswith("──"):
        ax6.text(0.05, y, label, fontsize=8.5, color="#FFD166",
                 fontweight="bold", transform=ax6.transAxes)
    elif label:
        ax6.text(0.05, y, label + ":", fontsize=8.5, color="#aaa", transform=ax6.transAxes)
        ax6.text(0.58, y, val,         fontsize=8.5, color=FG,    transform=ax6.transAxes)
    y -= 0.083

fig.suptitle("Hierarchical NAS Search Results", fontsize=16, fontweight="bold",
             y=0.98, color=FG)

plt.savefig("nas_results_1.png", dpi=150, bbox_inches="tight",
            facecolor=BG)
print("Saved.")