"""
eval/v5/figures_v5.py — v5 figures (PDF + PNG preview).

fig1_heatmap_v5 : TSR (headline: strict + state) and SSR_tf by tier x mode
fig2_pareto_v5  : TSR vs FLOPs per step relative to always-M0 (log x);
                  fixed modes, oracle, cascade, learned-router frontier, M3 variants
Reads metrics_v5.json, routers_v5.json, cost_v5.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

V5 = Path(__file__).resolve().parent
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})


def fig1(metrics):
    tiers = ["easy", "medium", "hard"]
    cols = ["M0", "M1", "M2", "M3", "oracle"]
    bd = metrics["step3_teacher_forced"]["by_difficulty"]
    n = {t: bd[t]["M0"]["TSR"]["n"] for t in tiers}
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.1), constrained_layout=True)
    for ax, metric, title in ((axes[0], "TSR", "TSR (strict + state)"),
                              (axes[1], "SSR_tf", "SSR (teacher-forced)")):
        M = [[bd[t][c][metric]["rate"] for c in cols] for t in tiers]
        ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for i in range(len(tiers)):
            for j in range(len(cols)):
                v = M[i][j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > 0.55 else INK)
        ax.set_xticks(range(len(cols)), ["M0", "M1", "M2", "M3", "Oracle"])
        ax.set_yticks(range(len(tiers)), [f"{t} (n={n[t]})" for t in tiers])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    axes[1].set_yticklabels([])
    for ext in ("pdf", "png"):
        fig.savefig(V5 / f"fig1_heatmap_v5.{ext}", dpi=200)
    plt.close(fig)


def fig2(metrics, routers, cost):
    P = routers["policies"]
    m0_flops = cost["modes"]["M0"]["flops_mean"]
    fig, ax = plt.subplots(figsize=(4.6, 3.0), constrained_layout=True)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)

    # learned pre-generation router (decision costs from FLOPs), sorted frontier
    pts = sorted(((p["cost_rel_M0_flops"], p["TSR"]) for p in P if p["group"] == "learned-flops"))
    front, best = [], -1
    for x, y in pts:
        if y > best:
            front.append((x, y)); best = y
    ax.plot([x for x, _ in front], [y for _, y in front], color=ORANGE, lw=2, zorder=2)
    ax.scatter([x for x, _ in pts], [y for _, y in pts], s=10, color=ORANGE, alpha=0.35,
               edgecolor="none", zorder=2)
    top = front[-1]
    ax.annotate("learned router (CV)", top, xytext=(0.55, top[1] + 0.006), textcoords="data",
                ha="left", color=INK2, fontsize=7,
                arrowprops={"arrowstyle": "-", "color": ORANGE, "lw": 0.6})

    for p in P:
        x, y = p["cost_rel_M0_flops"], p["TSR"]
        if p["group"] == "fixed":
            ax.scatter(x, y, s=40, color=INK2, edgecolor="white", linewidth=1.5, zorder=3)
            lab = p["policy"].replace("Fixed ", "")
            ax.annotate(lab, (x, y), xytext=(6, -10) if lab == "M0" else (5, 2),
                        textcoords="offset points", color=INK, fontsize=7.5)
        elif p["policy"].startswith("Oracle (cheapest-correct by flops)"):
            ax.scatter(x, y, s=60, marker="*", color=BLUE, edgecolor="white", linewidth=1, zorder=4)
            ax.annotate("oracle", (x, y), xytext=(5, 2), textcoords="offset points",
                        color=INK, fontsize=7.5)
        elif p["group"] == "cascade" and "M1" in p["policy"]:
            ax.scatter(x, y, s=40, marker="^", color=ORANGE, edgecolor="white", linewidth=1, zorder=4)
            ax.annotate("cascade M0→M1", (x, y), xytext=(6, 3), textcoords="offset points",
                        color=INK2, fontsize=7)

    names = {"m3_gptoss_3shot_v5": "gpt-oss 3-shot", "m3_gptoss_3shot_grounded_v5": "gpt-oss 3-shot (clean demos)",
             "m3_qwen27b_0shot_v5": "Qwen3.8-27B 0-shot", "m3_qwen27b_3shot_v5": "Qwen3.8-27B 3-shot"}
    for k, v in metrics.get("step8_m3_variants", {}).items():
        if not v.get("flops_mean"):
            continue
        x, y = v["flops_mean"] / m0_flops, v["TSR"]["rate"]
        ax.scatter(x, y, s=36, marker="D", color=AQUA, edgecolor="white", linewidth=1, zorder=4)
        ax.annotate(names.get(k, k), (x, y), xytext=(5, 2), textcoords="offset points",
                    color=INK2, fontsize=6.5)

    ax.set_xscale("log")
    ticks = [0.5, 1, 2, 5, 10, 20, 50]
    ax.set_xticks(ticks, [f"{t:g}×" for t in ticks])
    xmax = max(c.get_offsets()[:, 0].max() for c in ax.collections if len(c.get_offsets()))
    ax.set_xlim(0.5, xmax * 2.2)
    ax.set_xlabel("FLOPs per step relative to always-M0 (2·N_active·tokens)")
    ax.set_ylabel("TSR (strict + state)")
    for ext in ("pdf", "png"):
        fig.savefig(V5 / f"fig2_pareto_v5.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    metrics = json.loads((V5 / "metrics_v5.json").read_text())
    routers = json.loads((V5 / "routers_v5.json").read_text())
    cost = json.loads((V5 / "cost_v5.json").read_text())
    fig1(metrics)
    fig2(metrics, routers, cost)
    print("figures written")
