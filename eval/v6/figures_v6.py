"""
eval/v6/figures_v6.py — Step 7 figures (PDF + PNG preview), from metrics_v6.json / routers_v6.json / cost_v6.json.

fig1_heatmap_v6_<cond> : TSR (strict + state) | SSR teacher-forced | SSR rollout, tier x mode,
                         one shared colour scale (0-1). Cells without data (M3 not run yet, M3 has
                         no rollout) are hatched and marked "pending" / "n/a".
fig2_pareto_v6_<cond>  : TSR vs FLOPs per step relative to always-M0 (log x). Every point labelled,
                         legend, y-axis from 0 (no exaggerated gaps). Oracle = cheapest strictly
                         correct mode per step by FLOPs over the SAME modes as the tables
                         (M0-M3, M3 = gpt-oss-120b zero-shot run 1). M3 variants (Step 6) as diamonds.

Usage: venv/bin/python eval/v6/figures_v6.py --fig1 v1.2 --fig2 v1.1_v4adapter
(rerun with --fig2 v1.2 once the v1.2 M3 run is in routers_v6.json)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

V6 = Path(__file__).resolve().parent
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
COND_NAME = {"v1": "ViTool v1", "v1.1": "ViTool v1.1 (retrained adapter)",
             "v1.1_v4adapter": "ViTool v1.1", "v1.2": "ViTool v1.2"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})


def fig1(M, cond):
    o = M["conditions"][cond]
    tf, ro = o["teacher_forced"], o.get("rollout", {})
    tiers = ["easy", "medium", "hard"]
    cols = ["M0", "M1", "M2", "M3", "oracle"]
    n = {t: tf["by_difficulty"][t]["M0"]["TSR"]["n"] for t in tiers}
    oracle_modes = "M0–M3" if "M3" in tf["overall"] else "M0–M2"

    def val(metric, t, c):
        if metric == "SSR_rollout":
            r = ro.get(c)
            if r is None:
                return None
            return r.get("by_difficulty_SSR_rollout", {}).get(t, {}).get("rate")
        b = tf["by_difficulty"][t].get(c)
        return None if b is None else b[metric]["rate"]

    panels = [("TSR", "TSR (strict + state), TF"), ("SSR_tf", "SSR, teacher-forced (TF)"),
              ("SSR_rollout", "SSR, rollout")]
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.35), constrained_layout=True)
    for ax, (metric, title) in zip(axes, panels):
        ax.set_xlim(-0.5, len(cols) - 0.5)
        ax.set_ylim(len(tiers) - 0.5, -0.5)
        for i, t in enumerate(tiers):
            for j, c in enumerate(cols):
                v = val(metric, t, c)
                if v is None:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="white", edgecolor=GRID,
                                               hatch="////", linewidth=0.5))
                    pend = (c == "M3" and "M3" not in tf["overall"]) or (c == "oracle" and metric == "SSR_rollout") \
                        or (c == "M3" and metric == "SSR_rollout")
                    lab = "n/a" if metric == "SSR_rollout" and c in ("M3", "oracle") else ("pending" if pend else "–")
                    ax.text(j, i, lab, ha="center", va="center", fontsize=6.5, color=MUTED)
                    continue
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=cmap(v), edgecolor="white", lw=1.5))
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > 0.55 else INK)
        ax.set_xticks(range(len(cols)), ["M0", "M1", "M2", "M3", f"Oracle\n({oracle_modes})"])
        ax.set_yticks(range(len(tiers)), [f"{t} (n={n[t]})" for t in tiers])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=8.5, color=INK, loc="left")
    for ax in axes[1:]:
        ax.set_yticklabels([])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    cb = fig.colorbar(sm, ax=axes, fraction=0.025, pad=0.01)
    cb.set_label("rate (shared scale 0–1)", fontsize=7)
    cb.outline.set_visible(False)
    fig.suptitle(f"{COND_NAME[cond]} — 112 eval trajectories (TSR n per tier) / 284 steps", fontsize=8,
                 color=INK2, x=0.01, ha="left")
    tag = cond.replace(".", "")
    for ext in ("pdf", "png"):
        fig.savefig(V6 / f"fig1_heatmap_v6_{tag}.{ext}", dpi=200)
    plt.close(fig)


def fig2(M, R, C, cond):
    P = R[cond]["policies"]
    m0_flops = C[cond]["modes"]["M0"]["flops_mean"]
    fig, ax = plt.subplots(figsize=(6.0, 4.2), constrained_layout=True)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    pts = []  # (x, y, label, dx, dy, ha)

    lp = sorted(((p["cost_rel_M0_flops"], p["TSR"]) for p in P if p["group"] == "learned-flops"))
    front, best = [], -1
    for x, y in lp:
        if y > best:
            front.append((x, y)); best = y
    ax.scatter([x for x, _ in lp], [y for _, y in lp], s=10, color=ORANGE, alpha=0.3, edgecolor="none", zorder=2)
    ax.plot([x for x, _ in front], [y for _, y in front], color=ORANGE, lw=1.8, zorder=2)
    top = max(front, key=lambda p: p[1])
    pts.append((top[0], top[1], f"learned router, best ({top[1]:.3f})", 0, -14, "center"))

    for p in P:
        x, y = p["cost_rel_M0_flops"], p["TSR"]
        g, name = p["group"], p["policy"]
        if g == "fixed":
            ax.scatter(x, y, s=42, color=INK2, edgecolor="white", linewidth=1.4, zorder=4)
            pts.append((x, y, name.replace("Fixed ", "") + f" ({y:.3f})", 6, -3, "left"))
        elif name.startswith("Oracle (cheapest-correct by flops)"):
            ax.scatter(x, y, s=90, marker="*", color=BLUE, edgecolor="white", linewidth=0.8, zorder=5)
            pts.append((x, y, f"oracle by FLOPs ({y:.3f})", 7, 0, "left"))
        elif g == "cascade":
            esc = "M1" if "M1" in name else "M3"
            ax.scatter(x, y, s=38, marker="^", color=VIOLET, edgecolor="white", linewidth=1, zorder=4)
            pts.append((x, y, f"cascade M0→{esc} ({y:.3f})", 6, 6 if esc == "M3" else -10, "left"))
        elif name.startswith("Random"):
            ax.scatter(x, y, s=30, marker="s", color=MUTED, edgecolor="white", linewidth=1, zorder=3)
            pts.append((x, y, f"random ({y:.3f})", 6, -3, "left"))
        elif name.startswith("Length"):
            ax.scatter(x, y, s=30, marker="s", color=MUTED, edgecolor="white", linewidth=1, zorder=3)
            pts.append((x, y, f"length router ({y:.3f})", -6, -10, "right"))
        elif name.startswith("Difficulty"):
            ax.scatter(x, y, s=30, marker="P", color=MUTED, edgecolor="white", linewidth=1, zorder=3)
            pts.append((x, y, f"gold-tier escalation ({y:.3f})", 6, -3, "left"))

    var_name = {"m3_gptoss_3shot_v6": "gpt-oss 3-shot", "m3_qwen27b_0shot_v6": "Qwen3.8-27B 0-shot",
                "m3_qwen27b_3shot_v6": "Qwen3.8-27B 3-shot"}
    variants = M.get("step6_m3_variants_v11", {}) if cond.startswith("v1.1") else \
        {k.replace("_v12", "_v6"): v for k, v in M.get("m3_on_v12", {}).items() if "qwen" in k}
    if True:
        for k, v in variants.items():
            if k not in var_name:
                continue
            if "TSR" not in v:
                continue
            x, y = v["flops_mean"] / m0_flops, v["TSR"]["rate"]
            ax.scatter(x, y, s=34, marker="D", color=AQUA, edgecolor="white", linewidth=1, zorder=4)
            pts.append((x, y, f"{var_name[k]} ({y:.3f})", 6, -3, "left"))

    # every point gets a number next to it; the key (number -> name, TSR) sits in the empty lower area
    pts.sort(key=lambda p: (p[0], -p[1]))
    key_lines = []
    fig.canvas.draw()
    px = fig.dpi / 72
    groups = []  # points closer than 5 px share one label "4,6"
    for k, (x, y, lab, *_rest) in enumerate(pts, 1):
        key_lines.append(f"{k:>2}  {lab}")
        X, Y = ax.transData.transform((x, y))
        for g in groups:
            if abs(g["X"] - X) < 5 * px and abs(g["Y"] - Y) < 5 * px:
                g["ks"].append(k)
                break
        else:
            groups.append({"X": X, "Y": Y, "x": x, "y": y, "ks": [k]})
    taken = []
    offs = [(5, 5), (5, -11), (-12, 5), (-12, -11), (5, 14), (-14, 14)]
    for g in groups:
        for dx, dy in offs:
            T = (g["X"] + dx * px, g["Y"] + dy * px)
            if all(abs(T[0] - a_) > 14 * px or abs(T[1] - b_) > 9 * px for a_, b_ in taken):
                break
        taken.append(T)
        ax.annotate(",".join(map(str, g["ks"])), (g["x"], g["y"]), xytext=(dx, dy), textcoords="offset points",
                    fontsize=6.5, color=INK, fontweight="bold", va="bottom" if dy > 0 else "top",
                    ha="left" if dx > 0 else "right")
    half = (len(key_lines) + 1) // 2
    for c, chunk in enumerate((key_lines[:half], key_lines[half:])):
        ax.text(0.02 + 0.5 * c, 0.03, "\n".join(chunk), transform=ax.transAxes, fontsize=6.3, color=INK,
                va="bottom", ha="left", family="DejaVu Sans Mono")
    ax.set_xscale("log")
    ticks = [0.5, 1, 2, 5, 10, 20]
    ax.set_xticks(ticks, [f"{t:g}×" for t in ticks])
    ax.set_xlim(0.45, 30)
    ax.set_ylim(0, 1)
    ax.set_xlabel("FLOPs per step relative to always-M0 (2·N_active·tokens; log scale)")
    ax.set_ylabel("TSR (strict + state), teacher-forced")
    leg = [Line2D([], [], marker="o", ls="", color=INK2, label="fixed mode"),
           Line2D([], [], marker="*", ls="", color=BLUE, markersize=10, label="oracle (M0–M3, M3 = run 1)"),
           Line2D([], [], marker="^", ls="", color=VIOLET, label="error cascade"),
           Line2D([], [], color=ORANGE, lw=1.8, label="learned router (5-fold CV frontier)"),
           Line2D([], [], marker="s", ls="", color=MUTED, label="heuristic router"),
           Line2D([], [], marker="D", ls="", color=AQUA, label="M3 variant (Step 6)")]
    ax.legend(handles=leg, loc="upper left", fontsize=6.5, frameon=False, ncol=2)
    ax.set_title(f"{COND_NAME[cond]} — cost/quality of modes and routers (y-axis 0–1)", fontsize=8, loc="left",
                 color=INK)
    tag = cond.replace(".", "")
    for ext in ("pdf", "png"):
        fig.savefig(V6 / f"fig2_pareto_v6_{tag}.{ext}", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig1", default="v1.2")
    ap.add_argument("--fig2", default="v1.2")
    a = ap.parse_args()
    M = json.loads((V6 / "metrics_v6.json").read_text())
    R = json.loads((V6 / "routers_v6.json").read_text())
    C = json.loads((V6 / "cost_v6.json").read_text())
    fig1(M, a.fig1)
    fig2(M, R, C, a.fig2)
    print("figures written")


def fig2_col(R, cond="v1.2", out="fig2_pareto_v6_v12_col.pdf"):
    """IEEE single column: 3.5 x 2.6 in, every text >= 7 pt at print size, short labels next to
    points (adjustText), selected policies only, y 0.4-0.85."""
    from adjustText import adjust_text
    P = R[cond]["policies"]
    FS = 7
    with plt.rc_context({"font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
                         "legend.fontsize": FS}):
        fig, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)
        ax.grid(True, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        lp = sorted(((p["cost_rel_M0_flops"], p["TSR"]) for p in P if p["group"] == "learned-flops"))
        front, best = [], -1
        for x, y in lp:
            if y > best:
                front.append((x, y)); best = y
        ax.plot([x for x, _ in front], [y for _, y in front], color=ORANGE, lw=1.4, zorder=2)
        top = max(front, key=lambda p: p[1])
        texts, X, Y = [], [], []

        fixed_txt = []

        def pt(x, y, lab, pos=None, **kw):
            ax.scatter(x, y, zorder=4, edgecolor="white", linewidth=0.8, **kw)
            X.append(x); Y.append(y)
            if pos:  # hand-placed label for crowded spots, with a leader line
                ax.annotate(lab, (x, y), xytext=pos, textcoords="data", fontsize=FS, color=INK, ha="right",
                            va="center", arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.4})
                fixed_txt.append(lab)
            else:
                texts.append(ax.text(x, y, lab, fontsize=FS, color=INK))

        pt(*top, "learned (best)", s=22, color=ORANGE)
        for p in P:
            x, y, name = p["cost_rel_M0_flops"], p["TSR"], p["policy"]
            if p["group"] == "fixed":
                pt(x, y, name.replace("Fixed ", ""), s=26, color=INK2)
            elif name.startswith("Oracle (cheapest-correct by flops)"):
                pt(x, y, "oracle", s=60, marker="*", color=BLUE)
            elif name.startswith("Cascade M0→M1"):
                pt(x, y, "cascade", pos=(x / 1.25, y + 0.035), s=24, marker="^", color=VIOLET)
            elif name.startswith("Length"):
                pt(x, y, "length", s=20, marker="s", color=MUTED)
            elif name.startswith("Random"):
                pt(x, y, "random", s=20, marker="s", color=MUTED)
        ax.set_xscale("log")
        ticks = [0.5, 1, 2, 5]
        ax.set_xticks(ticks, [f"{t:g}×" for t in ticks])
        ax.minorticks_off()
        ax.set_xlim(0.5, 7)
        ax.set_ylim(0.4, 0.85)
        ax.set_xlabel("FLOPs per step (× always-M0, log)")
        ax.set_ylabel("TSR")
        adjust_text(texts, x=X, y=Y, ax=ax, expand=(1.3, 1.6),
                    arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.4})
        leg = [Line2D([], [], marker="o", ls="", color=INK2, markersize=4, label="fixed mode"),
               Line2D([], [], marker="*", ls="", color=BLUE, markersize=7, label="oracle"),
               Line2D([], [], color=ORANGE, lw=1.4, label="learned router"),
               Line2D([], [], marker="^", ls="", color=VIOLET, markersize=4, label="cascade"),
               Line2D([], [], marker="s", ls="", color=MUTED, markersize=4, label="heuristic")]
        ax.legend(handles=leg, loc="lower right", frameon=False, handletextpad=0.3, borderaxespad=0.2,
                  labelspacing=0.25)
        fig.savefig(V6 / out)
        fig.savefig(V6 / out.replace(".pdf", ".png"), dpi=300)
        plt.close(fig)
