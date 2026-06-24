"""
eval/figures.py — Sinh 2 hình chủ lực của paper từ oracle_report.json (v10.0).

  Fig 1: heatmap per-mode success × difficulty  → minh hoạ overthink/underthink (F1)
  Fig 2: Pareto cost-quality (fixed M0–M3 + oracle + random) → oracle-gap (F2)

INPUT: eval/oracle_report.json (output của eval/oracle.py).
OUTPUT: PDF + PNG (paper dùng PDF vector cho IEEE).

Dùng:
  python eval/figures.py --in eval/oracle_report.json --outdir eval/figs
  python eval/figures.py --selftest   # tạo report giả → vẽ thử, kiểm file ra
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODES = ["M0", "M1", "M2", "M3"]
DIFFS = ["easy", "medium", "hard"]
MODE_LABEL = {"M0": "M0\nfast", "M1": "M1\nthink", "M2": "M2\nsearch", "M3": "M3\ndefer"}


def _setup_mpl():
    import matplotlib
    matplotlib.use("Agg")  # không cần display
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 9, "font.family": "serif",
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 150,
    })
    return plt


# ============================================================
# Fig 1: heatmap mode × difficulty
# ============================================================
def plot_heatmap(report: dict, outdir: Path) -> Path:
    import numpy as np
    plt = _setup_mpl()
    hm = report.get("mode_success_heatmap", {})
    diffs = [d for d in DIFFS if d in hm]
    if not diffs:
        raise SystemExit("Không có mode_success_heatmap trong report.")
    mat = np.array([[hm[d].get(m, 0.0) for m in MODES] for d in diffs])

    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(MODES)))
    ax.set_xticklabels([MODE_LABEL[m] for m in MODES])
    ax.set_yticks(range(len(diffs)))
    ax.set_yticklabels(diffs)
    ax.set_xlabel("Compute mode")
    ax.set_ylabel("Difficulty")
    # annotate giá trị
    for i in range(len(diffs)):
        for j in range(len(MODES)):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="black" if 0.3 < v < 0.8 else "white", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Success rate", fontsize=8)
    ax.set_title("Per-mode success rate by difficulty", fontsize=9)
    fig.tight_layout()
    pdf = outdir / "fig1_heatmap.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(outdir / "fig1_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    return pdf


# ============================================================
# Fig 2: Pareto cost-quality
# ============================================================
def plot_pareto(report: dict, outdir: Path, baselines: dict | None = None) -> Path:
    plt = _setup_mpl()
    pts = report.get("pareto_points", {})
    if not pts:
        raise SystemExit("Không có pareto_points trong report.")

    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    style = {
        "fixed_M0": ("o", "#888888", "Fixed M0"),
        "fixed_M1": ("o", "#5B8FF9", "Fixed M1 (always-think)"),
        "fixed_M2": ("o", "#9B59B6", "Fixed M2"),
        "fixed_M3": ("o", "#E67E22", "Fixed M3 (defer)"),
        "random":   ("X", "#777777", "Random"),
        "oracle":   ("*", "#2ECC71", "Oracle (upper bound)"),
    }
    for key, (marker, color, label) in style.items():
        if key in pts:
            x, y = pts[key]["cost_per_step"], pts[key]["tsr"]
            size = 260 if key == "oracle" else 75
            ax.scatter(x, y, marker=marker, c=color, s=size, label=label, zorder=3,
                       edgecolors="black" if key == "oracle" else "none", linewidths=0.6)

    # overlay router thật (length, difficulty-escalation) nếu có baselines.json
    if baselines:
        for r in baselines.get("routers", []):
            nm = r["name"]
            if nm.startswith("Length"):
                ax.scatter(r["cost_per_step"], r["tsr"], marker="D", c="#16A085",
                           s=70, zorder=4, label="Length-router")
            elif nm.startswith("Difficulty"):
                ax.scatter(r["cost_per_step"], r["tsr"], marker="P", c="#C0392B",
                           s=80, zorder=4, label="Difficulty-escalation")

    # mũi tên oracle-gap: M1 -> oracle
    if "fixed_M1" in pts and "oracle" in pts:
        m1, orc = pts["fixed_M1"], pts["oracle"]
        ax.annotate("", xy=(orc["cost_per_step"], orc["tsr"]),
                    xytext=(m1["cost_per_step"], m1["tsr"]),
                    arrowprops=dict(arrowstyle="->", color="#2ECC71", lw=1.3, ls="--"))
        savings = report.get("overall_savings_vs_m1", {}).get("savings_pct")
        if savings is not None:
            ax.annotate(f"oracle gap:\n-{savings}% cost", xy=(2.1, 0.605),
                        fontsize=7.5, color="#1e8449", ha="center",
                        annotation_clip=False)

    ax.set_xlabel("Avg. compute cost per step (x, M0=1)")
    ax.set_ylabel("Task Success Rate")
    ax.set_title("Cost-quality Pareto (ViTool)", fontsize=9)
    ax.legend(fontsize=6.3, loc="upper right", framealpha=0.92, ncol=1)
    ax.grid(True, alpha=0.25, zorder=0)
    ax.margins(x=0.08, y=0.10)
    fig.tight_layout()
    pdf = outdir / "fig2_pareto.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(outdir / "fig2_pareto.png", bbox_inches="tight")
    plt.close(fig)
    return pdf


def generate_all(report: dict, outdir: str, baselines: dict | None = None) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    f1 = plot_heatmap(report, out)
    f2 = plot_pareto(report, out, baselines=baselines)
    print(f"✅ Fig 1 (heatmap): {f1}")
    print(f"✅ Fig 2 (Pareto):  {f2}")
    print(f"   (PNG kèm theo trong {out})")


# ============================================================
# Self-test: report giả → vẽ → kiểm file ra
# ============================================================
def _selftest():
    report = {
        "n_steps": 200,
        "overall_savings_vs_m1": {"savings_pct": 42.0, "oracle_tsr": 0.95, "m1_tsr": 0.78},
        "mode_success_heatmap": {
            "easy": {"M0": 0.88, "M1": 0.92, "M2": 0.92, "M3": 0.95},
            "medium": {"M0": 0.55, "M1": 0.78, "M2": 0.85, "M3": 0.90},
            "hard": {"M0": 0.20, "M1": 0.45, "M2": 0.72, "M3": 0.88},
        },
        "pareto_points": {
            "fixed_M0": {"cost_per_step": 1.0, "tsr": 0.55},
            "fixed_M1": {"cost_per_step": 3.0, "tsr": 0.78},
            "fixed_M2": {"cost_per_step": 5.0, "tsr": 0.83},
            "fixed_M3": {"cost_per_step": 15.0, "tsr": 0.91},
            "oracle": {"cost_per_step": 1.74, "tsr": 0.95},
            "random": {"cost_per_step": 6.0, "tsr": 0.72},
        },
    }
    outdir = Path("/tmp/vitool_figs_test")
    generate_all(report, str(outdir))
    # kiểm file tồn tại + > 0 byte
    for f in ["fig1_heatmap.pdf", "fig1_heatmap.png", "fig2_pareto.pdf", "fig2_pareto.png"]:
        p = outdir / f
        assert p.exists() and p.stat().st_size > 0, f"thiếu/rỗng: {f}"
    print(f"\n✅ figures self-test PASS — 4 file sinh ra OK trong {outdir}")


def main():
    ap = argparse.ArgumentParser(description="ViTool paper figures")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--in", dest="in_path")
    ap.add_argument("--outdir", default="eval/figs")
    ap.add_argument("--baselines", default=None, help="eval/baselines.json (overlay router lên Pareto)")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    if not args.in_path:
        raise SystemExit("Cần --in (oracle_report.json) hoặc --selftest.")
    with open(args.in_path) as f:
        report = json.load(f)
    baselines = None
    if args.baselines:
        with open(args.baselines) as f:
            baselines = json.load(f)
    generate_all(report, args.outdir, baselines=baselines)


if __name__ == "__main__":
    main()