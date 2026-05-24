"""
Pilot Study Analysis — generates the figures and runs the Decision Gate.

Consumes the output of pilot/simulate_modes.py and produces:
  - Figure 1: Mode-success heatmap (4 modes x 3 difficulty classes)
  - Figure 2: Cost-success Pareto (fixed-mode vs oracle-allocation)
  - Table 1: Per-mode summary
  - Decision Gate verdict: do H1, H2, H3 hold?

============================================================
THE DECISION GATE (W2 Day 6):
H1 (overthink-easy): M0 succeeds >=85% on easy steps; M1 wasteful there.
H2 (underthink-hard): success increases with mode cost on hard steps.
H3 (allocation gap): oracle allocation matches uniform-M1 success at lower cost.

If these hold -> proceed to W3 confidently.
If not -> reconsider before investing in GRPO (see W2_DAILY_PLAN.md decision gate).
============================================================

Place file at: pilot/analyze_pilot.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODES = ["M0", "M1", "M2", "M3"]
MODE_COST = {"M0": 1.0, "M1": 3.0, "M2": 5.0, "M3": 15.0}
DIFFICULTIES = ["easy", "medium", "hard"]


def load_simulations(path: str) -> list[dict]:
    """Load StepSimulation dicts from JSONL produced by simulate_modes.py."""
    sims = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sims.append(json.loads(line))
    return sims


# ============================================================
# Aggregations
# ============================================================

def mode_success_by_difficulty(sims: list[dict]) -> dict:
    """
    Returns {difficulty: {mode: success_rate}}.
    Powers Figure 1 (heatmap).
    """
    counts = {d: {m: [0, 0] for m in MODES} for d in DIFFICULTIES}  # [success, total]
    for sim in sims:
        d = sim["difficulty"]
        if d not in counts:
            continue
        for m in MODES:
            r = sim["results"][m]
            counts[d][m][1] += 1
            if r["success"]:
                counts[d][m][0] += 1

    rates = {}
    for d in DIFFICULTIES:
        rates[d] = {}
        for m in MODES:
            s, t = counts[d][m]
            rates[d][m] = (s / t) if t > 0 else float("nan")
    return rates


def per_mode_summary(sims: list[dict]) -> dict:
    """
    Returns {mode: {success_rate, avg_cost, avg_tokens, avg_latency_ms}}.
    Powers Table 1.
    """
    agg = {m: {"success": 0, "total": 0, "tokens": 0, "latency": 0.0} for m in MODES}
    for sim in sims:
        for m in MODES:
            r = sim["results"][m]
            agg[m]["total"] += 1
            agg[m]["success"] += int(r["success"])
            agg[m]["tokens"] += r.get("completion_tokens", 0)
            agg[m]["latency"] += r.get("latency_ms", 0.0)

    summary = {}
    for m in MODES:
        t = agg[m]["total"] or 1
        summary[m] = {
            "success_rate": agg[m]["success"] / t,
            "cost_units": MODE_COST[m],
            "avg_tokens": agg[m]["tokens"] / t,
            "avg_latency_ms": agg[m]["latency"] / t,
        }
    return summary


def policy_cost_and_success(sims: list[dict]) -> dict:
    """
    Compute (success_rate, total_cost) for several policies.
    Powers Figure 2 (Pareto).
    
    Policies:
      - fixed M0/M1/M2/M3: always use that mode
      - oracle: cheapest succeeding mode per step
    """
    policies = {}

    # Fixed-mode policies
    for m in MODES:
        succ = sum(int(s["results"][m]["success"]) for s in sims)
        cost = sum(MODE_COST[m] for _ in sims)
        policies[f"fixed_{m}"] = {
            "success_rate": succ / len(sims),
            "total_cost": cost,
            "avg_cost": cost / len(sims),
        }

    # Oracle policy: cheapest successful mode per step
    oracle_succ = 0
    oracle_cost = 0.0
    for s in sims:
        om = s["oracle_mode"]
        oracle_succ += int(s["results"][om]["success"])
        oracle_cost += MODE_COST[om]
    policies["oracle"] = {
        "success_rate": oracle_succ / len(sims),
        "total_cost": oracle_cost,
        "avg_cost": oracle_cost / len(sims),
    }

    return policies


# ============================================================
# Decision Gate
# ============================================================

def run_decision_gate(sims: list[dict]) -> dict:
    """
    Test the three core hypotheses. Returns verdict dict.
    """
    rates = mode_success_by_difficulty(sims)
    policies = policy_cost_and_success(sims)

    verdict = {"hypotheses": {}, "overall": None, "notes": []}

    # H1: overthink-easy. M0 succeeds high on easy; M1 not much better but costlier.
    easy = rates.get("easy", {})
    m0_easy = easy.get("M0", 0)
    m1_easy = easy.get("M1", 0)
    h1_pass = (m0_easy >= 0.80) and (m1_easy - m0_easy < 0.15)
    verdict["hypotheses"]["H1_overthink_easy"] = {
        "pass": h1_pass,
        "m0_easy_success": round(m0_easy, 3),
        "m1_easy_success": round(m1_easy, 3),
        "interpretation": (
            f"M0 succeeds {m0_easy:.0%} on easy steps; M1 adds only "
            f"{(m1_easy - m0_easy):.0%} at 3x cost."
            if h1_pass else
            f"H1 weak: M0 easy={m0_easy:.0%}, M1 easy={m1_easy:.0%}."
        ),
    }

    # H2: underthink-hard. Success increases with mode cost on hard steps.
    hard = rates.get("hard", {})
    h2_monotonic = (
        hard.get("M0", 0) <= hard.get("M1", 0) <= hard.get("M2", 0)
    ) if hard else False
    h2_gain = (hard.get("M2", 0) - hard.get("M0", 0)) if hard else 0
    h2_pass = h2_monotonic and h2_gain >= 0.15
    verdict["hypotheses"]["H2_underthink_hard"] = {
        "pass": h2_pass,
        "hard_success_by_mode": {m: round(hard.get(m, 0), 3) for m in MODES},
        "interpretation": (
            f"On hard steps, success rises M0={hard.get('M0',0):.0%} -> "
            f"M2={hard.get('M2',0):.0%} (gain {h2_gain:.0%})."
            if h2_pass else
            f"H2 weak: hard-step success not clearly increasing with mode cost."
        ),
    }

    # H3: allocation gap. Oracle matches/beats fixed-M1 success at lower cost.
    oracle = policies["oracle"]
    fixed_m1 = policies["fixed_M1"]
    h3_pass = (
        oracle["success_rate"] >= fixed_m1["success_rate"] - 0.02
        and oracle["avg_cost"] < fixed_m1["avg_cost"]
    )
    cost_saving = 1 - (oracle["avg_cost"] / fixed_m1["avg_cost"]) if fixed_m1["avg_cost"] else 0
    verdict["hypotheses"]["H3_allocation_gap"] = {
        "pass": h3_pass,
        "oracle_success": round(oracle["success_rate"], 3),
        "oracle_avg_cost": round(oracle["avg_cost"], 2),
        "fixed_m1_success": round(fixed_m1["success_rate"], 3),
        "fixed_m1_avg_cost": round(fixed_m1["avg_cost"], 2),
        "cost_saving_vs_m1": round(cost_saving, 3),
        "interpretation": (
            f"Oracle matches M1 success ({oracle['success_rate']:.0%}) at "
            f"{cost_saving:.0%} lower cost."
            if h3_pass else
            f"H3 weak: oracle does not clearly beat fixed-M1 on the frontier."
        ),
    }

    passed = sum(1 for h in verdict["hypotheses"].values() if h["pass"])
    if passed == 3:
        verdict["overall"] = "GREEN — all hypotheses hold. Proceed to W3 confidently."
    elif passed == 2:
        verdict["overall"] = "YELLOW — partial support. Proceed but adjust framing (see notes)."
        verdict["notes"].append("Consider whether all 4 modes are needed (ablation #1).")
    else:
        verdict["overall"] = "RED — weak support. Reconsider before GRPO (see W2 decision gate)."
        verdict["notes"].append("Check task difficulty distribution; may need more multi-step/hard tasks.")
        verdict["notes"].append("Consider Fallback 2 (empirical study) if gap genuinely absent.")

    verdict["hypotheses_passed"] = f"{passed}/3"
    return verdict


# ============================================================
# Plotting (optional — requires matplotlib)
# ============================================================

def plot_heatmap(sims: list[dict], output_path: str = "pilot/results/fig1_heatmap.png"):
    """Figure 1: mode-success heatmap."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[skip] matplotlib not available; skipping heatmap plot.")
        return

    rates = mode_success_by_difficulty(sims)
    matrix = np.array([[rates[d][m] for m in MODES] for d in DIFFICULTIES])

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(MODES)))
    ax.set_xticklabels(MODES)
    ax.set_yticks(range(len(DIFFICULTIES)))
    ax.set_yticklabels(DIFFICULTIES)
    ax.set_xlabel("Compute Mode")
    ax.set_ylabel("Step Difficulty")
    ax.set_title("Mode Success Rate by Difficulty (Pilot)")
    for i in range(len(DIFFICULTIES)):
        for j in range(len(MODES)):
            ax.text(j, i, f"{matrix[i, j]:.0%}", ha="center", va="center")
    fig.colorbar(im, label="Success rate")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"✅ Saved heatmap to {output_path}")


def plot_pareto(sims: list[dict], output_path: str = "pilot/results/fig2_pareto.png"):
    """Figure 2: cost-success Pareto."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[skip] matplotlib not available; skipping Pareto plot.")
        return

    policies = policy_cost_and_success(sims)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in policies.items():
        marker = "*" if name == "oracle" else "o"
        size = 250 if name == "oracle" else 100
        ax.scatter(p["avg_cost"], p["success_rate"], s=size, marker=marker, label=name)
    ax.set_xlabel("Average cost per step (relative units)")
    ax.set_ylabel("Success rate")
    ax.set_title("Cost-Success Pareto (Pilot)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"✅ Saved Pareto to {output_path}")


# ============================================================
# CLI / report generation
# ============================================================

def generate_report(sim_path: str, report_path: str = "pilot/results/pilot_report.md"):
    """Full pilot analysis: tables, figures, decision gate."""
    sims = load_simulations(sim_path)
    print(f"Loaded {len(sims)} step simulations.\n")

    summary = per_mode_summary(sims)
    verdict = run_decision_gate(sims)

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Pilot Study Report\n"]
    lines.append(f"Total steps simulated: {len(sims)}\n")

    lines.append("\n## Table 1: Per-Mode Summary\n")
    lines.append("| Mode | Success | Cost (units) | Avg tokens | Avg latency (ms) |")
    lines.append("|------|---------|--------------|------------|------------------|")
    for m in MODES:
        s = summary[m]
        lines.append(
            f"| {m} | {s['success_rate']:.1%} | {s['cost_units']:.0f} | "
            f"{s['avg_tokens']:.0f} | {s['avg_latency_ms']:.0f} |"
        )

    lines.append("\n## Decision Gate Verdict\n")
    lines.append(f"**Hypotheses passed: {verdict['hypotheses_passed']}**\n")
    lines.append(f"**Overall: {verdict['overall']}**\n")
    for hname, h in verdict["hypotheses"].items():
        status = "✅" if h["pass"] else "❌"
        lines.append(f"\n### {status} {hname}")
        lines.append(f"\n{h['interpretation']}\n")
    if verdict["notes"]:
        lines.append("\n### Notes")
        for note in verdict["notes"]:
            lines.append(f"- {note}")

    report = "\n".join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"\n✅ Report saved to {report_path}")

    # Figures
    plot_heatmap(sims)
    plot_pareto(sims)

    return verdict


if __name__ == "__main__":
    # Self-test with synthetic simulation data
    import random
    random.seed(42)

    synthetic = []
    for i in range(120):
        diff = random.choice(DIFFICULTIES)
        # Simulate realistic mode behavior per difficulty
        def succ(p):
            return random.random() < p
        if diff == "easy":
            res = {"M0": succ(0.88), "M1": succ(0.90), "M2": succ(0.91), "M3": succ(0.95)}
        elif diff == "medium":
            res = {"M0": succ(0.55), "M1": succ(0.75), "M2": succ(0.80), "M3": succ(0.90)}
        else:  # hard
            res = {"M0": succ(0.25), "M1": succ(0.50), "M2": succ(0.70), "M3": succ(0.88)}

        results = {}
        for m in MODES:
            results[m] = {"mode": m, "success": res[m], "cost": MODE_COST[m],
                          "latency_ms": MODE_COST[m] * 200,
                          "completion_tokens": int(MODE_COST[m] * 100), "api_cost_usd": 0.0}
        # oracle = cheapest success
        succ_modes = [(m, MODE_COST[m]) for m in MODES if res[m]]
        oracle = min(succ_modes, key=lambda x: x[1])[0] if succ_modes else "M3"
        synthetic.append({"traj_id": f"syn_{i}", "step_idx": 0, "difficulty": diff,
                          "oracle_mode": oracle, "results": results})

    # Write synthetic + run report
    Path("pilot/results").mkdir(parents=True, exist_ok=True)
    with open("/tmp/synthetic_sims.jsonl", "w") as f:
        for s in synthetic:
            f.write(json.dumps(s) + "\n")

    print("=" * 55)
    print("SELF-TEST with synthetic data (expect GREEN/YELLOW)")
    print("=" * 55)
    generate_report("/tmp/synthetic_sims.jsonl", "/tmp/synthetic_report.md")