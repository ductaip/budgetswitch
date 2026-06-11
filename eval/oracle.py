"""
eval/oracle.py — Oracle-gap + Pareto analysis cho ViTool (paper §5, v10.0).

ĐÂY LÀ LINH HỒN PAPER (đòn bẩy soundness lớn nhất): định lượng "một router
per-step lý tưởng tiết kiệm bao nhiêu compute" mà KHÔNG cần model học.

INPUT: file JSONL từ pilot/simulate_modes.py — mỗi dòng 1 step:
  {traj_id, step_idx, difficulty, oracle_mode, results: {M0:{success,cost,...}, ...}}

OUTPUT:
  - Oracle policy savings vs always-think (M1), per-difficulty + overall.
  - Pareto points (cost/query, TSR) cho: fixed M0/M1/M2/M3, oracle, random.
  - Bảng per-mode success theo difficulty (cho heatmap F1).

CRITICAL: oracle là tính TOÁN OFFLINE ("mode rẻ nhất mà success"), KHÔNG phải
policy học được (R4 — không liên quan GRPO). Đây là UPPER BOUND/cơ hội.

Dùng:
  python eval/oracle.py --in pilot/results.jsonl --out eval/oracle_report.json
  python eval/oracle.py --selftest      # test logic, không cần pilot thật
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODES = ["M0", "M1", "M2", "M3"]
MODE_COST = {"M0": 1.0, "M1": 3.0, "M2": 5.0, "M3": 15.0}
DIFFS = ["easy", "medium", "hard"]


# ============================================================
# Load pilot results
# ============================================================
def load_step_results(path: str) -> list[dict]:
    """Đọc JSONL step results từ simulate_modes."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _success(step: dict, mode: str) -> bool:
    r = step.get("results", {}).get(mode)
    return bool(r and r.get("success"))


def _cost(step: dict, mode: str) -> float:
    """Cost compute (multiplier). Ưu tiên cost ghi trong result, else MODE_COST."""
    r = step.get("results", {}).get(mode, {})
    return float(r.get("cost", MODE_COST[mode]))


def _usd(step: dict, mode: str) -> float:
    r = step.get("results", {}).get(mode, {})
    return float(r.get("api_cost_usd", 0.0))


# ============================================================
# Oracle policy
# ============================================================
def oracle_mode_for_step(step: dict) -> str:
    """Mode RẺ NHẤT mà success. Không mode nào success → M0 (fail RẺ).
    Cost-rational: oracle biết step vô vọng thì không đốt 15× để vẫn fail.
    (Fallback M3 cũ thổi cost oracle lên giả tạo trên các step unanswerable.)"""
    ok = [(m, MODE_COST[m]) for m in MODES if _success(step, m)]
    return min(ok, key=lambda x: x[1])[0] if ok else "M0"


def is_solvable(step: dict) -> bool:
    """Step có ÍT NHẤT 1 mode success."""
    return any(_success(step, m) for m in MODES)


# ============================================================
# Policy evaluation (cost + TSR cho 1 policy)
# ============================================================
def eval_fixed_mode(steps: list[dict], mode: str) -> dict:
    """Policy: luôn dùng 1 mode. TSR = % step success; cost = tổng cost."""
    n = len(steps)
    if n == 0:
        return {"tsr": 0.0, "cost_total": 0.0, "cost_per_step": 0.0, "usd_total": 0.0}
    succ = sum(_success(s, mode) for s in steps)
    cost = sum(_cost(s, mode) for s in steps)
    usd = sum(_usd(s, mode) for s in steps)
    return {"tsr": succ / n, "cost_total": cost, "cost_per_step": cost / n, "usd_total": usd}


def eval_oracle(steps: list[dict]) -> dict:
    """Policy oracle: mỗi step chọn mode rẻ nhất mà success."""
    n = len(steps)
    if n == 0:
        return {"tsr": 0.0, "cost_total": 0.0, "cost_per_step": 0.0, "usd_total": 0.0}
    cost = 0.0
    usd = 0.0
    succ = 0
    for s in steps:
        m = oracle_mode_for_step(s)
        cost += _cost(s, m)
        usd += _usd(s, m)
        succ += _success(s, m)  # =1 trừ khi không mode nào success
    return {"tsr": succ / n, "cost_total": cost, "cost_per_step": cost / n, "usd_total": usd}


def eval_random(steps: list[dict]) -> dict:
    """Policy random: TSR/cost kỳ vọng = trung bình 4 mode (đóng vai baseline ngẫu nhiên)."""
    n = len(steps)
    if n == 0:
        return {"tsr": 0.0, "cost_total": 0.0, "cost_per_step": 0.0, "usd_total": 0.0}
    tsr = sum(_success(s, m) for s in steps for m in MODES) / (n * len(MODES))
    cost = sum(_cost(s, m) for s in steps for m in MODES) / len(MODES)
    return {"tsr": tsr, "cost_total": cost, "cost_per_step": cost / n, "usd_total": 0.0}


# ============================================================
# Savings vs always-think (M1)
# ============================================================
def savings_vs_m1(steps: list[dict]) -> dict:
    """% compute tiết kiệm của oracle so với always-M1, ở TSR oracle≥M1."""
    m1 = eval_fixed_mode(steps, "M1")
    orc = eval_oracle(steps)
    if m1["cost_total"] <= 0:
        return {"savings_pct": 0.0, "oracle_tsr": orc["tsr"], "m1_tsr": m1["tsr"]}
    savings = (1 - orc["cost_total"] / m1["cost_total"]) * 100
    return {
        "savings_pct": round(savings, 1),
        "oracle_cost_per_step": round(orc["cost_per_step"], 3),
        "m1_cost_per_step": round(m1["cost_per_step"], 3),
        "oracle_tsr": round(orc["tsr"], 3),
        "m1_tsr": round(m1["tsr"], 3),
        "tsr_delta": round(orc["tsr"] - m1["tsr"], 3),
    }


# ============================================================
# Per-difficulty breakdown
# ============================================================
def by_difficulty(steps: list[dict]) -> dict:
    groups = defaultdict(list)
    for s in steps:
        groups[s.get("difficulty", "?")].append(s)
    report = {}
    for d in DIFFS:
        g = groups.get(d, [])
        if not g:
            report[d] = {"n": 0}
            continue
        report[d] = {"n": len(g), "savings": savings_vs_m1(g),
                     "oracle_mode_dist": _mode_dist([oracle_mode_for_step(s) for s in g])}
    return report


def _mode_dist(modes: list[str]) -> dict:
    n = len(modes)
    return {m: round(sum(1 for x in modes if x == m) / n, 3) for m in MODES} if n else {}


# ============================================================
# Heatmap data: per-mode success theo difficulty (F1)
# ============================================================
def mode_success_heatmap(steps: list[dict]) -> dict:
    """{difficulty: {mode: success_rate}} — dữ liệu cho Fig 1 (overthink/underthink)."""
    groups = defaultdict(list)
    for s in steps:
        groups[s.get("difficulty", "?")].append(s)
    hm = {}
    for d in DIFFS:
        g = groups.get(d, [])
        if not g:
            continue
        hm[d] = {m: round(sum(_success(s, m) for s in g) / len(g), 3) for m in MODES}
    return hm


# ============================================================
# Pareto points (cho Fig 2)
# ============================================================
def pareto_points(steps: list[dict]) -> dict:
    """{policy: {cost_per_step, tsr}} cho fixed modes + oracle + random."""
    pts = {f"fixed_{m}": eval_fixed_mode(steps, m) for m in MODES}
    pts["oracle"] = eval_oracle(steps)
    pts["random"] = eval_random(steps)
    return {k: {"cost_per_step": round(v["cost_per_step"], 3), "tsr": round(v["tsr"], 3)}
            for k, v in pts.items()}


# ============================================================
# Full report
# ============================================================
def build_report(steps: list[dict]) -> dict:
    n_unsolv = sum(1 for x in steps if not is_solvable(x))
    solvable = [x for x in steps if is_solvable(x)]
    return {
        "n_steps": len(steps),
        "unsolvable_rate": round(n_unsolv / len(steps), 3) if steps else 0.0,
        "overall_savings_vs_m1": savings_vs_m1(steps),
        "solvable_only_savings_vs_m1": savings_vs_m1(solvable),
        "by_difficulty": by_difficulty(steps),
        "mode_success_heatmap": mode_success_heatmap(steps),
        "pareto_points": pareto_points(steps),
        "_note": "Oracle = offline upper bound (mode rẻ nhất mà success). KHÔNG phải learned policy (R4).",
    }


def print_report(rep: dict) -> None:
    print(f"\n{'='*60}\nViTool Oracle-Gap Report ({rep['n_steps']} steps)\n{'='*60}")
    if "unsolvable_rate" in rep:
        print(f"  (unsolvable: {rep['unsolvable_rate']*100:.0f}% steps không mode nào success)")
    s = rep["overall_savings_vs_m1"]
    print(f"\n[OVERALL] Oracle tiết kiệm {s.get('savings_pct')}% compute vs always-think (M1)")
    print(f"  oracle: {s.get('oracle_cost_per_step')}×/step, TSR {s.get('oracle_tsr')}")
    print(f"  M1:     {s.get('m1_cost_per_step')}×/step, TSR {s.get('m1_tsr')} (ΔTSR {s.get('tsr_delta'):+})")
    print(f"\n[PER-DIFFICULTY savings vs M1]")
    for d in DIFFS:
        b = rep["by_difficulty"].get(d, {})
        if b.get("n"):
            print(f"  {d:7s} (n={b['n']:3d}): tiết kiệm {b['savings'].get('savings_pct')}% | "
                  f"oracle mode dist {b['oracle_mode_dist']}")
    print(f"\n[HEATMAP: per-mode success theo difficulty] (F1 — overthink/underthink)")
    print(f"  {'diff':8s} " + " ".join(f"{m:>7s}" for m in MODES))
    for d in DIFFS:
        hm = rep["mode_success_heatmap"].get(d)
        if hm:
            print(f"  {d:8s} " + " ".join(f"{hm[m]:7.2f}" for m in MODES))
    print(f"\n[PARETO points] (cost/step, TSR) — Fig 2")
    for pol, pt in rep["pareto_points"].items():
        print(f"  {pol:10s}: cost={pt['cost_per_step']:5.2f}  TSR={pt['tsr']:.3f}")


# ============================================================
# Self-test (data synthetic — chứng minh oracle/savings đúng)
# ============================================================
def _selftest():
    # Tạo step results giả khớp đúng giả thuyết overthink/underthink:
    # easy: M0 đã success (rẻ nhất) → oracle chọn M0
    # hard: M0 fail, M2 success → oracle chọn M2
    def mk(traj, idx, diff, succ_by_mode):
        return {"traj_id": traj, "step_idx": idx, "difficulty": diff,
                "results": {m: {"success": succ_by_mode[m], "cost": MODE_COST[m],
                                "api_cost_usd": (0.0005 if m == "M3" else 0.0)} for m in MODES}}

    steps = []
    # 10 easy: M0 success (mọi mode cũng success) → oracle = M0 (cost 1)
    for i in range(10):
        steps.append(mk(f"e{i}", 0, "easy", {"M0": True, "M1": True, "M2": True, "M3": True}))
    # 10 hard: M0+M1 fail, M2+M3 success → oracle = M2 (cost 5, rẻ hơn M3)
    for i in range(10):
        steps.append(mk(f"h{i}", 0, "hard", {"M0": False, "M1": False, "M2": True, "M3": True}))

    rep = build_report(steps)

    # M1 luôn-dùng: TSR = 10/20=0.5 (chỉ easy success), cost = 20*3=60
    m1 = eval_fixed_mode(steps, "M1")
    assert abs(m1["tsr"] - 0.5) < 1e-9 and abs(m1["cost_total"] - 60) < 1e-9, m1
    # Oracle: easy→M0 (10*1=10), hard→M2 (10*5=50) = 60 cost, TSR=1.0
    orc = eval_oracle(steps)
    assert abs(orc["cost_total"] - 60) < 1e-9 and abs(orc["tsr"] - 1.0) < 1e-9, orc
    print(f"✅ Oracle: cost {orc['cost_total']} (=M1) nhưng TSR {orc['tsr']} (>M1 {m1['tsr']})")
    print(f"   → oracle đạt TSR cao hơn HẲN ở cùng budget (đúng câu chuyện cần kể)")

    # heatmap đúng: easy M0=1.0, hard M0=0.0 M2=1.0
    hm = rep["mode_success_heatmap"]
    assert hm["easy"]["M0"] == 1.0 and hm["hard"]["M0"] == 0.0 and hm["hard"]["M2"] == 1.0
    print(f"✅ Heatmap đúng: easy M0={hm['easy']['M0']}, hard M0={hm['hard']['M0']}, hard M2={hm['hard']['M2']}")

    # per-difficulty oracle dist: easy toàn M0, hard toàn M2
    bd = rep["by_difficulty"]
    assert bd["easy"]["oracle_mode_dist"]["M0"] == 1.0
    assert bd["hard"]["oracle_mode_dist"]["M2"] == 1.0
    print(f"✅ Oracle dist: easy→M0 100%, hard→M2 100% (difficulty-aware đúng)")

    print_report(rep)
    print("\n✅ oracle.py self-test PASS.")


def main():
    ap = argparse.ArgumentParser(description="ViTool oracle-gap analysis")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--in", dest="in_path")
    ap.add_argument("--out", dest="out_path", default="eval/oracle_report.json")
    args = ap.parse_args()

    if args.selftest:
        _selftest()
        return
    if not args.in_path:
        raise SystemExit("Cần --in (file pilot results JSONL) hoặc --selftest.")
    steps = load_step_results(args.in_path)
    rep = build_report(steps)
    print_report(rep)
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved report → {args.out_path}")


if __name__ == "__main__":
    main()