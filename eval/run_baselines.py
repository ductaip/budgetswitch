"""
eval/run_baselines.py — Router thực tế đối chứng oracle (v10.0).

VÌ SAO: oracle là cận trên lý tưởng (biết mode nào thắng từng step). Để oracle-gap
có nghĩa, ta cần các ROUTER THẬT — chính sách chỉ dùng thông tin có lúc inference,
KHÔNG nhìn nhãn success. Khoảng cách router-tốt-nhất → oracle = dư địa cho LEARNED
routing (đúng động lực của paper).

3 nhóm baseline (ranh giới methodology rõ ràng):
  (A) Fixed       : luôn 1 mode (M0/M1/M2/M3). Không quyết định gì.
  (B) Heuristic   : length-router + random. CHỈ dùng độ dài query / ngẫu nhiên —
                    KHÔNG peek nhãn → đây là baseline "công bằng", router thật.
  (C) Semi-oracle : difficulty-escalation. Biết TIER (gold) nhưng KHÔNG biết mode
                    nào thắng → mạnh hơn heuristic, yếu hơn full oracle.
                    (Tier không có lúc inference thật → đánh dấu rõ là cận trên phụ.)

Dùng:
  python eval/run_baselines.py \
      --results pilot_results_v4.jsonl \
      --eval    data/vitool/pilot_eval.jsonl \
      --out     eval/baselines.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.vitool.schema import load_trajectories  # noqa: E402

MODES = ["M0", "M1", "M2", "M3"]
MODE_COST = {"M0": 1.0, "M1": 3.0, "M2": 5.0, "M3": 15.0}
ORDER = {"easy": 0, "medium": 1, "hard": 2}


def _success(step, mode):
    r = step.get("results", {}).get(mode)
    return bool(r and r.get("success"))


def _cost(step, mode):
    r = step.get("results", {}).get(mode, {})
    return float(r.get("cost", MODE_COST[mode]))


def evaluate_router(steps, choose):
    """choose: step -> mode. Trả (cost_per_step, tsr)."""
    n = len(steps)
    tot_cost = sum(_cost(s, choose(s)) for s in steps)
    tot_ok = sum(_success(s, choose(s)) for s in steps)
    return tot_cost / n, tot_ok / n


def random_expectation(steps):
    """Kỳ vọng GIẢI TÍCH của router ngẫu nhiên (trung bình đều trên 4 mode)."""
    n = len(steps)
    cost = sum(MODE_COST.values()) / len(MODES)  # = 6.0
    tsr = sum(sum(_success(s, m) for m in MODES) / len(MODES) for s in steps) / n
    return cost, tsr


def oracle_point(steps):
    """Full oracle: mode rẻ nhất mà success, else M0. (Khớp eval/oracle.py.)"""
    def choose(s):
        ok = [(m, MODE_COST[m]) for m in MODES if _success(s, m)]
        return min(ok, key=lambda x: x[1])[0] if ok else "M0"
    return evaluate_router(steps, choose)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="pilot_results_v4.jsonl")
    ap.add_argument("--eval", default="data/vitool/pilot_eval.jsonl",
                    help="để lấy độ dài query cho length-router")
    ap.add_argument("--out", default="eval/baselines.json")
    args = ap.parse_args()

    steps = [json.loads(l) for l in open(args.results)]
    n = len(steps)

    # độ dài query (ký tự) theo traj_id — cho length-router
    qlen = {t.traj_id: len(t.query) for t in load_trajectories(args.eval)}
    lens = sorted(qlen.values())
    # ngưỡng tercile: ngắn→M0, vừa→M1, dài→M2 (router thật, không peek nhãn)
    t1, t2 = lens[len(lens) // 3], lens[2 * len(lens) // 3]

    def length_router(s):
        L = qlen.get(s["traj_id"], t1)
        if L <= t1:
            return "M0"
        if L <= t2:
            return "M1"
        return "M2"

    # difficulty-escalation (semi-oracle: biết tier, fixed easy→M0/med→M1/hard→M3)
    def diff_router(s):
        return {"easy": "M0", "medium": "M1", "hard": "M3"}[s["difficulty"]]

    rows = []
    # (A) fixed
    for m in MODES:
        c, t = evaluate_router(steps, lambda s, mm=m: mm)
        rows.append((f"Fixed {m}", "A-fixed", c, t))
    # (B) heuristic thật
    cr, tr = random_expectation(steps)
    rows.append(("Random", "B-heuristic", cr, tr))
    cl, tl = evaluate_router(steps, length_router)
    rows.append((f"Length-router (≤{t1}/≤{t2} chars)", "B-heuristic", cl, tl))
    # (C) semi-oracle
    cd, td = evaluate_router(steps, diff_router)
    rows.append(("Difficulty-escalation (gold tier)", "C-semi-oracle", cd, td))
    # full oracle
    co, to = oracle_point(steps)
    rows.append(("Oracle (full, upper bound)", "D-oracle", co, to))

    # in bảng
    print(f"\n{'='*64}\nBASELINE ROUTERS vs ORACLE  ({n} held-out steps)\n{'='*64}")
    print(f"{'Policy':<42}{'cost/step':>10}{'TSR':>8}{'%oracle':>9}")
    print("-" * 69)
    for name, grp, c, t in rows:
        pct = f"{t/to*100:4.0f}%" if to > 0 else "  -- "
        print(f"{name:<42}{c:>10.2f}{t:>8.3f}{pct:>9}")
    print("-" * 69)
    # khoảng cách router-tốt-nhất (loại fixed+oracle) → oracle
    realistic = [(nm, c, t) for nm, grp, c, t in rows if grp in ("B-heuristic", "C-semi-oracle")]
    best = max(realistic, key=lambda x: x[2])
    print(f"\nRouter THẬT tốt nhất: {best[0]} — TSR {best[2]:.3f} @ {best[1]:.2f}×")
    print(f"Oracle:               TSR {to:.3f} @ {co:.2f}×")
    print(f"→ Oracle-gap còn lại: +{(to-best[2]):.3f} TSR / cost {best[1]:.2f}→{co:.2f}× "
          f"= dư địa cho LEARNED routing")

    out = {
        "n_steps": n,
        "length_thresholds_chars": [t1, t2],
        "routers": [
            {"name": nm, "group": grp, "cost_per_step": round(c, 3), "tsr": round(t, 3),
             "pct_of_oracle_tsr": round(t / to, 3) if to > 0 else None}
            for nm, grp, c, t in rows
        ],
        "oracle_gap_vs_best_realistic_router": {
            "best_router": best[0],
            "router_tsr": round(best[2], 3),
            "oracle_tsr": round(to, 3),
            "tsr_gap": round(to - best[2], 3),
        },
        "_note": "Nhóm A=fixed, B=heuristic (không peek nhãn), C=semi-oracle (biết tier), D=full oracle.",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved → {args.out}")


if __name__ == "__main__":
    main()