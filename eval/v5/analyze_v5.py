"""
eval/v5/analyze_v5.py — merge v5 runs, compare with v4, trajectory metrics.

HEADLINE METRIC: TSR = trajectory succeeds iff every step is strictly correct
AND executing the model's calls in order reproduces the gold final env state
("TSR strict+state"). Every SSR is labelled with its protocol:
  SSR_tf      teacher-forced (gold history), as in v4
  SSR_rollout end-to-end (model's own history)

Grounding views (eval/v5/grounding_v5.py), reported for the strict audit and the
lenient sensitivity audit:
  (a) all          : all 112 trajectories, as is
  (b) grounded     : drop trajectories with any ungrounded gold value
  (c) clarify_ok   : on ungrounded steps, a clarification question (no call
                     emitted, text asks the user for the missing info) counts as
                     correct; trajectories that use such credit are scored on
                     strict steps only (no state check — the episode would stop)

Inputs (eval/v5/): raw/local_tf_v5.jsonl, raw/local_rollout_v5.jsonl,
  m3/m3_run{1,2,3}_v5.jsonl, m3_variants/*.jsonl, cache/grounding_steps_v5.jsonl
Outputs: pilot_results_v5.jsonl, metrics_v5.json
"""

from __future__ import annotations

import json
import math
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.v5.common_v5 import ROOT, is_clarification, load_eval, state_match  # noqa: E402

V5 = ROOT / "eval/v5"
MODES = ["M0", "M1", "M2", "M3"]
N_BOOT = 10_000
SEED = 42
# USD per 1M tokens (input, output) for models missing from modes/m3_defer._PRICING.
# Source: console.groq.com/docs/model/qwen/qwen3.8-27b, retrieved 2026-10-27.
EXTRA_PRICES = {"qwen/qwen3.8-27b": (0.80, 4.00)}
# Active params per token (for FLOPs): gpt-oss-120b model card "117B parameters with
# 5.1B active parameters"; qwen3.8-27b Groq page "A dense model with 27 billion parameters".
N_ACTIVE_M3 = {"openai/gpt-oss-120b": 5.1e9, "qwen/qwen3.8-27b": 27e9}


def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def key(r):
    return (r["traj_id"], r["step_idx"])


# ------------------------------------------------------------ stats
def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 4), round(c + h, 4)]


def rate(xs):
    xs = [bool(x) for x in xs]
    return {"k": sum(xs), "n": len(xs), "rate": round(sum(xs) / len(xs), 4) if xs else None,
            "ci95": wilson(sum(xs), len(xs))}


def mcnemar_exact(a, b):
    """Two-sided exact McNemar on paired bools. a_wins: a ok & b fail."""
    n01 = sum((not x) and y for x, y in zip(a, b))
    n10 = sum(x and (not y) for x, y in zip(a, b))
    n = n01 + n10
    if n == 0:
        return {"a_wins": 0, "b_wins": 0, "p": 1.0}
    k = min(n01, n10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"a_wins": n10, "b_wins": n01, "p": round(p, 6)}


def paired_bootstrap(ta, tb, rng):
    """ta, tb: dict traj_id -> bool. Bootstrap over trajectories; diff = b - a."""
    ids = sorted(ta)
    diff = sum(tb[i] - ta[i] for i in ids) / len(ids)
    boots = sorted(sum(tb[i] - ta[i] for i in (ids[rng.randrange(len(ids))] for _ in ids)) / len(ids)
                   for _ in range(N_BOOT))
    lo, hi = boots[int(0.025 * N_BOOT)], boots[int(0.975 * N_BOOT) - 1]
    p = 2 * min(sum(b <= 0 for b in boots), sum(b >= 0 for b in boots)) / N_BOOT
    return {"diff_b_minus_a": round(diff, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "p_two_sided": round(min(1.0, p), 4)}


# ------------------------------------------------------------ data
def merge():
    tf = {key(r): r for r in read(V5 / "raw/local_tf_v5.jsonl")}
    m3runs = [{key(r): r for r in read(p)} for k in (1, 2, 3)
              if (p := V5 / f"m3/m3_run{k}_v5.jsonl").exists()]
    rows = []
    for k, r in tf.items():
        row = {kk: r[kk] for kk in ("traj_id", "step_idx", "domain", "difficulty", "protocol")}
        row["results"] = dict(r["results"])
        if m3runs:
            m3 = dict(m3runs[0][k]["result"])
            m3["success_runs"] = [run[k]["result"]["success"] for run in m3runs]
            row["results"]["M3"] = m3
        rows.append(row)
    rows.sort(key=key)
    with open(V5 / "pilot_results_v5.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows, m3runs


class Ctx:
    def __init__(self, rows):
        self.trajs = {t.traj_id: t for t in load_eval(ROOT / "data/vitool/pilot_eval.jsonl")}
        g = {}
        for line in open(V5 / "cache/grounding_steps_v5.jsonl"):
            x = json.loads(line)
            g[(x["traj_id"], x["step_idx"])] = (x["grounded"], x["grounded_lenient"])
        self.ground = g
        self.ids = sorted({r["traj_id"] for r in rows})
        self.by_t = defaultdict(list)
        for r in rows:
            self.by_t[r["traj_id"]].append(r)
        for v in self.by_t.values():
            v.sort(key=lambda r: r["step_idx"])
        self.tier = {t: self.trajs[t].steps[0].difficulty.value for t in self.ids}
        self.dom = {t: self.trajs[t].domain for t in self.ids}

    def traj_grounded(self, t, lenient=False):
        return all(self.ground[(t, i)][lenient] for i in range(self.trajs[t].num_steps))


def step_records(rows, pick):
    """pick(row) -> result dict of the chosen mode at that step (or None)."""
    return {key(r): pick(r) for r in rows}


def traj_outcomes(ctx, recs, lenient=False):
    """Per trajectory: strict (all steps), state, headline (strict&state),
    clarify_ok variant. recs: (traj_id, step_idx) -> result dict."""
    out = {}
    for t in ctx.ids:
        res = [recs[(t, i)] for i in range(ctx.trajs[t].num_steps)]
        strict = all(x["success"] for x in res)
        stm = state_match([x["parsed_call"] for x in res], ctx.trajs[t].expected_final_state)
        credit = [(not x["success"]) and (not ctx.ground[(t, i)][lenient])
                  and is_clarification(x.get("raw_output", ""), x["parse_ok"])
                  for i, x in enumerate(res)]
        strict_c = all(x["success"] or c for x, c in zip(res, credit))
        out[t] = {"strict": strict, "state": stm, "tsr": strict and stm,
                  "tsr_clarify_ok": strict_c and (stm or any(credit)),
                  "clarify_credit_steps": sum(credit)}
    return out


def views(ctx, recs, lenient=False):
    """TSR (headline) + SSR_tf under the three grounding views."""
    to = traj_outcomes(ctx, recs, lenient)
    keep = [t for t in ctx.ids if ctx.traj_grounded(t, lenient)]
    all_steps = [recs[(t, i)] for t in ctx.ids for i in range(ctx.trajs[t].num_steps)]
    kept_steps = [recs[(t, i)] for t in keep for i in range(ctx.trajs[t].num_steps)]
    clar = [x["success"] or ((not ctx.ground[(t, i)][lenient])
                             and is_clarification(x.get("raw_output", ""), x["parse_ok"]))
            for t in ctx.ids for i, x in enumerate(recs[(t, j)] for j in range(ctx.trajs[t].num_steps))]
    return {
        "a_all": {"TSR": rate(to[t]["tsr"] for t in ctx.ids),
                  "TSR_strict_only": rate(to[t]["strict"] for t in ctx.ids),
                  "SSR_tf": rate(x["success"] for x in all_steps)},
        "b_grounded_only": {"n_trajs": len(keep), "TSR": rate(to[t]["tsr"] for t in keep),
                            "SSR_tf": rate(x["success"] for x in kept_steps)},
        "c_clarify_ok": {"TSR": rate(to[t]["tsr_clarify_ok"] for t in ctx.ids),
                         "SSR_tf": rate(clar),
                         "clarification_credits": sum(to[t]["clarify_credit_steps"] for t in ctx.ids)},
    }, to


def block(ctx, rows, modes, sub_ids):
    """Headline TSR + SSR_tf per mode (+ oracle) on a subset of trajectories."""
    sub = set(sub_ids)
    d = {}
    for m in modes + ["oracle"]:
        recs = step_records(rows, (lambda r, m=m: r["results"][m]) if m != "oracle"
                            else (lambda r: oracle_pick(r, modes)))
        to = traj_outcomes(ctx, recs)
        d[m] = {"TSR": rate(to[t]["tsr"] for t in sub_ids),
                "SSR_tf": rate(recs[key(r)]["success"] for r in rows if r["traj_id"] in sub)}
    return d


def oracle_pick(r, modes, cost=(1, 3, 5, 15)):
    """Cheapest strictly-correct mode at this step (else M0) — assumed costs."""
    ok = [(cost[MODES.index(m)], m) for m in modes if r["results"][m]["success"]]
    return r["results"][min(ok)[1] if ok else "M0"]


# ------------------------------------------------------------ main
def main():
    rows, m3runs = merge()
    ctx = Ctx(rows)
    modes = [m for m in MODES if all(m in r["results"] for r in rows)]
    v4 = {key(r): r for r in read(ROOT / "pilot_results_v4.jsonl")}
    rng = random.Random(SEED)
    out = {"headline_metric": "TSR = all steps strictly correct AND final env state matches gold",
           "n_steps": len(rows), "n_trajs": len(ctx.ids), "modes": modes, "m3_runs": len(m3runs)}

    # ---------------- step 2: v4 vs v5 (teacher-forced)
    cmp = {}
    for m in modes:
        a = [v4[key(r)]["results"][m]["success"] for r in rows]
        b = [r["results"][m]["success"] for r in rows]
        res = [r["results"][m] for r in rows]
        lat = [x["latency_ms"] for x in res if x.get("latency_ms") is not None]
        cmp[m] = {
            "SSR_tf_v4": round(sum(a) / len(a), 4), "SSR_tf_v5": round(sum(b) / len(b), 4),
            "per_step_agreement": round(sum(x == y for x, y in zip(a, b)) / len(a), 4),
            "v4_ok_v5_fail": sum(x and not y for x, y in zip(a, b)),
            "v4_fail_v5_ok": sum(y and not x for x, y in zip(a, b)),
            "mcnemar_v4_vs_v5": mcnemar_exact(a, b),
            "parse_ok_rate": round(st.mean(x["parse_ok"] for x in res), 4),
            "exec_error_rate": round(st.mean(x["exec_error"] for x in res), 4),
            "mean_prompt_tokens": round(st.mean(x["prompt_tokens"] for x in res if x.get("prompt_tokens") is not None), 1),
            "mean_completion_tokens": round(st.mean(x["completion_tokens"] for x in res if x.get("completion_tokens") is not None), 1),
            "latency_ms_mean": round(st.mean(lat), 1), "latency_ms_median": round(st.median(lat), 1),
        }
        if "hit_max_new_tokens" in res[0]:
            cmp[m]["hit_max_new_tokens"] = sum(x["hit_max_new_tokens"] for x in res)
    if m3runs:
        per_run = []
        for run in m3runs:
            recs = {k: v["result"] for k, v in run.items()}
            to = traj_outcomes(ctx, recs)
            per_run.append({"SSR_tf": sum(x["success"] for x in recs.values()) / len(recs),
                            "TSR": sum(o["tsr"] for o in to.values()) / len(to)})
        pair = [sum(m3runs[i][k]["result"]["success"] == m3runs[j][k]["result"]["success"]
                    for k in m3runs[0]) / len(m3runs[0])
                for i in range(len(m3runs)) for j in range(i + 1, len(m3runs))]
        cmp["M3_repeats"] = {
            "SSR_tf_runs": [round(x["SSR_tf"], 4) for x in per_run],
            "SSR_tf_mean": round(st.mean(x["SSR_tf"] for x in per_run), 4),
            "SSR_tf_std": round(st.stdev(x["SSR_tf"] for x in per_run), 4) if len(per_run) > 1 else None,
            "TSR_runs": [round(x["TSR"], 4) for x in per_run],
            "TSR_mean": round(st.mean(x["TSR"] for x in per_run), 4),
            "TSR_std": round(st.stdev(x["TSR"] for x in per_run), 4) if len(per_run) > 1 else None,
            "pairwise_step_agreement": [round(x, 4) for x in pair],
            "api_errors": [sum(bool(v["result"].get("api_error")) for v in run.values()) for run in m3runs],
        }
    out["step2_v4_vs_v5"] = cmp

    # ---------------- step 3: TSR (headline) + SSR_tf, breakdowns, tests
    s3 = {"overall": block(ctx, rows, modes, ctx.ids)}
    s3["by_difficulty"] = {k: block(ctx, rows, modes, [t for t in ctx.ids if ctx.tier[t] == k])
                           for k in ("easy", "medium", "hard")}
    domains = sorted(set(ctx.dom.values()))
    s3["by_domain"] = {k: block(ctx, rows, modes, [t for t in ctx.ids if ctx.dom[t] == k])
                       for k in domains}
    s3["domain_macro_avg"] = {m: {s: round(st.mean(s3["by_domain"][d][m][s]["rate"] for d in domains), 4)
                                  for s in ("TSR", "SSR_tf")} for m in modes + ["oracle"]}
    tsr_t = {m: traj_outcomes(ctx, step_records(rows, lambda r, m=m: r["results"][m])) for m in modes}
    base = [r["results"]["M0"]["success"] for r in rows]
    s3["paired_tests_vs_M0"] = {
        f"M0_vs_{m}": {
            "SSR_tf_mcnemar": mcnemar_exact(base, [r["results"][m]["success"] for r in rows]),
            "TSR_paired_bootstrap": paired_bootstrap({t: o["tsr"] for t, o in tsr_t["M0"].items()},
                                                     {t: o["tsr"] for t, o in tsr_t[m].items()}, rng)}
        for m in modes[1:]}
    out["step3_teacher_forced"] = s3

    # ---------------- grounding views (a)/(b)/(c)
    gv = {}
    for lenient, name in ((False, "strict_audit"), (True, "lenient_audit")):
        gv[name] = {m: views(ctx, step_records(rows, lambda r, m=m: r["results"][m]), lenient)[0]
                    for m in modes}
        gv[name]["oracle"] = views(ctx, step_records(rows, lambda r: oracle_pick(r, modes)), lenient)[0]
    out["grounding_views"] = gv

    # ---------------- rollout + per-trajectory agreement with teacher-forced
    rp = V5 / "raw/local_rollout_v5.jsonl"
    if rp.exists():
        rr = read(rp)
        ro = {}
        for m in sorted({r["mode"] for r in rr if r["protocol"] == "rollout"}):
            recs = {key(r): r for r in rr if r["protocol"] == "rollout" and r["mode"] == m}
            fin = {r["traj_id"]: r["final_state_match"] for r in rr
                   if r["protocol"] == "rollout_final" and r["mode"] == m}
            strict = {t: all(recs[(t, i)]["success"] for i in range(ctx.trajs[t].num_steps)) for t in ctx.ids}
            ro_tsr = {t: strict[t] and fin[t] for t in ctx.ids}
            tf_o = tsr_t[m] if m in tsr_t else None
            entry = {
                "SSR_rollout": rate(x["success"] for x in recs.values()),
                "TSR_rollout": rate(ro_tsr.values()),
                "TSR_rollout_strict_only": rate(strict.values()),
                "final_state_match_only": rate(fin.values()),
                "by_difficulty_TSR_rollout": {k: rate(ro_tsr[t] for t in ctx.ids if ctx.tier[t] == k)
                                              for k in ("easy", "medium", "hard")},
                "exec_error_rate": round(st.mean(x["exec_error"] for x in recs.values()), 4),
                "grounding_views_strict_audit": views(ctx, recs, False)[0],
            }
            if tf_o:
                tf_tsr = {t: o["tsr"] for t, o in tf_o.items()}
                entry["per_traj_vs_teacher_forced"] = {
                    "TSR_flag_agreement": f"{sum(ro_tsr[t] == tf_tsr[t] for t in ctx.ids)}/{len(ctx.ids)}",
                    "tf_ok_rollout_fail": sorted(t for t in ctx.ids if tf_tsr[t] and not ro_tsr[t]),
                    "rollout_ok_tf_fail": sorted(t for t in ctx.ids if ro_tsr[t] and not tf_tsr[t]),
                    "strict_flag_agreement": f"{sum(strict[t] == tf_o[t]['strict'] for t in ctx.ids)}/{len(ctx.ids)}",
                    "SSR_tf_minus_SSR_rollout": round(
                        st.mean(r["results"][m]["success"] for r in rows)
                        - st.mean(x["success"] for x in recs.values()), 4),
                }
            ro[m] = entry
        out["step3_rollout"] = ro

    # ---------------- step 8: M3 variants (teacher-forced)
    var = {}
    for p in sorted((V5 / "m3_variants").glob("*_v5.jsonl")):
        rs = read(p)
        if len(rs) < len(rows):
            var[p.stem] = {"status": f"incomplete {len(rs)}/{len(rows)}"}
            continue
        recs = {key(r): r["result"] for r in rs}
        v, _ = views(ctx, recs, False)
        xs = list(recs.values())
        model = json.loads(p.with_suffix(".meta.json").read_text()).get("model") \
            if p.with_suffix(".meta.json").exists() else None
        if model in EXTRA_PRICES:  # v4 pricing table lacks this model; price from Groq page
            pi, po = EXTRA_PRICES[model]
            for x in xs:
                if x.get("prompt_tokens") is not None:
                    x["api_cost_usd"] = x["prompt_tokens"] / 1e6 * pi + x["completion_tokens"] / 1e6 * po
        usd = [x["api_cost_usd"] for x in xs if x.get("api_cost_usd") is not None]
        meta = json.loads(p.with_suffix(".meta.json").read_text()) if p.with_suffix(".meta.json").exists() else {}
        var[p.stem] = {
            "model": meta.get("model"), "fewshot": meta.get("fewshot_demos"),
            "grounded_demos": meta.get("grounded_demos", False),
            "TSR": v["a_all"]["TSR"], "SSR_tf": v["a_all"]["SSR_tf"], "grounding_views": v,
            "parse_ok_rate": round(st.mean(x["parse_ok"] for x in xs), 4),
            "mean_prompt_tokens": round(st.mean(x["prompt_tokens"] for x in xs if x.get("prompt_tokens")), 1),
            "mean_completion_tokens": round(st.mean(x["completion_tokens"] for x in xs if x.get("completion_tokens") is not None), 1),
            "usd_total_284": round(sum(usd), 4) if len(usd) == len(xs) else None,
            "N_active": N_ACTIVE_M3.get(model),
            "flops_mean": float(f"{2 * N_ACTIVE_M3[model] * st.mean((x['prompt_tokens'] or 0) + (x['completion_tokens'] or 0) for x in xs):.4g}")
            if model in N_ACTIVE_M3 else None,
            "api_errors": sum(bool(x.get("api_error")) for x in xs),
        }
    out["step8_m3_variants"] = var

    (V5 / "metrics_v5.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


if __name__ == "__main__":
    o = main()
    print(json.dumps({k: o[k] for k in ("step2_v4_vs_v5",)}, indent=1))
