"""
eval/v6/analyze_v6.py — Step 5: metrics on ViTool v1.1 and the v1 vs v1.1 comparison.

HEADLINE METRIC: TSR = every step strictly correct AND executing the model's calls
reproduces the gold final env state. Every SSR names its protocol (SSR_tf teacher-
forced, SSR_rollout end-to-end). Statistics reused from eval/v5/analyze_v5.py
(Wilson CIs, exact McNemar on steps, paired bootstrap 10k by trajectory, seed 42).

Conditions (same 112 eval trajectories / 284 steps, same gold calls):
  v1            v1 queries, v4 adapter        (eval/v5 runs, unchanged)
  v1.1_v4adapter v1.1 queries, v4 adapter     (isolates "better queries")
  v1.1          v1.1 queries, adapter retrained on v1.1 train  <- MAIN
M3 = gpt-oss-120b zero-shot on the condition's queries (run 1 is the M3 used in
tables/oracle; runs 2-3 give the spread). v1.1_v4adapter reuses the v1.1 M3 runs.

Grounding per step = Gate G3 (data/grounding_gate.py) on the condition's queries.
Clarification analysis: an output with no tool call that asks the user for
information (eval/v5/common_v5.is_clarification). On v1 it is arguably right on
G3-ungrounded steps; on v1.1 every step is grounded, so it is wrong.

Outputs: eval/v6/pilot_results_v6.jsonl (main condition, merged), eval/v6/metrics_v6.json
"""

from __future__ import annotations

import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.grounding_gate import grounding_report  # noqa: E402
from eval.v5.analyze_v5 import (  # noqa: E402
    EXTRA_PRICES, MODES, N_ACTIVE_M3, SEED, block, key, mcnemar_exact, oracle_pick, paired_bootstrap, rate, read,
    step_records, traj_outcomes,
)
from eval.v5.common_v5 import ROOT, is_clarification, load_catalog, load_eval  # noqa: E402

V5, V6 = ROOT / "eval/v5", ROOT / "eval/v6"
DATA = ROOT / "data/vitool"

CONDITIONS = {
    "v1": {"eval": DATA / "pilot_eval.jsonl", "tf": V5 / "raw/local_tf_v5.jsonl",
           "rollout": V5 / "raw/local_rollout_v5.jsonl",
           "m3": [V5 / f"m3/m3_run{k}_v5.jsonl" for k in (1, 2, 3)],
           "adapter": "v4 (tool_use_lora_v2)"},
    "v1.1_v4adapter": {"eval": DATA / "pilot_eval_v1.1.jsonl", "tf": V6 / "raw/local_tf_v4adapter_v11.jsonl",
                       "rollout": V6 / "raw/local_rollout_v4adapter_v11.jsonl",
                       "m3": [V6 / f"m3/m3_run{k}_v6.jsonl" for k in (1, 2, 3)],
                       "adapter": "v4 (tool_use_lora_v2)"},
    "v1.1": {"eval": DATA / "pilot_eval_v1.1.jsonl", "tf": V6 / "raw/local_tf_v6adapter_v11.jsonl",
             "rollout": V6 / "raw/local_rollout_v6adapter_v11.jsonl",
             "m3": [V6 / f"m3/m3_run{k}_v6.jsonl" for k in (1, 2, 3)],
             "adapter": "v6 (retrained on v1.1 train)"},
    "v1.2": {"eval": DATA / "pilot_eval_v1.2.jsonl", "tf": V6 / "raw/local_tf_v4adapter_v12.jsonl",
             "rollout": V6 / "raw/local_rollout_v4adapter_v12.jsonl",
             "m3": [V6 / "m3_v12/m3_gptoss_0shot_run1_v12.jsonl"],
             "adapter": "v4 (tool_use_lora_v2)"},
}
MAIN = "v1.1"
RESULTS_FILE = {"v1.1": "pilot_results_v6.jsonl", "v1.1_v4adapter": "pilot_results_v11_v4adapter.jsonl",
                "v1.2": "pilot_results_v12.jsonl"}


class Ctx:
    """Same interface as eval/v5/analyze_v5.Ctx, but grounding = G3 on this condition's queries."""

    def __init__(self, rows, eval_path, catalog):
        self.trajs = {t.traj_id: t for t in load_eval(eval_path)}
        self.ground = {}
        for t in self.trajs.values():
            bad = grounding_report(t.query, [(s.expected_tool, s.expected_params) for s in t.steps], catalog)
            badi = {b["step"] for b in bad}
            for i in range(t.num_steps):
                g = i not in badi
                self.ground[(t.traj_id, i)] = (g, g)  # (strict, lenient) slots of the v5 helpers
        self.ids = sorted({r["traj_id"] for r in rows})
        self.by_t = defaultdict(list)
        for r in rows:
            self.by_t[r["traj_id"]].append(r)
        self.tier = {t: self.trajs[t].steps[0].difficulty.value for t in self.ids}
        self.dom = {t: self.trajs[t].domain for t in self.ids}

    def traj_grounded(self, t, lenient=False):
        return all(self.ground[(t, i)][lenient] for i in range(self.trajs[t].num_steps))


def merge(cond):
    c = CONDITIONS[cond]
    if not c["tf"].exists():
        return None, []
    tf = {key(r): r for r in read(c["tf"])}
    m3runs = [{key(r): r for r in read(p)} for p in c["m3"] if p.exists() and len(read(p)) == len(tf)]
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
    return rows, m3runs


def clarification(ctx, rows, modes, rollout=None):
    out = {}
    for m in modes:
        steps = [(r, r["results"][m]) for r in rows]
        cl = [(ctx.ground[key(r)][0], is_clarification(x.get("raw_output", ""), x["parse_ok"])) for r, x in steps]
        g = [c for gr, c in cl if gr]
        u = [c for gr, c in cl if not gr]
        out[m] = {"teacher_forced": {
            "clarify_rate_all_steps": rate(c for _, c in cl),
            "clarify_rate_on_grounded_steps (wrong)": rate(g) if g else None,
            "clarify_rate_on_ungrounded_steps (arguably right)": rate(u) if u else None,
            "n_ungrounded_steps": len(u)}}
    if rollout:
        for m, recs in rollout.items():
            xs = list(recs.values())
            out.setdefault(m, {})["rollout"] = {
                "clarify_rate_all_steps": rate(is_clarification(x.get("raw_output", ""), x["parse_ok"]) for x in xs)}
    return out


def analyze(cond, catalog, rng):
    rows, m3runs = merge(cond)
    if rows is None:
        return {"status": f"missing {CONDITIONS[cond]['tf'].name}"}, None, None
    ctx = Ctx(rows, CONDITIONS[cond]["eval"], catalog)
    modes = [m for m in MODES if all(m in r["results"] for r in rows)]
    o = {"queries": CONDITIONS[cond]["eval"].name, "adapter": CONDITIONS[cond]["adapter"],
         "n_steps": len(rows), "n_trajs": len(ctx.ids), "modes": modes, "m3_runs": len(m3runs),
         "g3_ungrounded_trajs": sum(not ctx.traj_grounded(t) for t in ctx.ids),
         "g3_ungrounded_steps": sum(not v[0] for k, v in ctx.ground.items() if k[0] in set(ctx.ids))}

    tf = {"overall": block(ctx, rows, modes, ctx.ids)}
    tf["by_difficulty"] = {k: block(ctx, rows, modes, [t for t in ctx.ids if ctx.tier[t] == k])
                           for k in ("easy", "medium", "hard")}
    domains = sorted(set(ctx.dom.values()))
    tf["by_domain"] = {k: block(ctx, rows, modes, [t for t in ctx.ids if ctx.dom[t] == k]) for k in domains}
    tf["domain_macro_avg"] = {m: {s: round(st.mean(tf["by_domain"][d][m][s]["rate"] for d in domains), 4)
                                  for s in ("TSR", "SSR_tf")} for m in modes + ["oracle"]}
    tsr_t = {m: traj_outcomes(ctx, step_records(rows, lambda r, m=m: r["results"][m])) for m in modes}
    base = [r["results"]["M0"]["success"] for r in rows]
    tf["paired_tests_vs_M0"] = {
        f"M0_vs_{m}": {"SSR_tf_mcnemar": mcnemar_exact(base, [r["results"][m]["success"] for r in rows]),
                       "TSR_paired_bootstrap": paired_bootstrap({t: x["tsr"] for t, x in tsr_t["M0"].items()},
                                                                {t: x["tsr"] for t, x in tsr_t[m].items()}, rng)}
        for m in modes[1:]}
    tf["state_check_changes"] = {m: sorted(t for t, x in tsr_t[m].items() if x["strict"] and not x["state"])
                                 for m in modes}
    o["teacher_forced"] = tf

    if m3runs:
        per = []
        for run in m3runs:
            recs = {k: v["result"] for k, v in run.items()}
            to = traj_outcomes(ctx, recs)
            per.append({"SSR_tf": sum(x["success"] for x in recs.values()) / len(recs),
                        "TSR": sum(x["tsr"] for x in to.values()) / len(to)})
        o["M3_repeats"] = {"SSR_tf_runs": [round(x["SSR_tf"], 4) for x in per],
                           "TSR_runs": [round(x["TSR"], 4) for x in per],
                           "TSR_mean": round(st.mean(x["TSR"] for x in per), 4),
                           "TSR_std": round(st.stdev(x["TSR"] for x in per), 4) if len(per) > 1 else None,
                           "api_errors": [sum(bool(v["result"].get("api_error")) for v in run.values())
                                          for run in m3runs]}

    ro, ro_recs = {}, {}
    rp = CONDITIONS[cond]["rollout"]
    if rp.exists():
        rr = read(rp)
        for m in sorted({r["mode"] for r in rr if r["protocol"] == "rollout"}):
            recs = {key(r): r for r in rr if r["protocol"] == "rollout" and r["mode"] == m}
            ro_recs[m] = recs
            fin = {r["traj_id"]: r["final_state_match"] for r in rr
                   if r["protocol"] == "rollout_final" and r["mode"] == m}
            strict = {t: all(recs[(t, i)]["success"] for i in range(ctx.trajs[t].num_steps)) for t in ctx.ids}
            ro_tsr = {t: strict[t] and fin[t] for t in ctx.ids}
            tf_tsr = {t: x["tsr"] for t, x in tsr_t[m].items()}
            ro[m] = {"SSR_rollout": rate(x["success"] for x in recs.values()),
                     "TSR_rollout": rate(ro_tsr.values()),
                     "by_difficulty_TSR_rollout": {k: rate(ro_tsr[t] for t in ctx.ids if ctx.tier[t] == k)
                                                   for k in ("easy", "medium", "hard")},
                     "by_difficulty_SSR_rollout": {k: rate(x["success"] for (t, _), x in recs.items()
                                                           if ctx.tier[t] == k) for k in ("easy", "medium", "hard")},
                     "exec_error_rate": round(st.mean(x["exec_error"] for x in recs.values()), 4),
                     "TSR_flag_agreement_with_tf": f"{sum(ro_tsr[t] == tf_tsr[t] for t in ctx.ids)}/{len(ctx.ids)}",
                     "tf_ok_rollout_fail": sorted(t for t in ctx.ids if tf_tsr[t] and not ro_tsr[t]),
                     "rollout_ok_tf_fail": sorted(t for t in ctx.ids if ro_tsr[t] and not tf_tsr[t]),
                     "SSR_tf_minus_SSR_rollout": round(st.mean(r["results"][m]["success"] for r in rows)
                                                       - st.mean(x["success"] for x in recs.values()), 4)}
        o["rollout"] = ro
    o["clarification"] = clarification(ctx, rows, modes, ro_recs)
    return o, rows, tsr_t


def headline(o):
    """Flat dict of every headline number for the v1 vs v1.1 table."""
    if "teacher_forced" not in o:
        return {}
    h = {}
    tf = o["teacher_forced"]
    for m, v in tf["overall"].items():
        h[f"TSR {m}"] = v["TSR"]["rate"]
        h[f"SSR_tf {m}"] = v["SSR_tf"]["rate"]
        h[f"TSR {m} domain-macro"] = tf["domain_macro_avg"][m]["TSR"]
        for k in ("easy", "medium", "hard"):
            h[f"TSR {m} {k}"] = tf["by_difficulty"][k][m]["TSR"]["rate"]
    for m, v in o.get("rollout", {}).items():
        h[f"TSR_rollout {m}"] = v["TSR_rollout"]["rate"]
        h[f"SSR_rollout {m}"] = v["SSR_rollout"]["rate"]
    if "M3_repeats" in o:
        h["TSR M3 mean of runs"] = o["M3_repeats"]["TSR_mean"]
    for m, v in o["clarification"].items():
        h[f"clarify rate {m} (tf)"] = v["teacher_forced"]["clarify_rate_all_steps"]["rate"]
    return h


def main():
    catalog = load_catalog(DATA / "tool_catalog.json")
    rng = random.Random(SEED)
    out = {"headline_metric": "TSR = all steps strictly correct AND final env state matches gold",
           "m3_in_tables_and_oracle": "M3 run 1 of the condition (runs 2-3 only in M3_repeats)",
           "conditions": {}}
    tsrs = {}
    for cond in CONDITIONS:
        o, rows, tsr_t = analyze(cond, catalog, rng)
        out["conditions"][cond] = o
        tsrs[cond] = tsr_t
        if cond in RESULTS_FILE and rows:
            with open(V6 / RESULTS_FILE[cond], "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # v1 vs v1.1 table + paired tests on the same trajectories
    H = {c: headline(o) for c, o in out["conditions"].items()}
    keys = list(H["v1"]) if H["v1"] else []
    out["v1_vs_v11_table"] = [{"metric": k, **{c: H[c].get(k) for c in CONDITIONS},
                               "delta_v11_minus_v1": (round(H[MAIN][k] - H["v1"][k], 4)
                                                      if H[MAIN].get(k) is not None and H["v1"].get(k) is not None
                                                      else None),
                               "delta_v12_minus_v1": (round(H["v1.2"][k] - H["v1"][k], 4)
                                                      if H["v1.2"].get(k) is not None and H["v1"].get(k) is not None
                                                      else None)} for k in keys]
    pt = {}
    for a, b in (("v1", "v1.1_v4adapter"), ("v1.1_v4adapter", "v1.1"), ("v1", "v1.1"),
                 ("v1.1_v4adapter", "v1.2"), ("v1", "v1.2")):
        if tsrs.get(a) and tsrs.get(b):
            pt[f"{a} -> {b}"] = {m: paired_bootstrap({t: x["tsr"] for t, x in tsrs[a][m].items()},
                                                     {t: x["tsr"] for t, x in tsrs[b][m].items()}, rng)
                                 for m in tsrs[a] if m in tsrs[b]}
    out["paired_TSR_bootstrap_between_conditions"] = pt
    out["step6_m3_variants_v11"] = m3_variants(
        [V6 / "m3/m3_run1_v6.jsonl"] + sorted((V6 / "m3_variants").glob("*_v6.jsonl")), CONDITIONS[MAIN]["eval"])
    out["m3_on_v12"] = m3_variants(sorted((V6 / "m3_v12").glob("*_v12.jsonl")), CONDITIONS["v1.2"]["eval"])
    (V6 / "metrics_v6.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    return out


def m3_variants(files, eval_path):
    """M3 runs (teacher-forced) on one eval file: TSR, SSR_tf, tokens, cost, FLOPs."""
    if not eval_path.exists():
        return {}
    trajs = {t.traj_id: t for t in load_eval(eval_path)}
    res = {}
    for p in files:
        if not p.exists():
            continue
        rs = read(p)
        name = "m3_gptoss_0shot_run1_v6" if p.parent.name == "m3" else p.stem
        if len(rs) < 284:
            res[name] = {"status": f"incomplete {len(rs)}/284"}
            continue
        meta = json.loads(p.with_suffix(".meta.json").read_text()) if p.with_suffix(".meta.json").exists() else {}
        model = meta.get("model")
        recs = {key(r): r["result"] for r in rs}
        by = defaultdict(dict)
        for (t, i), x in recs.items():
            by[t][i] = x
        from eval.v5.common_v5 import state_match
        tsr = {t: all(v[i]["success"] for i in range(len(v)))
               and state_match([v[i]["parsed_call"] for i in range(len(v))], trajs[t].expected_final_state)
               for t, v in by.items()}
        xs = list(recs.values())
        if model in EXTRA_PRICES:
            pi, po = EXTRA_PRICES[model]
            for x in xs:
                if x.get("prompt_tokens") is not None:
                    x["api_cost_usd"] = x["prompt_tokens"] / 1e6 * pi + x["completion_tokens"] / 1e6 * po
        usd = [x["api_cost_usd"] for x in xs if x.get("api_cost_usd") is not None]
        tok = [(x.get("prompt_tokens") or 0) + (x.get("completion_tokens") or 0) for x in xs]
        res[name] = {
            "model": model, "fewshot_demos": meta.get("fewshot_demos"),
            "TSR": rate(tsr.values()), "SSR_tf": rate(x["success"] for x in xs),
            "by_difficulty_TSR": {k: rate(tsr[t] for t in tsr if trajs[t].steps[0].difficulty.value == k)
                                  for k in ("easy", "medium", "hard")},
            "parse_ok_rate": round(st.mean(x["parse_ok"] for x in xs), 4),
            "clarify_steps": sum(is_clarification(x.get("raw_output", ""), x["parse_ok"]) for x in xs),
            "mean_prompt_tokens": round(st.mean(x.get("prompt_tokens") or 0 for x in xs), 1),
            "mean_completion_tokens": round(st.mean(x.get("completion_tokens") or 0 for x in xs), 1),
            "usd_total_284": round(sum(usd), 4) if len(usd) == len(xs) else None,
            "N_active": N_ACTIVE_M3.get(model),
            "flops_mean": float(f"{2 * N_ACTIVE_M3[model] * st.mean(tok):.4g}") if model in N_ACTIVE_M3 else None,
            "api_errors": sum(bool(x.get("api_error")) for x in xs),
        }
    return res


if __name__ == "__main__":
    o = main()
    for r in o["v1_vs_v11_table"]:
        if r["metric"].startswith(("TSR M", "TSR oracle", "SSR_tf", "TSR_rollout", "clarify")) \
                and not r["metric"].endswith(("easy", "medium", "hard", "macro")):
            print(f"{r['metric']:<28}" + "".join(f"{str(r[c]):>16}" for c in CONDITIONS))
