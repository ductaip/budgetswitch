"""
eval/v6/report_tables_v6.py — markdown tables for REPORT_v6.md, generated from
metrics_v6.json, routers_v6.json, cost_v6.json, g3_summary.json, rewrite_stats.json
(so no number in the report is typed by hand). Writes eval/v6/report_tables_v6.md.
"""

from __future__ import annotations

import json
from pathlib import Path

V6 = Path(__file__).resolve().parent
CONDS = ["v1", "v1.1_v4adapter", "v1.1"]


def f(x, nd=3):
    return "–" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def ci(r):
    return f"{r['rate']:.3f} [{r['ci95'][0]:.2f}, {r['ci95'][1]:.2f}]"


def main():
    M = json.loads((V6 / "metrics_v6.json").read_text())
    R = json.loads((V6 / "routers_v6.json").read_text()) if (V6 / "routers_v6.json").exists() else {}
    out = []

    # 1. v1 vs v1.1 headline table
    out.append("### T1. Main table — v1 / v1.1 / v1.2 (all with the v4 adapter; retrained v1.1 for reference)\n")
    out.append("| metric | v1 | v1.1 | **v1.2** | Δ v1.2 − v1 | v1.1, retrained adapter |")
    out.append("|---|---|---|---|---|---|")
    for r in M["v1_vs_v11_table"]:
        d = r.get("delta_v12_minus_v1")
        out.append(f"| {r['metric']} | {f(r['v1'])} | {f(r['v1.1_v4adapter'])} | **{f(r.get('v1.2'))}** | "
                   f"{'' if d is None else f'{d:+.3f}'} | {f(r['v1.1'])} |")

    # 2. main condition with CIs
    for cond in ("v1.2", "v1.1_v4adapter", "v1.1", "v1"):
        o = M["conditions"].get(cond, {})
        if "teacher_forced" not in o:
            continue
        tf = o["teacher_forced"]
        out.append(f"\n### T2-{cond}. TSR (strict + state) and SSR, Wilson 95% CI — {cond}\n")
        out.append("| mode | TSR (tf) | SSR_tf | TSR_rollout | SSR_rollout | TSR domain-macro |")
        out.append("|---|---|---|---|---|---|")
        for m, v in tf["overall"].items():
            ro = o.get("rollout", {}).get(m)
            out.append(f"| {m} | {ci(v['TSR'])} | {ci(v['SSR_tf'])} | "
                       f"{ci(ro['TSR_rollout']) if ro else '–'} | {ci(ro['SSR_rollout']) if ro else '–'} | "
                       f"{tf['domain_macro_avg'][m]['TSR']:.3f} |")
        out.append(f"\nBy tier (TSR, tf) — {cond}\n")
        out.append("| mode | " + " | ".join(f"{k} (n={tf['by_difficulty'][k]['M0']['TSR']['n']})"
                                          for k in ("easy", "medium", "hard")) + " |")
        out.append("|---|---|---|---|")
        for m in tf["overall"]:
            out.append(f"| {m} | " + " | ".join(f"{tf['by_difficulty'][k][m]['TSR']['rate']:.3f}"
                                                for k in ("easy", "medium", "hard")) + " |")
        out.append(f"\nBy domain (TSR, tf) — {cond}\n")
        doms = list(tf["by_domain"])
        out.append("| mode | " + " | ".join(f"{d} (n={tf['by_domain'][d]['M0']['TSR']['n']})" for d in doms) + " |")
        out.append("|---" * (len(doms) + 1) + "|")
        for m in tf["overall"]:
            out.append(f"| {m} | " + " | ".join(f"{tf['by_domain'][d][m]['TSR']['rate']:.3f}" for d in doms) + " |")
        out.append(f"\nPaired tests vs M0 — {cond}\n")
        out.append("| comparison | SSR_tf McNemar (M0 wins / other wins, p) | TSR paired bootstrap diff [95% CI], p |")
        out.append("|---|---|---|")
        for k, v in tf["paired_tests_vs_M0"].items():
            mc, pb = v["SSR_tf_mcnemar"], v["TSR_paired_bootstrap"]
            out.append(f"| {k} | {mc['a_wins']} / {mc['b_wins']}, p={mc['p']:.3g} | "
                       f"{pb['diff_b_minus_a']:+.3f} [{pb['ci95'][0]:+.3f}, {pb['ci95'][1]:+.3f}], p={pb['p_two_sided']:.3g} |")
        if "M3_repeats" in o:
            r3 = o["M3_repeats"]
            out.append(f"\nM3 repeats ({cond}): TSR per run {r3['TSR_runs']}, mean {r3['TSR_mean']}, "
                       f"sd {r3['TSR_std']}; SSR_tf per run {r3['SSR_tf_runs']}; API errors {r3['api_errors']}.")

    # 3. condition-to-condition paired bootstrap
    out.append("\n### T3. Paired TSR bootstrap between conditions (same 112 trajectories)\n")
    out.append("| from → to | mode | Δ TSR [95% CI] | p |")
    out.append("|---|---|---|---|")
    for k, d in M["paired_TSR_bootstrap_between_conditions"].items():
        for m, v in d.items():
            out.append(f"| {k} | {m} | {v['diff_b_minus_a']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] | "
                       f"{v['p_two_sided']:.3g} |")

    # 4. clarification
    out.append("\n### T4. Clarification questions (no call, asks the user), teacher-forced\n")
    out.append("| mode | v1: all steps | v1: on G3-ungrounded steps | v1: on grounded steps | "
               "v1.1: all steps (all grounded → wrong) |")
    out.append("|---|---|---|---|---|")
    c1 = M["conditions"]["v1"]["clarification"]
    c2 = M["conditions"].get("v1.1", {}).get("clarification", {})
    for m in c1:
        a = c1[m]["teacher_forced"]
        b = c2.get(m, {}).get("teacher_forced")
        g = a["clarify_rate_on_grounded_steps (wrong)"]
        u = a["clarify_rate_on_ungrounded_steps (arguably right)"]
        out.append(f"| {m} | {a['clarify_rate_all_steps']['k']}/{a['clarify_rate_all_steps']['n']} | "
                   f"{u['k']}/{u['n']} | {g['k']}/{g['n']} | "
                   f"{(str(b['clarify_rate_all_steps']['k']) + '/' + str(b['clarify_rate_all_steps']['n'])) if b else '–'} |")

    # 5. routers
    for cond, rr in R.items():
        out.append(f"\n### T5-{cond}. Policies (teacher-forced outcomes), cost relative to always-M0\n")
        out.append("| policy | TSR | SSR_tf | assumed | tokens | FLOPs |")
        out.append("|---|---|---|---|---|---|")
        pol = rr["policies"]
        show = [p for p in pol if p["group"] in ("fixed", "heuristic", "semi-oracle", "oracle", "cascade")]
        for cm in ("assumed", "tokens", "flops"):
            lp = [p for p in pol if p["group"] == f"learned-{cm}"]
            if lp:
                best = max(lp, key=lambda p: (p["TSR"], -p[f"cost_rel_M0_{cm}"]))
                show.append(best | {"policy": best["policy"] + " (best TSR)"})
        for p in show:
            out.append(f"| {p['policy']} | {p['TSR']:.3f} | {p['SSR_tf']:.3f} | {p['cost_rel_M0_assumed']:.2f}× | "
                       f"{p['cost_rel_M0_tokens']:.2f}× | {p['cost_rel_M0_flops']:.2f}× |")
        out.append(f"\nOracle savings — {cond}\n")
        out.append("| cost model | oracle cost (×M0) | M1 cost (×M0) | oracle saves vs M1 | oracle TSR | M1 TSR |")
        out.append("|---|---|---|---|---|---|")
        for cm, v in rr["oracle_savings"].items():
            out.append(f"| {cm} | {v['oracle_cost_rel_M0']:.3f} | {v['M1_cost_rel_M0']:.3f} | "
                       f"{v['oracle_saves_pct_vs_M1']:.1f}% | {v['oracle_TSR']:.3f} | {v['M1_TSR']:.3f} |")
        cs = rr["cascade_trigger_stats"]
        out.append(f"\nCascade trigger stats — {cond}: {json.dumps(cs)}")
        out.append(f"\nLearned-router AUC (CV) — {cond}: {json.dumps(rr['learned_router_auc'])}")
        out.append(f"\nOracle with each M3 repeat — {cond}: {json.dumps(rr['oracle_with_each_M3_repeat'])}")
    out.append("\n### T6. Step 6 — M3 variants on v1.1 eval (teacher-forced)\n")
    out.append("| variant | model | TSR | SSR_tf | TSR hard | clarify steps | mean tokens (in/out) | FLOPs/step | USD/284 | API errors |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for k, v in list(M.get("step6_m3_variants_v11", {}).items()) + [(k + " (v1.2 eval)", v) for k, v in M.get("m3_on_v12", {}).items()]:
        if "TSR" not in v:
            out.append(f"| {k} | {v.get('status')} |||||||||")
            continue
        out.append(f"| {k} | {v['model']} | {ci(v['TSR'])} | {ci(v['SSR_tf'])} | {v['by_difficulty_TSR']['hard']['rate']:.3f} | "
                   f"{v['clarify_steps']} | {v['mean_prompt_tokens']:.0f}/{v['mean_completion_tokens']:.0f} | "
                   f"{v['flops_mean']:.3g} | {f(v['usd_total_284'], 4)} | {v['api_errors']} |")
    import sys
    sys.path.insert(0, str(V6.parents[1]))
    from data.grounding_gate import ENUM_PHRASES
    out.append("\n### T7. G3 v1.2 phrase table (Vietnamese wording → enum value)\n")
    out.append("An enum value is grounded only if the query matches one of these patterns (regex, on the "
               "lower-cased query and on its accent-free form), or contains the value itself, or the value "
               "comes from an earlier observation. Source: `data/grounding_gate.py::ENUM_PHRASES`.\n")
    out.append("| tool.param | value | query patterns |")
    out.append("|---|---|---|")
    for tp, d in ENUM_PHRASES.items():
        for v, pats in d.items():
            out.append(f"| {tp} | `{v}` | " + ", ".join(f"`{x}`".replace("|", "\\|") for x in pats) + " |")
    (V6 / "report_tables_v6.md").write_text("\n".join(out) + "\n")
    print("\n".join(out[:40]))


if __name__ == "__main__":
    main()
