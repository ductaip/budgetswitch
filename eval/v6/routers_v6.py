"""
eval/v6/routers_v6.py — Step 5: cost models, oracle and routers on v1 and v1.1.

Same policies, cost models and code as eval/v5/routers_v5.py (fixed, random,
length, difficulty semi-oracle, oracle x 3 cost models, cascade M0->M1/M3 on
parse/exec error, learned pre-generation router with GroupKFold(5) by trajectory,
cascade+learned), run per condition on teacher-forced step outcomes:
  v1   : eval/v5/pilot_results_v5.jsonl, v1 queries (reproduces routers_v5.json)
  v1.1 : eval/v6/pilot_results_v6.jsonl, v1.1 queries, adapter retrained on v1.1
Learned-router features use the condition's own queries (TF-IDF + e5 embeddings;
v1.1 embeddings cached in eval/v6/cache/e5_query_v11.npz).

Outputs: eval/v6/cost_v6.json, eval/v6/routers_v6.json  ({condition: ...})
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.v5.common_v5 import ROOT, load_catalog, load_eval, state_match  # noqa: E402
from eval.v5.routers_v5 import (  # noqa: E402
    ASSUMED, LAMBDAS, MODES, N_ACTIVE, N_E5, SEED, TAUS, cost_tables, cv_probs, evaluate,
    pareto, pick_eval,
)

V5, V6 = ROOT / "eval/v5", ROOT / "eval/v6"
DATA = ROOT / "data/vitool"
CONDS = {
    "v1": dict(results=V5 / "pilot_results_v5.jsonl", eval_path=DATA / "pilot_eval.jsonl",
               e5_path=V5 / "cache/e5_query.npz", tf_meta=V5 / "raw/local_tf_v5.meta.json",
               m3_paths=[V5 / f"m3/m3_run{k}_v5.jsonl" for k in (1, 2, 3)]),
    "v1.1_v4adapter": dict(results=V6 / "pilot_results_v11_v4adapter.jsonl", eval_path=DATA / "pilot_eval_v1.1.jsonl",
                           e5_path=V6 / "cache/e5_query_v11.npz", tf_meta=V6 / "raw/local_tf_v4adapter_v11.meta.json",
                           m3_paths=[V6 / f"m3/m3_run{k}_v6.jsonl" for k in (1, 2, 3)]),
    "v1.2": dict(results=V6 / "pilot_results_v12.jsonl", eval_path=DATA / "pilot_eval_v1.2.jsonl",
                 e5_path=V6 / "cache/e5_query_v12.npz", tf_meta=V6 / "raw/local_tf_v4adapter_v12.meta.json",
                 m3_paths=[V6 / "m3_v12/m3_gptoss_0shot_run1_v12.jsonl"]),
    "v1.1": dict(results=V6 / "pilot_results_v6.jsonl", eval_path=DATA / "pilot_eval_v1.1.jsonl",
                 e5_path=V6 / "cache/e5_query_v11.npz", tf_meta=V6 / "raw/local_tf_v6adapter_v11.meta.json",
                 m3_paths=[V6 / f"m3/m3_run{k}_v6.jsonl" for k in (1, 2, 3)]),
}


class Data:
    def __init__(self, rows, eval_path):
        self.rows = rows
        self.trajs = {t.traj_id: t for t in load_eval(eval_path)}
        self.groups = [r["traj_id"] for r in self.rows]
        self.n = len(self.rows)
        self.Y = np.array([[r["results"][m]["success"] for m in MODES] for r in self.rows], float)
        self.calls = [[r["results"][m]["parsed_call"] for m in MODES] for r in self.rows]
        self.tidx = {}
        for i, g in enumerate(self.groups):
            self.tidx.setdefault(g, []).append(i)
        for g in self.tidx:
            self.tidx[g].sort(key=lambda i: self.rows[i]["step_idx"])
        self._cache = {}

    def tsr(self, final_mode):
        ok = 0
        for g, idx in self.tidx.items():
            ch = tuple(int(final_mode[i]) for i in idx)
            k = (g, ch)
            if k not in self._cache:
                strict = all(self.Y[i, c] for i, c in zip(idx, ch))
                self._cache[k] = strict and state_match(
                    [self.calls[i][c] for i, c in zip(idx, ch)], self.trajs[g].expected_final_state)
            ok += self._cache[k]
        return ok / len(self.tidx)


def ensure_e5(eval_path, out):
    if out.exists():
        return
    from sentence_transformers import SentenceTransformer
    tr = load_eval(eval_path)
    m = SentenceTransformer("intfloat/multilingual-e5-base", device="cpu")
    texts = [f"query: {t.query}" for t in tr]
    emb = m.encode(texts, normalize_embeddings=True, batch_size=16)
    ntok = [len(m.tokenizer(x)["input_ids"]) for x in texts]
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, traj_id=np.array([t.traj_id for t in tr]), emb=emb, ntok=np.array(ntok),
             nparams=sum(p.numel() for p in m.parameters()))


def run(rows, eval_path, e5_path, tf_meta, m3_paths):
    d = Data(rows, eval_path)
    e5 = np.load(e5_path)
    e5_idx = {t: i for i, t in enumerate(e5["traj_id"])}
    catalog = load_catalog(ROOT / "data/vitool/tool_catalog.json")
    costs, raw = cost_tables(d, e5, e5_idx)
    n, Y, idx = d.n, d.Y, np.arange(d.n)

    # ---------------- cost_v5.json
    hw = {}
    for nm, p in (("local", tf_meta), ("m3", m3_paths[0].with_suffix(".meta.json"))):
        if p.exists():
            hw[nm] = json.loads(p.read_text()).get("hardware")
    cost_out = {"N_active": N_ACTIVE, "N_e5": N_E5,
                "M2_e5_query_tokens_mean": round(float(raw["e5_tokens"].mean()), 1), "modes": {}}
    for j, m in enumerate(MODES):
        cost_out["modes"][m] = {
            "prompt_tokens_mean": round(float(raw["prompt"][:, j].mean()), 1),
            "completion_tokens_mean": round(float(raw["completion"][:, j].mean()), 1),
            "total_tokens_mean": round(float(costs["tokens"][:, j].mean()), 1),
            "flops_mean": float(f"{costs['flops'][:, j].mean():.4g}"),
            "latency_ms_mean": round(float(np.nanmean(raw["latency_ms"][:, j])), 1),
            "latency_ms_median": round(float(np.nanmedian(raw["latency_ms"][:, j])), 1),
            "usd_mean": float(f"{raw['usd'][:, j].mean():.4g}"),
            "usd_total_284": round(float(raw["usd"][:, j].sum()), 4),
            "hardware": hw.get("m3") if m == "M3" else hw.get("local"),
            "relative_to_M0": {cm: round(float(C[:, j].mean() / C[:, 0].mean()), 3)
                               for cm, C in costs.items()},
        }
    results = {"cost": cost_out}

    # ---------------- fixed / heuristic / semi-oracle / oracle
    pol = [pick_eval(d, costs, f"Fixed {m}", "fixed", np.full(n, j)) for j, m in enumerate(MODES)]

    # random: Monte-Carlo (1000 draws) of uniform per-step mode choice
    rng = np.random.default_rng(SEED)
    draws = [rng.integers(0, 4, n) for _ in range(1000)]
    pol.append({"policy": "Random (uniform, 1000 draws)", "group": "heuristic",
                "TSR": round(float(np.mean([d.tsr(p) for p in draws])), 4),
                "SSR_tf": round(float(Y.mean()), 4),
                **{f"cost_rel_M0_{cm}": round(float(C.mean(1).sum() / C[:, 0].sum()), 4)
                   for cm, C in costs.items()}})

    qlen = {t: len(d.trajs[t].query) for t in d.trajs}
    lens = sorted(qlen.values())
    t1, t2 = lens[len(lens) // 3], lens[2 * len(lens) // 3]
    pol.append(pick_eval(d, costs, "Length router (terciles → M0/M1/M2)", "heuristic",
                         np.array([0 if qlen[g] <= t1 else (1 if qlen[g] <= t2 else 2) for g in d.groups]),
                         {"thresholds_chars": [t1, t2]}))
    pol.append(pick_eval(d, costs, "Difficulty escalation (gold tier; semi-oracle)", "semi-oracle",
                         np.array([{"easy": 0, "medium": 1, "hard": 3}[r["difficulty"]] for r in d.rows])))

    oracle_pk = {}
    for cm, C in costs.items():
        pk = np.where(Y.max(1) > 0, np.where(Y > 0, C, np.inf).argmin(1), 0)
        oracle_pk[cm] = pk
        e = pick_eval(d, costs, f"Oracle (cheapest-correct by {cm})", "oracle", pk)
        e["mode_share"] = {m: round(float((pk == j).mean()), 3) for j, m in enumerate(MODES)}
        pol.append(e)

    # ---------------- step 6: execution-feedback cascade
    trig = np.array([(not r["results"]["M0"]["parse_ok"]) or r["results"]["M0"]["exec_error"]
                     for r in d.rows])
    for esc in (1, 3):
        final = np.where(trig, esc, 0)
        pol.append(evaluate(d, costs, f"Cascade M0→{MODES[esc]} on parse/exec error", "cascade", final,
                            {cm: C[:, 0] + trig * C[:, esc] for cm, C in costs.items()},
                            {"trigger_rate": round(float(trig.mean()), 4)}))
    m0_fail = Y[:, 0] == 0
    trigger_stats = {
        "trigger_rate": round(float(trig.mean()), 4),
        "P(M0 wrong | trigger)": round(float(m0_fail[trig].mean()), 4) if trig.any() else None,
        "P(trigger | M0 wrong)": round(float(trig[m0_fail].mean()), 4) if m0_fail.any() else None,
        "P(escalated mode right | trigger)": {MODES[e]: round(float(Y[trig, e].mean()), 4)
                                              for e in (1, 3)} if trig.any() else None,
    }

    # ---------------- step 7: learned routers (GroupKFold by trajectory)
    from sklearn.metrics import roc_auc_score
    learned_auc = {}
    for variant, with_sig in (("pre-generation", False), ("post-M0 (cascade+learned)", True)):
        Pr, folds = cv_probs(d, e5, e5_idx, catalog, with_sig)
        learned_auc[variant] = {m: round(float(roc_auc_score(Y[:, j], Pr[:, j])), 3)
                                if 0 < Y[:, j].mean() < 1 else None for j, m in enumerate(MODES)}
        for cm, C in costs.items():
            # decision costs = fold-train mean cost per mode (per-step cost unknown before generation)
            mc = np.zeros_like(C)
            for f in range(5):
                tr = folds != f
                mc[folds == f] = C[tr].mean(0) / C[tr, 0].mean()
            if not with_sig:
                for lam in LAMBDAS:
                    pol.append(pick_eval(d, costs, f"Learned pre-gen λ={lam} [{cm}]", f"learned-{cm}",
                                         np.argmax(Pr - lam * mc, 1),
                                         {"decision_cost_model": cm, "lambda": lam}))
            else:
                for tau in TAUS:
                    for lam in (0, 0.05, 0.2):
                        alt = 1 + np.argmax(Pr[:, 1:] - lam * mc[:, 1:], 1)
                        esc = Pr[:, 0] < tau
                        pol.append(evaluate(
                            d, costs, f"Cascade+learned τ={tau} λ={lam} [{cm}]", f"cascade-learned-{cm}",
                            np.where(esc, alt, 0),
                            {k: CC[:, 0] + esc * CC[idx, alt] for k, CC in costs.items()},
                            {"decision_cost_model": cm, "tau": tau, "lambda": lam,
                             "escalation_rate": round(float(esc.mean()), 4)}))

    # ---------------- key question: oracle savings vs always-think (M1)
    savings = {}
    for cm, C in costs.items():
        oc = C[idx, oracle_pk[cm]].sum()
        savings[cm] = {
            "oracle_cost_rel_M0": round(float(oc / C[:, 0].sum()), 4),
            "M1_cost_rel_M0": round(float(C[:, 1].sum() / C[:, 0].sum()), 4),
            "oracle_saves_pct_vs_M1": round(float((1 - oc / C[:, 1].sum()) * 100), 1),
            "oracle_saves_pct_vs_M0": round(float((1 - oc / C[:, 0].sum()) * 100), 1),
            "oracle_TSR": round(d.tsr(oracle_pk[cm]), 4), "M1_TSR": round(d.tsr(np.full(n, 1)), 4),
        }

    # sensitivity: oracle with each M3 repeat
    m3_sens = []
    for k, p in enumerate(m3_paths, 1):
        if not p.exists():
            continue
        mk = {(x["traj_id"], x["step_idx"]): x["result"] for x in map(json.loads, open(p))}
        if len(mk) < n:  # repeat still running (Groq quota): skip, reported as missing
            m3_sens.append({"m3_run": k, "status": f"incomplete {len(mk)}/{n}"})
            continue
        saveY, savecalls = d.Y[:, 3].copy(), [c[3] for c in d.calls]
        for i, r in enumerate(d.rows):
            x = mk[(r["traj_id"], r["step_idx"])]
            d.Y[i, 3] = x["success"]
            d.calls[i][3] = x["parsed_call"]
        d._cache = {}
        pk = np.where(d.Y.max(1) > 0, np.where(d.Y > 0, costs["assumed"], np.inf).argmin(1), 0)
        m3_sens.append({"m3_run": k, "oracle_TSR": round(d.tsr(pk), 4),
                        "oracle_SSR_tf": round(float(d.Y.max(1).mean()), 4),
                        "M3_TSR": round(d.tsr(np.full(n, 3)), 4)})
        d.Y[:, 3] = saveY
        for i in range(n):
            d.calls[i][3] = savecalls[i]
        d._cache = {}

    out = {
        "n_steps": n, "cost_models": list(costs),
        "note": "TSR = all steps strictly correct AND final env state matches (teacher-forced "
                "step outcomes composed per trajectory). SSR_tf = teacher-forced step success. "
                "cost_rel_M0_* = policy total cost / always-M0 total cost under that model.",
        "policies": pol,
        "pareto_front_TSR": {cm: pareto(pol, cm) for cm in costs},
        "oracle_savings": savings,
        "oracle_with_each_M3_repeat": m3_sens,
        "cascade_trigger_stats": trigger_stats,
        "learned_router_auc": learned_auc,
    }
    results["routers"] = out
    return results


def main():
    cost, rout = {}, {}
    for name, c in CONDS.items():
        if not c["results"].exists():
            print(f"skip {name}: {c['results'].name} missing")
            continue
        ensure_e5(c["eval_path"], c["e5_path"])
        rows = [json.loads(l) for l in open(c["results"], encoding="utf-8")]
        if not all("M3" in r["results"] for r in rows):
            print(f"skip {name}: M3 missing")
            continue
        random.seed(SEED)
        res = run(rows, c["eval_path"], c["e5_path"], c["tf_meta"], c["m3_paths"])
        cost[name], rout[name] = res["cost"], res["routers"]
        print(name, json.dumps(res["routers"]["oracle_savings"], indent=0))
    (V6 / "cost_v6.json").write_text(json.dumps(cost, indent=2))
    (V6 / "routers_v6.json").write_text(json.dumps(rout, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
