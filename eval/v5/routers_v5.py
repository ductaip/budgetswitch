"""
eval/v5/routers_v5.py — steps 5, 6, 7: measured cost models, oracle, routers,
execution-feedback cascade, cross-validated learned routers.

Headline metric: TSR = every step strictly correct AND executing the chosen
calls in order reproduces the gold final env state. SSR_tf = teacher-forced
step success. Policies are evaluated on teacher-forced step outcomes (the only
protocol where all four modes were run on every step).

Cost models (per step, per mode; policies reported relative to always-M0):
  assumed : 1 / 3 / 5 / 15
  tokens  : prompt_tokens + completion_tokens actually used at that step
  flops   : 2 * N_active * (prompt + completion) [+ 2 * N_e5 * e5_query_tokens for M2]
            N_active: Qwen2.5-3B = 3.09e9; gpt-oss-120b = 5.1e9 (model card:
            "117B parameters with 5.1B active parameters"); e5-base = 278e6.

Outputs: eval/v5/cost_v5.json, eval/v5/routers_v5.json
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.v5.common_v5 import ROOT, load_catalog, load_eval, state_match  # noqa: E402

V5 = ROOT / "eval/v5"
MODES = ["M0", "M1", "M2", "M3"]
N_ACTIVE = {"M0": 3.09e9, "M1": 3.09e9, "M2": 3.09e9, "M3": 5.1e9}
N_E5 = 278_043_648
ASSUMED = np.array([1.0, 3.0, 5.0, 15.0])
SEED = 42
LAMBDAS = [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
TAUS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


class Data:
    def __init__(self):
        self.rows = [json.loads(l) for l in open(V5 / "pilot_results_v5.jsonl", encoding="utf-8")]
        self.trajs = {t.traj_id: t for t in load_eval(ROOT / "data/vitool/pilot_eval.jsonl")}
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
        """final_mode[i] = index of the mode whose call is the executed one at step i."""
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


def cost_tables(d, e5, e5_idx):
    P = np.zeros((d.n, 4)); Cc = np.zeros((d.n, 4)); usd = np.zeros((d.n, 4))
    lat = np.full((d.n, 4), np.nan)
    for i, r in enumerate(d.rows):
        for j, m in enumerate(MODES):
            x = r["results"][m]
            P[i, j] = x.get("prompt_tokens") or 0
            Cc[i, j] = x.get("completion_tokens") or 0
            usd[i, j] = x.get("api_cost_usd") or 0.0
            if x.get("latency_ms") is not None:
                lat[i, j] = x["latency_ms"]
    tok = P + Cc
    fl = 2 * np.array([N_ACTIVE[m] for m in MODES])[None, :] * tok
    e5tok = np.array([e5["ntok"][e5_idx[r["traj_id"]]] for r in d.rows])
    fl[:, 2] += 2 * N_E5 * e5tok
    return ({"assumed": np.tile(ASSUMED, (d.n, 1)), "tokens": tok, "flops": fl},
            {"prompt": P, "completion": Cc, "usd": usd, "latency_ms": lat, "e5_tokens": e5tok})


def evaluate(d, costs, name, group, final_mode, cost_vecs, extra=None):
    idx = np.arange(d.n)
    out = {"policy": name, "group": group,
           "TSR": round(d.tsr(final_mode), 4),
           "SSR_tf": round(float(d.Y[idx, final_mode].mean()), 4)}
    for cm, C in costs.items():
        out[f"cost_rel_M0_{cm}"] = round(float(cost_vecs[cm].sum() / C[:, 0].sum()), 4)
    if extra:
        out.update(extra)
    return out


def pick_eval(d, costs, name, group, pick, extra=None):
    idx = np.arange(d.n)
    return evaluate(d, costs, name, group, pick, {cm: C[idx, pick] for cm, C in costs.items()}, extra)


def features(d, e5, e5_idx, catalog, train_idx, with_m0_signals):
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    q = [d.trajs[r["traj_id"]].query for r in d.rows]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2,
                          sublinear_tf=True).fit([q[i] for i in train_idx])
    tools = sorted(catalog)
    tix = {t: k for k, t in enumerate(tools)}
    prev = {(r["traj_id"], r["step_idx"]): r["results"]["M0"]["parsed_call"].get("tool_name")
            for r in d.rows}
    dense = []
    for r in d.rows:
        oh = np.zeros(len(tools) + 1)
        if r["step_idx"] > 0:
            p = prev.get((r["traj_id"], r["step_idx"] - 1))
            oh[tix[p] if p in tix else -1] = 1
        num = [r["step_idx"], r["step_idx"] == 0]
        if with_m0_signals:
            x0 = r["results"]["M0"]
            num += [x0["parse_ok"], x0["exec_error"]]
        dense.append(np.concatenate([e5["emb"][e5_idx[r["traj_id"]]], oh, num]))
    return hstack([vec.transform(q), csr_matrix(np.array(dense, float))]).tocsr()


def cv_probs(d, e5, e5_idx, catalog, with_m0_signals):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    Pr = np.zeros_like(d.Y)
    folds = np.zeros(d.n, int)
    for f, (tr, te) in enumerate(GroupKFold(5).split(d.Y, groups=d.groups)):
        folds[te] = f
        X = features(d, e5, e5_idx, catalog, tr, with_m0_signals)
        for j in range(4):
            y = d.Y[tr, j]
            if y.min() == y.max():
                Pr[te, j] = y[0]
                continue
            clf = LogisticRegression(C=1.0, max_iter=5000, random_state=SEED)
            Pr[te, j] = clf.fit(X[tr], y).predict_proba(X[te])[:, 1]
    return Pr, folds


def pareto(points, cm, metric="TSR"):
    pts = sorted(points, key=lambda p: (p[f"cost_rel_M0_{cm}"], -p[metric]))
    front, best = [], -1.0
    for p in pts:
        if p[metric] > best:
            front.append({"policy": p["policy"], "cost_rel_M0": p[f"cost_rel_M0_{cm}"],
                          "TSR": p["TSR"], "SSR_tf": p["SSR_tf"]})
            best = p[metric]
    return front


def main():
    d = Data()
    e5 = np.load(V5 / "cache/e5_query.npz")
    e5_idx = {t: i for i, t in enumerate(e5["traj_id"])}
    catalog = load_catalog(ROOT / "data/vitool/tool_catalog.json")
    costs, raw = cost_tables(d, e5, e5_idx)
    n, Y, idx = d.n, d.Y, np.arange(d.n)

    # ---------------- cost_v5.json
    hw = {}
    for p in (V5 / "raw/local_tf_v5.meta.json", V5 / "m3/m3_run1_v5.meta.json"):
        if p.exists():
            hw[p.name] = json.loads(p.read_text()).get("hardware")
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
            "hardware": hw.get("m3_run1_v5.meta.json") if m == "M3" else hw.get("local_tf_v5.meta.json"),
            "relative_to_M0": {cm: round(float(C[:, j].mean() / C[:, 0].mean()), 3)
                               for cm, C in costs.items()},
        }
    (V5 / "cost_v5.json").write_text(json.dumps(cost_out, indent=2))

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
    for k in (1, 2, 3):
        p = V5 / f"m3/m3_run{k}_v5.jsonl"
        if not p.exists():
            continue
        mk = {(x["traj_id"], x["step_idx"]): x["result"] for x in map(json.loads, open(p))}
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
    (V5 / "routers_v5.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    for p in pol:
        if p["group"] in ("fixed", "heuristic", "semi-oracle", "oracle", "cascade"):
            print(f"{p['policy']:<50} TSR {p['TSR']:.3f} SSR_tf {p['SSR_tf']:.3f}  "
                  + "  ".join(f"{cm}:{p[f'cost_rel_M0_{cm}']:.2f}x" for cm in costs))
    print(json.dumps(savings, indent=1))
    print(json.dumps(trigger_stats, indent=1))
    print(json.dumps(learned_auc, indent=1))


if __name__ == "__main__":
    random.seed(SEED)
    main()
