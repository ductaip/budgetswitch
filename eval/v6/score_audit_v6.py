"""
eval/v6/score_audit_v6.py — Step 0: score the grounding audit against two annotators.

Usage:
  python eval/v6/score_audit_v6.py \
      --a1 eval/v6/audit_sample_annotator1.json --a2 eval/v6/audit_sample_annotator2.json \
      [--adjudicated eval/v6/audit_sample_adjudicated.json]
(--a1/--a2 may also point to audit_sample.json or audit_sample.csv filled with
label_annotator_1 / label_annotator_2.)

Reports (eval/v6/audit_validation.json):
  - label validity check; raw agreement; Cohen's kappa (4 labels, and binary
    ungrounded vs grounded)
  - audit precision = P(truly ungrounded | audit flags it), Wilson 95% CI, for the
    strict and lenient audit. Truth = adjudicated label if given, else items where
    the annotators agree (disagreements listed for adjudication), plus each
    annotator alone.
  - audit recall, estimated with stratum weights (flagged vs unflagged
    population sizes; the unflagged stratum has only 20 items → wide CI)
  - truly-ungrounded values by param type; enum_justified stats
  - GATE: proceed only if precision >= 0.9
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parent
LABELS = ["inferable_from_query", "from_prior_observation", "tool_default_or_enum", "ungrounded"]


def wilson(k, n, z=1.959964):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def kappa(a, b, cats):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[c] * cb[c] for c in cats) / (n * n)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


def load_labels(path, which):
    """-> {item_id: (label, enum_justified, note)}"""
    p = Path(path)
    if p.suffix == ".csv":
        rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
        return {r["item_id"]: ((r.get(f"label_annotator_{which}") or "").strip() or None,
                               (r.get(f"enum_justified_annotator_{which}") or "").strip() or None,
                               r.get(f"note_annotator_{which}", "")) for r in rows}
    d = json.loads(p.read_text())
    out = {}
    for it in d["items"]:
        if "label" in it:
            out[it["item_id"]] = (it["label"], it.get("enum_justified"), it.get("note", ""))
        else:
            out[it["item_id"]] = (it.get(f"label_annotator_{which}"), it.get(f"enum_justified_annotator_{which}"),
                                  it.get(f"note_annotator_{which}", ""))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a1", default=str(OUT / "audit_sample_annotator1.json"))
    ap.add_argument("--a2", default=str(OUT / "audit_sample_annotator2.json"))
    ap.add_argument("--adjudicated", default=None,
                    help="optional file with a 'label' per item resolving disagreements")
    ap.add_argument("--out", default=str(OUT / "audit_validation.json"))
    args = ap.parse_args()

    key = json.loads((OUT / "audit_sample_key.json").read_text())
    pop = key["population"]
    K = {x["item_id"]: x for x in key["items"]}
    A1, A2 = load_labels(args.a1, 1), load_labels(args.a2, 2)

    problems = []
    for name, A in (("annotator1", A1), ("annotator2", A2)):
        for iid in K:
            lab, ej, _ = A.get(iid, (None, None, ""))
            if lab not in LABELS:
                problems.append(f"{name} {iid}: label {lab!r} not one of {LABELS}")
            if lab == "tool_default_or_enum" and ej not in ("y", "n"):
                problems.append(f"{name} {iid}: enum_justified must be 'y' or 'n'")
    if problems:
        print("LABEL FILE PROBLEMS — fix and rerun:\n  " + "\n  ".join(problems[:50]))
        raise SystemExit(1)

    ids = sorted(K)
    l1 = [A1[i][0] for i in ids]
    l2 = [A2[i][0] for i in ids]
    b1 = [x == "ungrounded" for x in l1]
    b2 = [x == "ungrounded" for x in l2]
    adj = load_labels(args.adjudicated, 1) if args.adjudicated else {}
    disagree = [i for i, x, y in zip(ids, l1, l2) if x != y]
    bin_disagree = [i for i, x, y in zip(ids, b1, b2) if x != y]

    def truth_sets():
        sets = {"annotator1": {i: A1[i][0] == "ungrounded" for i in ids},
                "annotator2": {i: A2[i][0] == "ungrounded" for i in ids},
                "agreed_items_only": {i: A1[i][0] == "ungrounded" for i in ids
                                      if (A1[i][0] == "ungrounded") == (A2[i][0] == "ungrounded")}}
        if adj:
            sets["adjudicated"] = {i: (adj[i][0] if i in adj and adj[i][0] else A1[i][0]) == "ungrounded"
                                   for i in ids}
        return sets

    res = {}
    for tname, T in truth_sets().items():
        r = {}
        for audit in ("strict", "lenient"):
            flag = {i: K[i][f"audit_flag_{audit}"] for i in T}
            fl = [i for i in T if flag[i]]
            tp = sum(T[i] for i in fl)
            # recall via stratum weights (sample was stratified on the STRICT flag)
            fs = [i for i in T if K[i]["stratum"] == "flagged"]
            us = [i for i in T if K[i]["stratum"] == "unflagged"]
            p_f = sum(T[i] for i in fs) / len(fs) if fs else 0
            p_u = sum(T[i] for i in us) / len(us) if us else 0
            if audit == "strict":
                est_tp, est_fn = pop["flagged_values"] * p_f, pop["unflagged_values"] * p_u
            else:  # lenient flags are a subset of strict flags
                fl_l = [i for i in fs if flag[i]]
                share_l = len(fl_l) / len(fs) if fs else 0
                p_fl = sum(T[i] for i in fl_l) / len(fl_l) if fl_l else 0
                p_fnl = (sum(T[i] for i in fs if not flag[i]) / (len(fs) - len(fl_l))) if len(fs) > len(fl_l) else 0
                est_tp = pop["flagged_values"] * share_l * p_fl
                est_fn = pop["flagged_values"] * (1 - share_l) * p_fnl + pop["unflagged_values"] * p_u
            r[audit] = {
                "precision": round(tp / len(fl), 3) if fl else None, "precision_ci95": wilson(tp, len(fl)),
                "n_flagged_items": len(fl), "true_ungrounded_among_flagged": tp,
                "ungrounded_rate_in_unflagged_sample": f"{sum(T[i] for i in us)}/{len(us)}",
                "recall_estimate_weighted": round(est_tp / (est_tp + est_fn), 3) if est_tp + est_fn else None,
                "gate_precision_>=_0.9": (tp / len(fl) >= 0.9) if fl else None,
            }
        res[tname] = r

    truth = truth_sets().get("adjudicated") or truth_sets()["agreed_items_only"]
    by_type = Counter(K[i]["param_type"] for i, v in truth.items() if v)
    fp_types = Counter(K[i]["param_type"] for i, v in truth.items() if K[i]["audit_flag_strict"] and not v)
    ej = Counter((a, A[i][1]) for a, A in (("a1", A1), ("a2", A2)) for i in ids
                 if A[i][0] == "tool_default_or_enum")
    out = {
        "n_items": len(ids),
        "agreement_4label": round(1 - len(disagree) / len(ids), 3),
        "kappa_4label": kappa(l1, l2, LABELS),
        "agreement_binary": round(1 - len(bin_disagree) / len(ids), 3),
        "kappa_binary_ungrounded": kappa(b1, b2, [True, False]),
        "label_distribution": {"annotator1": Counter(l1), "annotator2": Counter(l2)},
        "items_to_adjudicate_binary": bin_disagree,
        "items_disagreeing_4label": disagree,
        "audit_vs_truth": res,
        "truly_ungrounded_by_param_type": dict(by_type.most_common()),
        "audit_false_positives_by_param_type": dict(fp_types.most_common()),
        "enum_justified_counts": {f"{a}:{v}": c for (a, v), c in ej.items()},
    }
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False, default=dict))
    print(json.dumps(out, indent=1, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    main()
