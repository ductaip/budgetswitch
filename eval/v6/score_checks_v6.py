"""
eval/v6/score_checks_v6.py — agreement for rewrite_check and v11_spotcheck.

Reads eval/v6/{rewrite_check,v11_spotcheck}_annotator{1,2}.json, validates the
filled fields, writes eval/v6/checks_validation.json.
  rewrite_check : % meaning_preserved=y per annotator (+ Wilson CI), raw agreement and
                  Cohen's kappa; natural: mean per annotator, exact agreement,
                  quadratic-weighted kappa
  v11_spotcheck : % trajectories with a fabricated value per annotator, agreement, kappa,
                  union list of reported values (to fix before the camera-ready)
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.v6.score_audit_v6 import kappa, wilson  # noqa: E402

V6 = Path(__file__).resolve().parent


def load(name):
    return [{x["item_id"]: x for x in json.loads((V6 / f"{name}_annotator{a}.json").read_text())["items"]}
            for a in (1, 2)]


def wkappa(a, b, cats):
    k = len(cats)
    idx = {c: i for i, c in enumerate(cats)}
    n = len(a)
    O = [[0] * k for _ in range(k)]
    for x, y in zip(a, b):
        O[idx[x]][idx[y]] += 1
    ra, rb = Counter(a), Counter(b)
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = (i - j) ** 2 / (k - 1) ** 2
            num += w * O[i][j] / n
            den += w * ra[cats[i]] * rb[cats[j]] / n / n
    return round(1 - num / den, 3) if den else 1.0


def main():
    out, problems = {}, []
    A = load("rewrite_check")
    ids = sorted(A[0])
    for a, X in enumerate(A, 1):
        for i in ids:
            if X[i]["meaning_preserved"] not in ("y", "n"):
                problems.append(f"rewrite_check a{a} {i}: meaning_preserved must be 'y'/'n'")
            if X[i]["natural"] not in (1, 2, 3, "1", "2", "3"):
                problems.append(f"rewrite_check a{a} {i}: natural must be 1/2/3")
    S = load("v11_spotcheck")
    sids = sorted(S[0])
    for a, X in enumerate(S, 1):
        for i in sids:
            if X[i]["fabricated_value_found"] not in ("y", "n"):
                problems.append(f"v11_spotcheck a{a} {i}: fabricated_value_found must be 'y'/'n'")
    if problems:
        print("LABEL FILE PROBLEMS — fix and rerun:\n  " + "\n  ".join(problems[:60]))
        raise SystemExit(1)

    mp = [[X[i]["meaning_preserved"] for i in ids] for X in A]
    nat = [[int(X[i]["natural"]) for i in ids] for X in A]
    out["rewrite_check"] = {
        "n": len(ids),
        "meaning_preserved_rate": [{"k": m.count("y"), "rate": round(m.count("y") / len(m), 3),
                                    "ci95": wilson(m.count("y"), len(m))} for m in mp],
        "meaning_agreement": round(sum(x == y for x, y in zip(*mp)) / len(ids), 3),
        "meaning_kappa": kappa(mp[0], mp[1], ["y", "n"]),
        "natural_mean": [round(sum(n) / len(n), 2) for n in nat],
        "natural_distribution": [dict(Counter(n)) for n in nat],
        "natural_exact_agreement": round(sum(x == y for x, y in zip(*nat)) / len(ids), 3),
        "natural_weighted_kappa": wkappa(nat[0], nat[1], [1, 2, 3]),
        "both_say_meaning_not_preserved": [i for i in ids if all(X[i]["meaning_preserved"] == "n" for X in A)],
        "any_says_meaning_not_preserved": [i for i in ids if any(X[i]["meaning_preserved"] == "n" for X in A)],
        "any_natural_1": [i for i in ids if any(int(X[i]["natural"]) == 1 for X in A)],
    }
    fb = [[X[i]["fabricated_value_found"] for i in sids] for X in S]
    out["v11_spotcheck"] = {
        "n": len(sids),
        "fabricated_rate": [{"k": f.count("y"), "rate": round(f.count("y") / len(f), 3),
                             "ci95": wilson(f.count("y"), len(f))} for f in fb],
        "agreement": round(sum(x == y for x, y in zip(*fb)) / len(sids), 3),
        "kappa": kappa(fb[0], fb[1], ["y", "n"]),
        "reported_values": {i: {"traj_id": S[0][i]["traj_id"],
                                "annotator1": S[0][i].get("fabricated_values"),
                                "annotator2": S[1][i].get("fabricated_values")}
                            for i in sids if any(X[i]["fabricated_value_found"] == "y" for X in S)},
    }
    (V6 / "checks_validation.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
