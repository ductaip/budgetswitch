"""
eval/v5/relaxed_v5.py — step 4: relaxed / execution-equivalence success.

A call is RELAXED-correct if either
  (a) normalized match: same tool (case-insensitive) and every gold param is
      present and equal after normalization (whitespace, case, Unicode NFC,
      dates dd/mm/yyyy | d-m-yyyy | yyyy-mm-dd -> ISO, numeric strings -> numbers,
      thousands separators, lists compared as sorted multisets). Extra params
      are allowed (same as strict). The JSON-schema check is NOT applied.
  (b) execution equivalence: on the world after the gold prefix, the predicted
      call and the gold call both execute without error, call the same tool,
      and produce the same observation AND the same mutation-state delta.
      (Deviation from the brief: delta alone is empty for every read-only tool,
      so any executable read-only call would pass; observation equality is
      added to keep the criterion meaningful.)

Inputs: eval/v5/pilot_results_v5.jsonl (+ m3 runs for M3 mean)
Outputs: eval/v5/relaxed_v5.json, eval/v5/relaxed_examples_v5.jsonl (20 random,
         seed 42, strict-fail & relaxed-pass, for manual checking)
"""

from __future__ import annotations

import json
import random
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.v5.common_v5 import ROOT, execute, gold_prefix, load_catalog, load_eval  # noqa: E402

V5 = ROOT / "eval/v5"
SEED = 42

_DATE_DMY = re.compile(r"^(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})$")
_DATE_YMD = re.compile(r"^(\d{4})[/\-.](\d{1,2})[/\-.](\d{1,2})$")
_NUM = re.compile(r"^[+-]?\d+([.,]\d+)?$")
_THOUSANDS = re.compile(r"^\d{1,3}([.,]\d{3})+$")


def norm(v):
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = unicodedata.normalize("NFC", v).strip().lower()
        s = re.sub(r"\s+", " ", s)
        m = _DATE_DMY.match(s)
        if m:
            return f"{int(m[3]):04d}-{int(m[2]):02d}-{int(m[1]):02d}"
        m = _DATE_YMD.match(s)
        if m:
            return f"{int(m[1]):04d}-{int(m[2]):02d}-{int(m[3]):02d}"
        if _THOUSANDS.match(s):
            return float(re.sub(r"[.,]", "", s))
        if _NUM.match(s):
            return float(s.replace(",", "."))
        return s
    if isinstance(v, (list, tuple)):
        return tuple(sorted((norm(x) for x in v), key=lambda x: json.dumps(x, default=str)))
    if isinstance(v, dict):
        return tuple(sorted((str(k).lower(), norm(x)) for k, x in v.items()))
    return v


def normalized_match(call, step) -> bool:
    name = call.get("tool_name")
    if not isinstance(name, str) or name.strip().lower() != step.expected_tool.lower():
        return False
    p = call.get("params") or {}
    if not isinstance(p, dict):
        return False
    return all(k in p and norm(p[k]) == norm(v) for k, v in step.expected_params.items())


def exec_equivalent(call, step, world) -> bool:
    gold = {"tool_name": step.expected_tool, "params": step.expected_params}
    e1, _, o1, w1 = execute(call, world)
    e2, _, o2, w2 = execute(gold, world)
    if e1 or e2 or call.get("tool_name") != step.expected_tool:
        return False
    return o1 == o2 and w1.snapshot() == w2.snapshot()


def main():
    catalog = load_catalog(ROOT / "data/vitool/tool_catalog.json")  # noqa: F841
    trajs = {t.traj_id: t for t in load_eval(ROOT / "data/vitool/pilot_eval.jsonl")}
    rows = [json.loads(l) for l in open(V5 / "pilot_results_v5.jsonl", encoding="utf-8")]
    modes = [m for m in ("M0", "M1", "M2", "M3") if all(m in r["results"] for r in rows)]

    per = []  # (row, mode, strict, norm, exec, relaxed)
    for r in rows:
        t = trajs[r["traj_id"]]
        step = t.steps[r["step_idx"]]
        _, world = gold_prefix(t, r["step_idx"])
        for m in modes:
            x = r["results"][m]
            call = x["parsed_call"]
            nm = normalized_match(call, step)
            ee = exec_equivalent(call, step, world)
            x["relaxed_norm"], x["relaxed_exec"] = nm, ee
            x["relaxed_success"] = bool(x["success"] or nm or ee)
            per.append((r, m, x["success"], nm, ee, x["relaxed_success"]))

    def tsr(mode, field):
        t = defaultdict(list)
        for r in rows:
            t[r["traj_id"]].append(bool(r["results"][mode][field]))
        return sum(all(v) for v in t.values()) / len(t)

    out = {"definition": __doc__.strip().split("\n\n")[1], "modes": {}}
    for m in modes:
        xs = [r["results"][m] for r in rows]
        out["modes"][m] = {
            "SSR_strict": round(sum(x["success"] for x in xs) / len(xs), 4),
            "SSR_relaxed": round(sum(x["relaxed_success"] for x in xs) / len(xs), 4),
            "TSR_strict": round(tsr(m, "success"), 4),
            "TSR_relaxed": round(tsr(m, "relaxed_success"), 4),
            "strict_fail_relaxed_pass": sum(x["relaxed_success"] and not x["success"] for x in xs),
            "via_norm_only": sum(x["relaxed_norm"] and not x["relaxed_exec"] and not x["success"] for x in xs),
            "via_exec_only": sum(x["relaxed_exec"] and not x["relaxed_norm"] and not x["success"] for x in xs),
            "relaxed_fail_but_strict_pass": sum(x["success"] and not (x["relaxed_norm"] or x["relaxed_exec"]) for x in xs),
        }
    strict_unsolv = [r for r in rows if not any(r["results"][m]["success"] for m in modes)]
    out["unsolvable_strict"] = len(strict_unsolv)
    out["unsolvable_strict_now_solvable_relaxed"] = sum(
        any(r["results"][m]["relaxed_success"] for m in modes) for r in strict_unsolv)
    out["oracle"] = {
        "SSR_strict": round(sum(any(r["results"][m]["success"] for m in modes) for r in rows) / len(rows), 4),
        "SSR_relaxed": round(sum(any(r["results"][m]["relaxed_success"] for m in modes) for r in rows) / len(rows), 4),
    }
    (V5 / "relaxed_v5.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    ex_path = V5 / "relaxed_examples_v5.jsonl"
    if ex_path.exists() and any(json.loads(l).get("manual_verdict") for l in open(ex_path, encoding="utf-8")):
        print("relaxed_examples_v5.jsonl has manual verdicts — not overwritten")
        print(json.dumps(out, indent=1, ensure_ascii=False))
        return rows
    cand = [(r, m) for r, m, s, nm, ee, rx in per if rx and not s]
    rng = random.Random(SEED)
    pick = rng.sample(cand, min(20, len(cand)))
    with open(ex_path, "w", encoding="utf-8") as f:
        for r, m in pick:
            st = trajs[r["traj_id"]].steps[r["step_idx"]]
            x = r["results"][m]
            f.write(json.dumps({
                "traj_id": r["traj_id"], "step_idx": r["step_idx"], "mode": m,
                "query": trajs[r["traj_id"]].query,
                "gold": {"tool_name": st.expected_tool, "params": st.expected_params,
                         "fuzzy_keys": st.fuzzy_keys},
                "pred": x["parsed_call"], "relaxed_norm": x["relaxed_norm"],
                "relaxed_exec": x["relaxed_exec"], "manual_verdict": None,
            }, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return rows


if __name__ == "__main__":
    main()
