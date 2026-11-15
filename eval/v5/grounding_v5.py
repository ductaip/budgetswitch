"""
eval/v5/grounding_v5.py — grounding audit of gold parameters (all 375 trajectories).

For every gold param value (list elements and nested values checked one by one)
the value is GROUNDED if it can be obtained from:
  query       : appears in the user query after normalization
                (NFC, lowercase, collapsed whitespace; also diacritic-free and
                space-free forms; dates dd/mm/yyyy <-> yyyy-mm-dd; numbers incl.
                "1.000.000", "1,5 triệu", "500k", "2 tỷ")
  prev_obs    : appears in the observation of an earlier step (gold prefix
                replayed through env/mock_tools.py)
  enum        : value is in the tool schema's enum list (or equals a default;
                the catalog currently declares no defaults)
  boolean     : boolean param (yes/no is inferable from wording)
otherwise UNGROUNDED. A step is ungrounded if any gold param value is
ungrounded; a trajectory is ungrounded if any step is.

Limitation: translations/paraphrases (e.g. "Hà Nội" -> "Hanoi", "TP.HCM" ->
"Ho Chi Minh City") are flagged ungrounded unless the diacritic-free form
matches. The saved list is meant for manual review.

Outputs: eval/v5/grounding_v5.json (counts), eval/v5/grounding_ungrounded_v5.jsonl
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.v5.common_v5 import ROOT, gold_prefix, load_catalog, load_eval  # noqa: E402

V5 = ROOT / "eval/v5"
MULT = {"triệu": 1e6, "tr": 1e6, "trieu": 1e6, "k": 1e3, "nghìn": 1e3, "ngàn": 1e3,
        "nghin": 1e3, "ngan": 1e3, "tỷ": 1e9, "tỉ": 1e9, "ty": 1e9, "m": 1e6}


def nfc(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(s)).strip().lower())


def strip_accents(s):
    s = unicodedata.normalize("NFD", s).replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def text_forms(s):
    a = nfc(s)
    b = strip_accents(a)
    return {a, b, a.replace(" ", ""), b.replace(" ", "")}


_NUM_RE = re.compile(r"(?<![\w.,])(\d+(?:[.,]\d+)*)\s*(triệu|trieu|tr|tỷ|tỉ|ty|nghìn|nghin|ngàn|ngan|k|m)?\b")


def query_numbers(q):
    out = set()
    for m in _NUM_RE.finditer(nfc(q)):
        raw, unit = m.group(1), m.group(2)
        cands = set()
        if re.fullmatch(r"\d{1,3}([.,]\d{3})+", raw):
            cands.add(float(re.sub(r"[.,]", "", raw)))
        try:
            cands.add(float(raw.replace(",", ".")))
        except ValueError:
            pass
        for c in cands:
            out.add(c)
            if unit:
                out.add(c * MULT[unit])
    return out


def date_forms(v):
    s = nfc(v)
    m = re.fullmatch(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})", s)
    if m:
        y, mo, d = int(m[1]), int(m[2]), int(m[3])
    else:
        m = re.fullmatch(r"(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})", s)
        if not m:
            return set()
        d, mo, y = int(m[1]), int(m[2]), int(m[3])
    return {f"{d:02d}/{mo:02d}/{y}", f"{d}/{mo}/{y}", f"{y}-{mo:02d}-{d:02d}",
            f"{d:02d}-{mo:02d}-{y}", f"{d}-{mo}-{y}", f"{d:02d}/{mo:02d}", f"{d}/{mo}",
            f"ngày {d} tháng {mo}", f"{d} tháng {mo}"}


def flatten(v):
    if isinstance(v, dict):
        for x in v.values():
            yield from flatten(x)
    elif isinstance(v, (list, tuple)):
        for x in v:
            yield from flatten(x)
    else:
        yield v


def as_number(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str) and re.fullmatch(r"\s*\d+([.,]\d+)?\s*", v):
        return float(v.strip().replace(",", "."))
    return None


def ground_value(v, pschema, query, qforms, qnums, prev_vals, prev_text):
    if isinstance(v, bool):
        return "boolean"
    enum = pschema.get("enum") or (pschema.get("items") or {}).get("enum") or []
    if v in enum or ("default" in pschema and v == pschema["default"]):
        return "enum"
    n = as_number(v)
    if n is not None and n in qnums:
        return "query"
    if isinstance(v, str):
        if any(f and f in q for f in text_forms(v) for q in qforms):
            return "query"
        if any(df in qforms_raw for df in date_forms(v) for qforms_raw in qforms):
            return "query"
    if v in prev_vals or (n is not None and n in {as_number(x) for x in prev_vals}):
        return "prev_obs"
    if isinstance(v, str) and len(v) >= 3 and nfc(v) in prev_text:
        return "prev_obs"
    return None


# Lenient sensitivity: values that are not literally present but are a safe
# inference from the wording. Reported separately; the primary audit is strict.
IMPLICIT_ONE = {"quantity", "passengers", "num_passengers", "guests", "rooms"}
LEXICON = {
    "usd": ["đô", "dollar", "usd", "$"], "vnd": ["việt nam đồng", "tiền việt", "vnđ", "vnd", "đồng"],
    "eur": ["euro", "eur"], "jpy": ["yên", "yen", "jpy"], "krw": ["won", "krw"],
    "cny": ["nhân dân tệ", "tệ", "cny"], "gbp": ["bảng anh", "gbp"], "thb": ["baht", "bạt"],
    "fr": ["pháp"], "en": ["anh", "english"], "vi": ["việt"], "ja": ["nhật"], "ko": ["hàn"],
    "zh": ["trung"], "de": ["đức"], "es": ["tây ban nha"], "th": ["thái"],
    "hà nội": ["hn", "hà nội", "ha noi"], "hn": ["hà nội"],
    "tp.hcm": ["hcm", "sài gòn", "hồ chí minh", "sg"], "hcm": ["tp.hcm", "sài gòn", "hồ chí minh"],
    "hồ chí minh": ["hcm", "sài gòn"], "đà nẵng": ["đn", "da nang"],
}


def lenient_ok(param, value, query):
    if param in IMPLICIT_ONE and as_number(value) == 1.0:
        return "implicit_one"
    q = nfc(query)
    for key in (nfc(value), strip_accents(nfc(value))):
        if key in LEXICON and any(w in q for w in LEXICON[key]):
            return "lexicon"
    return None


def main():
    catalog = load_catalog(ROOT / "data/vitool/tool_catalog.json")
    eval_ids = {t.traj_id for t in load_eval(ROOT / "data/vitool/pilot_eval.jsonl")}
    trajs = load_eval(ROOT / "data/vitool/sft_train.jsonl") + load_eval(ROOT / "data/vitool/pilot_eval.jsonl")
    assert len(trajs) == 375 and len({t.traj_id for t in trajs}) == 375

    rows, how = [], Counter()
    for t in trajs:
        qforms = text_forms(t.query)
        qnums = query_numbers(t.query)
        for i, s in enumerate(t.steps):
            hist, _ = gold_prefix(t, i)
            prev_vals = [x for _, o in hist for x in flatten(o)]
            prev_text = nfc(json.dumps([o for _, o in hist], ensure_ascii=False))
            props = catalog.get(s.expected_tool, {}).get("parameters", {}).get("properties", {})
            bad = []
            for k, val in s.expected_params.items():
                for v in flatten(val) if isinstance(val, (list, dict)) else [val]:
                    src = ground_value(v, props.get(k, {}), t.query, qforms, qnums,
                                       prev_vals, prev_text)
                    how[src or "UNGROUNDED"] += 1
                    if src is None:
                        len_src = lenient_ok(k, v, t.query)
                        how[f"lenient:{len_src}" if len_src else "UNGROUNDED_lenient"] += 1
                        bad.append({"param": k, "value": v, "lenient_ok": len_src})
            rows.append({"traj_id": t.traj_id, "split": "eval" if t.traj_id in eval_ids else "train",
                         "domain": t.domain, "difficulty": s.difficulty.value, "step_idx": i,
                         "tool": s.expected_tool, "query": t.query,
                         "ungrounded": bad, "grounded": not bad,
                         "grounded_lenient": all(b["lenient_ok"] for b in bad)})

    by_traj = defaultdict(list)
    for r in rows:
        by_traj[r["traj_id"]].append(r)
    traj_ok = {k: all(r["grounded"] for r in v) for k, v in by_traj.items()}
    traj_ok_len = {k: all(r["grounded_lenient"] for r in v) for k, v in by_traj.items()}
    meta = {k: (v[0]["split"], v[0]["domain"], v[0]["difficulty"]) for k, v in by_traj.items()}

    def tab(key_fn):
        d = defaultdict(lambda: {"steps": 0, "steps_ungrounded": 0, "steps_ungrounded_lenient": 0,
                                 "trajs": 0, "trajs_ungrounded": 0, "trajs_ungrounded_lenient": 0})
        for r in rows:
            x = d[key_fn(r)]
            x["steps"] += 1
            x["steps_ungrounded"] += not r["grounded"]
            x["steps_ungrounded_lenient"] += not r["grounded_lenient"]
        for k, ok in traj_ok.items():
            sp, dom, tier = meta[k]
            x = d[key_fn({"split": sp, "domain": dom, "difficulty": tier})]
            x["trajs"] += 1
            x["trajs_ungrounded"] += not ok
            x["trajs_ungrounded_lenient"] += not traj_ok_len[k]
        return dict(sorted(d.items()))

    out = {
        "definition": __doc__.strip().split("\n\n")[1],
        "param_value_sources": dict(how),
        "overall": tab(lambda r: "all"),
        "by_split": tab(lambda r: r["split"]),
        "by_domain": tab(lambda r: r["domain"]),
        "by_difficulty": tab(lambda r: r["difficulty"]),
        "eval_by_domain": tab(lambda r: r["domain"] if r["split"] == "eval" else "_train"),
        "eval_by_difficulty": tab(lambda r: r["difficulty"] if r["split"] == "eval" else "_train"),
        "top_ungrounded_params": Counter(f"{r['tool']}.{b['param']}" for r in rows
                                         for b in r["ungrounded"]).most_common(20),
        "top_ungrounded_values": Counter(str(b["value"]) for r in rows
                                         for b in r["ungrounded"]).most_common(20),
    }
    (V5 / "grounding_v5.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    with open(V5 / "grounding_ungrounded_v5.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            if not r["grounded"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(V5 / "cache/grounding_steps_v5.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("traj_id", "step_idx", "split", "grounded", "grounded_lenient")}) + "\n")
    print(json.dumps({k: out[k] for k in ("param_value_sources", "overall", "by_split",
                                           "by_difficulty", "top_ungrounded_params",
                                           "top_ungrounded_values")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
