"""
eval/v6/rewrite_queries_v6.py — Step 2: repair queries that fail Gate G3 -> ViTool v1.1.

Only the QUERY changes. Gold calls, traj_id, domain, tier, expected_final_state
and split membership are copied unchanged (asserted at the end).

For each trajectory failing G3 (data/grounding_gate.py):
  1. LLM rewrite at temperature 0 (prompt below, logged with its hash): keep the
     intent and all existing content, add only the missing values, natural Vietnamese.
  2. Auto-accept only if every pipeline gate passes on (new query, gold steps):
     structure, resolve, 1 (schema), 1.5 (placeholder), 2 (executable), G3
     -- via data.build_dataset.accept_trajectory -- plus two sanity checks:
       * no id that only exists in an earlier observation (PROD..., FL..., CART...)
         is copied into the query (those must stay "from_prior_observation")
       * >= 70% of the old query's words are kept (content preserved)
  3. Otherwise retry up to 3 times, feeding back the failure reason; then flag
     for manual fixing (query kept as v1 in the output and listed in the report).

Outputs:
  data/vitool/vitool_v1.1.jsonl              all 375 (train then eval), field "split" added
  data/vitool/sft_train_v1.1.jsonl           263, same order as sft_train.jsonl
  data/vitool/pilot_eval_v1.1.jsonl          112, same order as pilot_eval.jsonl
  eval/v6/query_diffs.jsonl                  old/new query + attempts per rewritten trajectory
  eval/v6/rewrite_meta.json                  model, prompt + hash, counts, manual-fix list
Resumable: finished trajectories are cached in eval/v6/cache/rewrite_cache.jsonl.

Usage: venv/bin/python eval/v6/rewrite_queries_v6.py [--model openai/gpt-oss-20b] [--limit N]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.build_dataset import DEP_FIELDS, _ID_KEYS, _harvest_ids, _is_placeholder, accept_trajectory  # noqa: E402
from data.grounding_gate import flatten, grounding_report, nfc  # noqa: E402
from data.vitool.schema import load_trajectories  # noqa: E402
from env.mock_tools import WorldState, run_step  # noqa: E402
from eval.v5.run_m3_v5 import GroqClient, load_key  # noqa: E402

V6 = ROOT / "eval/v6"
DATA = ROOT / "data/vitool"
ORDER = {"easy": 0, "medium": 1, "hard": 2}

SYSTEM = """Bạn là biên tập viên dữ liệu cho trợ lý ảo tiếng Việt biết gọi công cụ.
Bạn nhận một câu yêu cầu của người dùng và danh sách GIÁ TRỊ CÒN THIẾU: đó là những giá trị \
hệ thống bắt buộc phải dùng nhưng người dùng chưa hề nói ra trong câu.
Hãy viết lại câu yêu cầu sao cho người dùng tự nêu đủ các giá trị đó.

Quy tắc:
1. Giữ nguyên ý định và MỌI nội dung đã có trong câu gốc; không bỏ, không đổi thông tin nào.
2. Chỉ thêm các giá trị trong danh sách, không thêm thông tin nào khác.
3. Viết giá trị NGUYÊN VĂN như trong danh sách (tên, mã, địa chỉ, từ khoá giữ đúng chữ). \
Số tiền có thể viết kiểu Việt Nam (500.000đ, 2 triệu); số lượng viết bằng chữ số (2 chiếc, 1 cái).
4. Câu tự nhiên như người Việt gõ cho trợ lý; có thể thành 2-3 câu ngắn nếu cần.
5. Không nhắc tên công cụ, tên tham số tiếng Anh (customer_id, order_id...), JSON hay mã kỹ thuật \
do hệ thống sinh ra. Mã/số người dùng cung cấp thì gọi bằng tiếng Việt, ví dụ "mã đơn hàng DH001", \
"số tài khoản 1234567890", "mã khách hàng KH001".
Chỉ trả về JSON: {"query": "<câu mới>"}"""

PROMPT_HASH = hashlib.sha256(SYSTEM.encode()).hexdigest()[:16]


TARGETS = {"v1.1": {"src": "", "sfx": "", "enum_rule": "v1.1"},
           "v1.2": {"src": "_v1.1", "sfx": "_v12", "enum_rule": "v1.2"}}
ENUM_RULE = "v1.1"
# v1.2 only: enum codes get a Vietnamese wording hint in the missing-value list, so the
# model writes "thanh toán khi nhận hàng" rather than the code "cod". SYSTEM is unchanged.
ENUM_HINT = {"cod": "thanh toán khi nhận hàng", "momo": "ví MoMo", "vnpay": "VNPay",
             "bank_transfer": "chuyển khoản", "shopee": "Shopee", "lazada": "Lazada", "tiki": "Tiki",
             "sendo": "Sendo", "original": "hoàn về phương thức thanh toán ban đầu", "wallet": "hoàn về ví",
             "bank": "hoàn về tài khoản ngân hàng", "economy": "hạng phổ thông", "business": "hạng thương gia",
             "first": "hạng nhất", "residence": "thông tin cư trú", "birth": "khai sinh", "marriage": "kết hôn",
             "death": "khai tử", "water": "tiền nước", "electricity": "tiền điện", "politics": "chính trị",
             "sports": "thể thao", "tech": "công nghệ", "entertainment": "giải trí", "lost": "bị mất",
             "stolen": "bị trộm", "damaged": "bị hỏng", "fraud": "gian lận", "home": "vay mua nhà",
             "car": "vay mua ô tô", "personal": "vay tiêu dùng", "savings": "tiết kiệm", "loan": "vay",
             "credit_card": "thẻ tín dụng", "branch": "chi nhánh", "atm": "ATM"}


def load_catalog():
    raw = json.loads((DATA / "tool_catalog.json").read_text())
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def describe_missing(bad, steps, catalog):
    lines = []
    for b in bad:
        spec = catalog.get(b["tool"], {})
        pdesc = spec.get("parameters", {}).get("properties", {}).get(b["param"], {}).get("description", "")
        what = f"{spec.get('description_vi') or spec.get('description', b['tool'])}"
        extra = f" ({pdesc})" if pdesc else ""
        hint = ""
        if ENUM_RULE == "v1.2" and str(b["value"]) in ENUM_HINT:
            hint = f" — đây là mã lựa chọn, hãy diễn đạt bằng tiếng Việt, vd \"{ENUM_HINT[str(b['value'])]}\""
        lines.append(f"- {b['param']}{extra} cho bước '{what}': {json.dumps(b['value'], ensure_ascii=False)}{hint}")
    plan = "\n".join(
        f"{i + 1}. {catalog.get(t, {}).get('description_vi', t)}: "
        + json.dumps(p, ensure_ascii=False) for i, (t, p) in enumerate(steps))
    return "\n".join(lines), plan


def user_msg(query, missing, plan):
    return (f"Câu gốc:\n{query}\n\nGIÁ TRỊ CÒN THIẾU (phải xuất hiện trong câu mới):\n{missing}\n\n"
            f"Để hiểu ngữ cảnh, đây là các bước hệ thống sẽ thực hiện (KHÔNG chép mã hệ thống sinh ra "
            f"như mã sản phẩm, mã chuyến bay, mã giỏ hàng vào câu):\n{plan}")


def obs_only_ids(query, steps):
    """ids that appear in earlier observations and not in the old query."""
    world, ids = WorldState(), set()
    def harvest(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in _ID_KEYS and isinstance(v, (str, int)):
                    ids.add(str(v))
                else:
                    harvest(v)
        elif isinstance(o, list):
            for x in o:
                harvest(x)
    for t, p in steps:
        try:
            harvest(run_step(t, p, world))
        except Exception:  # noqa: BLE001
            pass
    q = nfc(query)
    return {i for i in ids if nfc(i) not in q}


def words(s):
    return re.findall(r"\w+", nfc(s))


def check(new_q, old_q, steps, tier, catalog, leak_ids, pnames=()):
    ok, _, _, reason = accept_trajectory(steps, catalog, difficulty=tier, query=new_q, enum_rule=ENUM_RULE)
    if not ok:
        bad = grounding_report(new_q, steps, catalog, enum_rule=ENUM_RULE)
        if bad:
            vals = "; ".join(f"{b['param']}={json.dumps(b['value'], ensure_ascii=False)}" for b in bad)
            return False, f"Câu mới vẫn thiếu (hoặc không viết nguyên văn) các giá trị: {vals}"
        return False, f"Không qua cổng kiểm tra: {reason}"
    leaked = [i for i in leak_ids if nfc(i) in nfc(new_q)]
    if leaked:
        return False, f"Câu mới chép mã do hệ thống sinh ra ({', '.join(leaked)}); hãy bỏ các mã này."
    named = [p for p in pnames if p in new_q]
    if named:
        return False, f"Câu mới chứa tên tham số kỹ thuật ({', '.join(named)}); hãy diễn đạt bằng tiếng Việt tự nhiên (vd 'mã khách hàng KH001')."
    ow = words(old_q)
    nw = set(words(new_q))
    kept = sum(w in nw for w in ow) / max(1, len(ow))
    if kept < 0.7:
        return False, f"Câu mới làm mất nội dung câu gốc (chỉ giữ {kept:.0%} số từ); hãy giữ nguyên câu gốc và chỉ chèn thêm."
    return True, f"ok (kept {kept:.0%} words)"


def parse_query(text):
    text = (text or "").strip()
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            q = json.loads(m.group(0)).get("query")
            if isinstance(q, str) and q.strip():
                return q.strip()
        except json.JSONDecodeError:
            pass
    return None


def rewrite_one(client, model, t, steps, tier, catalog, bad, max_retries=3):
    missing, plan = describe_missing(bad, steps, catalog)
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user_msg(t.query, missing, plan)}]
    # ids the query must NOT copy: produced by observations, unless they are a missing value
    leak_ids = obs_only_ids(t.query, steps) - {str(b["value"]) for b in bad}
    pnames = {k for _, p in steps for k in p if "_" in k}
    attempts = []
    for k in range(1 + max_retries):
        kw = dict(model=model, messages=msgs, temperature=0, seed=42, max_tokens=4000)
        if "gpt-oss" in model:
            kw["reasoning_effort"] = "low"
        r, _, n_retry, _ = client.call(**kw)
        text = r.choices[0].message.content or ""
        q = parse_query(text)
        if q is None:
            ok, why = False, "Không đọc được JSON {\"query\": ...}; chỉ trả về đúng JSON."
        else:
            ok, why = check(q, t.query, steps, tier, catalog, leak_ids, pnames)
        attempts.append({"attempt": k + 1, "query": q, "ok": ok, "reason": why, "api_retries": n_retry})
        if ok:
            return q, attempts
        msgs += [{"role": "assistant", "content": text},
                 {"role": "user", "content": f"Chưa đạt: {why}\nHãy sửa lại. Chỉ trả về JSON {{\"query\": ...}}."}]
    return None, attempts


def placeholder_repairs(splits):
    """Gold values that are placeholders (Gate 1.5, e.g. order_id="id_đơn_hàng"):
    if the same id type was produced by an EARLIER step's observation, fill that
    value into the gold call (logged); otherwise the trajectory is excluded from v1.1."""
    fixes, excluded = {}, {}
    for split, trajs in splits:
        for t in trajs:
            produced, world, changes = {}, WorldState(), []
            for i, s in enumerate(t.steps):
                for k, v in s.expected_params.items():
                    if _is_placeholder(v) is None:
                        continue
                    idtype = DEP_FIELDS.get(s.expected_tool, {}).get(k)
                    if idtype and idtype in produced:
                        changes.append({"step": i, "param": k, "old": v, "new": produced[idtype],
                                        "source": f"observation of an earlier step ({idtype})"})
                    else:
                        excluded[t.traj_id] = {"split": split, "step": i, "param": k, "value": v,
                                               "reason": "placeholder gold value not derivable from any "
                                                         "earlier observation (no earlier step produces it)"}
                try:
                    _harvest_ids(run_step(s.expected_tool, dict(s.expected_params), world), produced)
                except Exception:  # noqa: BLE001
                    pass
            if changes and t.traj_id not in excluded:
                fixes[t.traj_id] = changes
    return fixes, excluded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-oss-20b")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--target", default="v1.1", choices=["v1.1", "v1.2"],
                    help="v1.1: from v1 files, G3 enum rule v1.1; v1.2: from v1.1 files, enum must be stated")
    args = ap.parse_args()

    catalog = load_catalog()
    T = TARGETS[args.target]
    global ENUM_RULE
    ENUM_RULE = T["enum_rule"]
    splits = [("train", load_trajectories(str(DATA / f"sft_train{T['src']}.jsonl"))),
              ("eval", load_trajectories(str(DATA / f"pilot_eval{T['src']}.jsonl")))]
    cache_p = V6 / f"cache/rewrite_cache{T['sfx']}.jsonl"
    cache_p.parent.mkdir(parents=True, exist_ok=True)
    cache = {}
    if cache_p.exists():
        for line in cache_p.open():
            d = json.loads(line)
            if d["model"] == args.model and d["prompt_hash"] == PROMPT_HASH:
                cache[d["traj_id"]] = d
    client = GroqClient(load_key())

    todo = []
    for split, trajs in splits:
        for t in trajs:
            steps = [(s.expected_tool, s.expected_params) for s in t.steps]
            bad = grounding_report(t.query, steps, catalog, enum_rule=ENUM_RULE)
            if bad:
                todo.append((split, t, steps, bad))
    print(f"{len(todo)} trajectories fail G3; {len(cache)} cached", flush=True)

    done = 0
    for split, t, steps, bad in todo:
        if t.traj_id in cache:
            continue
        if args.limit is not None and done >= args.limit:
            break
        tier = max((s.difficulty.value for s in t.steps), key=lambda d: ORDER[d])
        new_q, attempts = rewrite_one(client, args.model, t, steps, tier, catalog, bad)
        rec = {"traj_id": t.traj_id, "split": split, "domain": t.domain, "difficulty": tier,
               "model": args.model, "prompt_hash": PROMPT_HASH, "old_query": t.query,
               "new_query": new_q, "status": "accepted" if new_q else "manual_fix_needed",
               "n_attempts": len(attempts), "missing_values": bad, "attempts": attempts}
        with cache_p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        cache[t.traj_id] = rec
        done += 1
        print(f"[{len(cache)}/{len(todo)}] {t.traj_id}: {rec['status']} after {len(attempts)}", flush=True)

    if args.limit is not None:
        return
    # ---- assemble v1.1 (only query changes) ----
    manual_p = V6 / "manual_fixes.json"
    manual = json.loads(manual_p.read_text()) if manual_p.exists() else {}
    fixes, excluded = placeholder_repairs(splits)
    assert not fixes, "gold placeholder fill-ins need final-state recomputation; not implemented"
    out_all, diffs = [], []
    for split, trajs in splits:
        rows = []
        for t in trajs:
            if t.traj_id in excluded:
                continue
            d = t.to_dict()
            rec = cache.get(t.traj_id)
            if t.traj_id in manual:
                d["query"] = manual[t.traj_id]
            elif rec and rec["new_query"]:
                d["query"] = rec["new_query"]
            rows.append(d)
            if rec and t.traj_id not in excluded:
                diffs.append({k: rec[k] for k in ("traj_id", "split", "domain", "difficulty", "status",
                                                  "n_attempts", "old_query")}
                             | {"new_query": d["query"] if d["query"] != t.query else None,
                                "manual_fix": t.traj_id in manual,
                                "missing_values": rec["missing_values"]})
        name = "sft_train" if split == "train" else "pilot_eval"
        with open(DATA / f"{name}_{args.target}.jsonl", "w", encoding="utf-8") as f:
            for d in rows:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        out_all += [d | {"split": split} for d in rows]
    with open(DATA / f"vitool_{args.target}.jsonl", "w", encoding="utf-8") as f:
        for d in out_all:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(V6 / f"query_diffs{T['sfx']}.jsonl", "w", encoding="utf-8") as f:
        for d in diffs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # invariants: nothing but the query changed
    old = {t.traj_id: (s, t.to_dict()) for s, ts in splits for t in ts if t.traj_id not in excluded}
    assert len(out_all) == len(old) == sum(len(ts) for _, ts in splits) - len(excluded)
    for d in out_all:
        s, o = old[d["traj_id"]]
        assert d["split"] == s
        assert {k: v for k, v in d.items() if k not in ("query", "split")} == \
               {k: v for k, v in o.items() if k != "query"}, d["traj_id"]
    still = [d["traj_id"] for d in out_all
             if grounding_report(d["query"], [(s["expected_tool"], s["expected_params"]) for s in d["steps"]], catalog,
                                 enum_rule=ENUM_RULE)]
    meta = {"model": args.model, "endpoint": "groq", "temperature": 0, "seed": 42,
            "reasoning_effort": "low" if "gpt-oss" in args.model else None,
            "prompt_hash": PROMPT_HASH, "system_prompt": SYSTEM,
            "user_prompt_template": user_msg("{query}", "{missing}", "{plan}"),
            "max_retries": 3, "n_failing_g3_v1": len(todo),
            "accepted_auto": sum(1 for r in cache.values() if r["status"] == "accepted"),
            "accepted_by_attempt": {str(k): sum(1 for r in cache.values()
                                                if r["status"] == "accepted" and r["n_attempts"] == k)
                                    for k in range(1, 5)},
            "manual_fix_needed": sorted(r["traj_id"] for r in cache.values()
                                        if r["status"] != "accepted"),
            "manual_fixes_applied": sorted(manual),
            "gold_placeholder_fills": fixes, "excluded_from_v1.1": excluded,
            "n_v1.1": {"all": len(out_all), "train": sum(d["split"] == "train" for d in out_all),
                       "eval": sum(d["split"] == "eval" for d in out_all)},
            "still_failing_g3_in_v1.1": still}
    meta["target"], meta["enum_rule"], meta["source_files"] = args.target, ENUM_RULE, T["src"] or "v1"
    meta["enum_hints_in_user_message"] = args.target == "v1.2"
    (V6 / f"rewrite_meta{T['sfx']}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in meta.items() if "prompt" not in k}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
