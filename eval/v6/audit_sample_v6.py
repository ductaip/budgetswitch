"""
eval/v6/audit_sample_v6.py — Step 0: build the manual-validation sample for the
v5 grounding audit (eval/v5/grounding_v5.py).

Unit = one gold parameter value (list elements separately), same unit as the audit.
Sample (seed 42): 40 values the strict audit flags as ungrounded, stratified by
tier x domain (proportional, largest remainder, >= 1 per non-empty cell), plus
20 unflagged values stratified by tier. Shuffled; item ids A01..A60.

Outputs (eval/v6/):
  audit_sample.json              all items + instructions; label slots for both annotators
  audit_sample_annotator1.json   same items, one label slot (label independently!)
  audit_sample_annotator2.json
  audit_sample.csv               same content, spreadsheet form
  audit_sample_key.json          audit decision per item — annotators must NOT open
  audit_value_frequency.json     frequency of each ungrounded value across splits
  ANNOTATION_GUIDE.md            labeling instructions (also embedded in the JSON)
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.v5.common_v5 import ROOT, gold_prefix, load_catalog, load_eval  # noqa: E402
from eval.v5.grounding_v5 import (  # noqa: E402
    flatten, ground_value, lenient_ok, nfc, query_numbers, text_forms,
)

OUT = ROOT / "eval/v6"
SEED = 42
N_FLAGGED, N_UNFLAGGED = 40, 20
LABELS = ["inferable_from_query", "from_prior_observation", "tool_default_or_enum", "ungrounded"]

PARAM_TYPE = {
    "account_number": "account number", "to_account": "account number", "from_account": "account number",
    "tax_code": "tax code", "customer_id": "customer ID", "order_id": "order ID",
    "citizen_id": "citizen/ID number", "bhxh_number": "citizen/ID number", "id_number": "citizen/ID number",
    "card_number": "card number", "voucher_code": "voucher code", "booking_id": "booking ID",
    "shipping_address": "address", "address": "address", "office": "office/place",
    "full_name": "person name", "passenger_names": "person name", "recipient_name": "person name",
    "date": "date", "checkin": "date", "checkout": "date", "from_date": "date", "to_date": "date",
    "max_price": "money amount", "amount": "money amount", "initial_deposit": "money amount",
    "quantity": "count", "passengers": "count", "guests": "count",
    "keyword": "free text (product/query)", "product_name": "free text (product/query)",
    "text": "free text (product/query)", "query": "free text (product/query)",
    "from_currency": "currency/language code", "to_currency": "currency/language code",
    "source_lang": "currency/language code", "target_lang": "currency/language code",
    "cart_id": "system ID from earlier step", "product_id": "system ID from earlier step",
    "flight_id": "system ID from earlier step", "hotel_id": "system ID from earlier step",
    "receipt_number": "receipt/plate/customer code", "customer_code": "receipt/plate/customer code",
    "plate_number": "receipt/plate/customer code",
    "platform": "categorical option", "payment_method": "categorical option", "reason": "categorical option",
    "category": "categorical option", "refund_method": "categorical option", "utility_type": "categorical option",
    "seat_class": "categorical option", "document_type": "categorical option", "product_type": "categorical option",
    "loan_type": "categorical option", "room_type": "categorical option", "type": "categorical option",
    "origin": "place", "destination": "place", "city": "place",
    "service_name": "free text (product/query)", "service": "free text (product/query)",
    "note": "free text (product/query)", "expression": "free text (product/query)",
    "term_months": "count", "year": "date",
}

GUIDE = """\
# Hướng dẫn gán nhãn — kiểm tra audit grounding (Step 0, v6)

## Mục đích
Bộ dữ liệu ViTool có các giá trị tham số "gold" (đáp án) mà model phải sinh ra.
Một script tự động (audit) đã đánh dấu những giá trị **không thể suy ra** từ
thông tin model được nhìn thấy. Ta cần biết audit đó đúng đến đâu trước khi sửa
dữ liệu dựa trên nó. Bạn gán nhãn 60 giá trị; một phần được audit đánh dấu, một
phần không — **bạn không được biết mục nào thuộc loại nào** (đừng mở
`audit_sample_key.json`).

## Mỗi mục cho bạn xem
- `query`: câu yêu cầu của người dùng (model chỉ thấy câu này + các bước trước).
- `previous_steps`: các lời gọi tool ở bước trước **và observation** (kết quả trả
  về) của chúng. Đây là thông tin model có tại bước này.
- `tool`, `tool_description`, `param_schema`: tool đang gọi và mô tả tham số
  (kiểu, `enum` = danh sách giá trị cho phép, `required`).
- `gold_call`: toàn bộ lời gọi gold của bước này (để hiểu ngữ cảnh).
- `param` và `gold_value`: **giá trị cần gán nhãn**. Nếu là phần tử của một danh
  sách (vd 1 tên hành khách), `gold_value` là phần tử đó.

## Câu hỏi
"Một người cẩn thận, chỉ đọc `query` và `previous_steps`, có viết ra **đúng giá
trị này** (cho phép khác định dạng) mà không phải đoán không?"

## Nhãn — chọn đúng 1, xét theo thứ tự ưu tiên từ trên xuống
1. `inferable_from_query` — giá trị có trong query, nguyên văn **hoặc** sau khi
   chuẩn hoá/suy luận chắc chắn:
   - khác định dạng: `25/12/2026` ↔ `2026-12-25`; `1,5 triệu` ↔ `1500000`;
     `500k` ↔ `500000`; có/không dấu; viết hoa/thường; `HN` ↔ `Hà Nội`;
   - dịch/chuẩn hoá tên: `tiếng Pháp` → `fr`; `đô la Mỹ` → `USD`;
   - ngày tương đối khi query có mốc rõ (vd "ngày mai" và query có ngày hôm nay);
   - số lượng 1 khi query nói rõ một món/một người ("mua **một** chiếc", "đặt vé
     cho tôi") và không có con số nào khác.
   Nếu phải chọn giữa nhiều giá trị hợp lý → KHÔNG phải nhãn này.
2. `from_prior_observation` — giá trị xuất hiện trong observation của một bước
   trước (vd `flight_id`, `cart_id`, `product_id` do bước search/add trả về).
   Chỉ tính observation, không tính params của bước trước (nếu param bước trước
   cũng là giá trị bịa thì vẫn là bịa).
3. `tool_default_or_enum` — giá trị nằm trong `enum` của tham số (hoặc là
   `default` khai báo trong `param_schema`). Điền thêm `enum_justified`:
   `y` nếu ý nghĩa query ủng hộ đúng lựa chọn này (vd "mất thẻ" → `lost`),
   `n` nếu query không nói gì và đây chỉ là một lựa chọn tuỳ ý.
4. `ungrounded` — không thuộc 3 loại trên: giá trị phải bịa/đoán (số tài khoản,
   mã khách hàng, mã đơn, mã voucher, địa chỉ, tên người, giá tiền, ngày… không
   có trong query/observation; số lượng khi query không nói gì).

## Quy tắc thêm
- Chấm **giá trị**, không chấm việc tham số có nên tồn tại hay không.
- Không dùng kiến thức bên ngoài để "đoán đúng" (vd tự biết mã sân bay) — trừ
  các ánh xạ chuẩn hoá phổ biến ở mục 1.
- Phân vân giữa `inferable_from_query` và `ungrounded` → chọn cái bạn tin hơn và
  ghi lý do vào `note`. Mọi mục đều phải có đúng 1 nhãn.
- Làm **độc lập**, không trao đổi với người kia cho tới khi cả hai nộp xong.

## Cách điền
- File JSON riêng của bạn (`audit_sample_annotator1.json` hoặc `..._annotator2.json`):
  với mỗi mục, sửa `"label": null` thành một trong 4 chuỗi trên (giữ dấu ngoặc
  kép), `"enum_justified"` thành `"y"`/`"n"` khi nhãn là `tool_default_or_enum`,
  và `"note"` nếu cần. Không sửa field khác.
- Hoặc dùng `audit_sample.csv` (cột `label_annotator_1` / `label_annotator_2`).
- Gửi lại file; script `eval/v6/score_audit_v6.py` sẽ kiểm tra định dạng và tính
  precision/recall của audit + Cohen's kappa.

## Ví dụ
| query (rút gọn) | param = gold_value | nhãn |
|---|---|---|
| "Chuyển 2 triệu cho tài khoản 0123456789" | amount = 2000000 | inferable_from_query |
| "Khoá thẻ vì bị mất" | reason = lost (enum) | tool_default_or_enum, enum_justified=y |
| "Thanh toán giỏ hàng" (không nói phương thức) | payment_method = cod (enum) | tool_default_or_enum, enum_justified=n |
| bước trước trả `{"cart_id": "CART25E4F4"}` | cart_id = CART25E4F4 | from_prior_observation |
| "Xem lịch sử giao dịch tháng này" | account_number = 1234567890 | ungrounded |
| "Mua laptop Dell và theo dõi đơn" | shipping_address = 123 Nguyễn Trãi | ungrounded |
"""


def param_type(name):
    return PARAM_TYPE.get(name, "other")


def enumerate_values():
    """Every gold param value of all 375 trajectories with the audit's decision."""
    catalog = load_catalog(ROOT / "data/vitool/tool_catalog.json")
    eval_ids = {t.traj_id for t in load_eval(ROOT / "data/vitool/pilot_eval.jsonl")}
    trajs = load_eval(ROOT / "data/vitool/sft_train.jsonl") + load_eval(ROOT / "data/vitool/pilot_eval.jsonl")
    out = []
    for t in trajs:
        qforms, qnums = text_forms(t.query), query_numbers(t.query)
        for i, s in enumerate(t.steps):
            hist, _ = gold_prefix(t, i)
            prev_vals = [x for _, o in hist for x in flatten(o)]
            prev_text = nfc(json.dumps([o for _, o in hist], ensure_ascii=False))
            tool = catalog.get(s.expected_tool, {})
            props = tool.get("parameters", {}).get("properties", {})
            req = tool.get("parameters", {}).get("required", [])
            for k, val in s.expected_params.items():
                vals = list(flatten(val)) if isinstance(val, (list, dict)) else [val]
                for j, v in enumerate(vals):
                    src = ground_value(v, props.get(k, {}), t.query, qforms, qnums, prev_vals, prev_text)
                    out.append({
                        "traj_id": t.traj_id, "split": "eval" if t.traj_id in eval_ids else "train",
                        "domain": t.domain, "tier": s.difficulty.value, "step_idx": i,
                        "query": t.query, "tool": s.expected_tool,
                        "tool_description": f"{tool.get('description', '')} / {tool.get('description_vi', '')}",
                        "param": k + (f"[{j}]" if isinstance(val, list) else ""),
                        "param_name": k,
                        "param_schema": {**props.get(k, {}), "required": k in req},
                        "gold_value": v,
                        "gold_call": {"tool_name": s.expected_tool, "params": s.expected_params},
                        "previous_steps": [{"step": n, "call": c, "observation": o}
                                           for n, (c, o) in enumerate(hist)],
                        "audit_strict": src or "UNGROUNDED",
                        "audit_lenient": src or (f"lenient:{lenient_ok(k, v, t.query)}"
                                                 if lenient_ok(k, v, t.query) else "UNGROUNDED"),
                    })
    return out


def stratified(pool, n, key, rng):
    cells = defaultdict(list)
    for x in pool:
        cells[key(x)].append(x)
    keys = sorted(cells)
    quota = {k: max(1, len(cells[k]) * n / len(pool)) for k in keys}
    alloc = {k: int(quota[k]) for k in keys}
    rest = n - sum(alloc.values())
    for k in sorted(keys, key=lambda k: quota[k] - int(quota[k]), reverse=True):
        if rest <= 0:
            break
        if alloc[k] < len(cells[k]):
            alloc[k] += 1
            rest -= 1
    while sum(alloc.values()) > n:  # many tiny cells: trim the largest allocations
        k = max(keys, key=lambda k: alloc[k])
        alloc[k] -= 1
    picked = []
    for k in keys:
        picked += rng.sample(cells[k], min(alloc[k], len(cells[k])))
    return picked, {f"{k[0]}|{k[1]}" if isinstance(k, tuple) else k: (alloc[k], len(cells[k])) for k in keys}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    vals = enumerate_values()
    rng = random.Random(SEED)
    flagged = [v for v in vals if v["audit_strict"] == "UNGROUNDED"]
    unflagged = [v for v in vals if v["audit_strict"] != "UNGROUNDED"]
    fs, f_alloc = stratified(flagged, N_FLAGGED, lambda x: (x["tier"], x["domain"]), rng)
    us, u_alloc = stratified(unflagged, N_UNFLAGGED, lambda x: x["tier"], rng)
    items = fs + us
    rng.shuffle(items)

    shown = ["item_id", "query", "previous_steps", "tool", "tool_description", "param",
             "param_schema", "gold_value", "gold_call", "traj_id", "step_idx", "domain", "tier", "split"]
    public, key = [], []
    for n, x in enumerate(items, 1):
        x["item_id"] = f"A{n:02d}"
        public.append({k: x[k] for k in shown})
        key.append({"item_id": x["item_id"], "audit_strict": x["audit_strict"],
                    "audit_lenient": x["audit_lenient"],
                    "audit_flag_strict": x["audit_strict"] == "UNGROUNDED",
                    "audit_flag_lenient": x["audit_lenient"] == "UNGROUNDED",
                    "stratum": "flagged" if x in fs else "unflagged",
                    "param_type": param_type(x["param_name"])})

    instructions = {"guide_markdown": GUIDE, "allowed_labels": LABELS,
                    "fields_to_fill": ["label", "enum_justified (only for tool_default_or_enum: y/n)", "note"]}
    both = {"instructions": instructions,
            "items": [{**p, "label_annotator_1": None, "enum_justified_annotator_1": None, "note_annotator_1": "",
                       "label_annotator_2": None, "enum_justified_annotator_2": None, "note_annotator_2": ""}
                      for p in public]}
    (OUT / "audit_sample.json").write_text(json.dumps(both, ensure_ascii=False, indent=2))
    for a in (1, 2):
        one = {"annotator": a, "instructions": instructions,
               "items": [{**p, "label": None, "enum_justified": None, "note": ""} for p in public]}
        (OUT / f"audit_sample_annotator{a}.json").write_text(json.dumps(one, ensure_ascii=False, indent=2))
    with open(OUT / "audit_sample.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "query", "previous_steps (call -> observation)", "tool", "tool_description",
                    "param", "param_schema", "gold_value", "gold_call",
                    "label_annotator_1", "enum_justified_annotator_1", "note_annotator_1",
                    "label_annotator_2", "enum_justified_annotator_2", "note_annotator_2"])
        for p in public:
            prev = "\n".join(f"[{s['step']}] {s['call']['tool_name']}({json.dumps(s['call']['params'], ensure_ascii=False)})"
                             f" -> {json.dumps(s['observation'], ensure_ascii=False)}" for s in p["previous_steps"]) or "(none)"
            w.writerow([p["item_id"], p["query"], prev, p["tool"], p["tool_description"], p["param"],
                        json.dumps(p["param_schema"], ensure_ascii=False), json.dumps(p["gold_value"], ensure_ascii=False),
                        json.dumps(p["gold_call"], ensure_ascii=False), "", "", "", "", "", ""])
    (OUT / "audit_sample_key.json").write_text(json.dumps(
        {"note": "DO NOT SHOW TO ANNOTATORS", "seed": SEED,
         "population": {"flagged_values": len(flagged), "unflagged_values": len(unflagged)},
         "allocation_flagged_tier|domain": f_alloc, "allocation_unflagged_tier": u_alloc,
         "items": key}, ensure_ascii=False, indent=2))
    (OUT / "ANNOTATION_GUIDE.md").write_text(GUIDE)

    # value frequency of ungrounded values across splits ("conventional" values)
    freq = defaultdict(lambda: {"train": 0, "eval": 0, "params": Counter(), "trajs": set()})
    for v in flagged:
        k = json.dumps(v["gold_value"], ensure_ascii=False)
        freq[k][v["split"]] += 1
        freq[k]["params"][v["param_name"]] += 1
        freq[k]["trajs"].add(v["traj_id"])
    rows = sorted(({"value": k, "total": f["train"] + f["eval"], "train": f["train"], "eval": f["eval"],
                    "n_trajectories": len(f["trajs"]), "params": dict(f["params"])}
                   for k, f in freq.items()), key=lambda r: -r["total"])
    rep = [r for r in rows if r["total"] >= 2]
    by_type = Counter(param_type(v["param_name"]) for v in flagged)
    summary = {
        "flagged_values_total": len(flagged),
        "distinct_flagged_values": len(rows),
        "values_occurring_>=2_times": len(rep),
        "flagged_occurrences_covered_by_repeated_values": sum(r["total"] for r in rep),
        "repeated_values_in_both_splits": sum(1 for r in rep if r["train"] and r["eval"]),
        "eval_flagged_occurrences_whose_value_also_appears_in_train": sum(
            r["eval"] for r in rows if r["train"] and r["eval"]),
        "eval_flagged_occurrences": sum(r["eval"] for r in rows),
        "flagged_by_param_type_(audit,_unvalidated)": dict(by_type.most_common()),
        "values": rows,
    }
    (OUT / "audit_value_frequency.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "values"}, ensure_ascii=False, indent=1))
    print("top repeated:", [(r["value"], r["train"], r["eval"]) for r in rows[:15]])
    print("flagged alloc:", f_alloc)
    print("unflagged alloc:", u_alloc)


if __name__ == "__main__":
    main()
