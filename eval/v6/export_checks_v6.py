"""
eval/v6/export_checks_v6.py — manual checks after Step 2 (seed 42).

1) rewrite_check (brief Step 2): 50 random accepted rewrites, old vs new query.
   Each author fills: meaning_preserved (y/n), natural (1-3).
2) v11_spotcheck (author request): 30 random v1.1 trajectories (all splits),
   query + full gold calls with previous observations. Each author fills:
   fabricated_value_found (y/n), fabricated_values (list "param=value"), note.

Writes for each check: <name>.csv (both annotators' columns) and
<name>_annotator1.json / _annotator2.json (one file per author, easier to fill),
plus CHECKS_GUIDE.md. Score with: venv/bin/python eval/v6/score_checks_v6.py
"""

from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from eval.v5.common_v5 import gold_prefix, load_eval  # noqa: E402

V6 = ROOT / "eval/v6"
SEED = 42

GUIDE = """# Hướng dẫn kiểm tra sau Step 2 (ViTool v1.1)

Hai tác giả làm **độc lập**, mỗi người một file `_annotator1.json` / `_annotator2.json`.
Chỉ sửa các trường đang là `null` (và `note` nếu cần). Không sửa trường khác.

## 1. rewrite_check (50 mục): query cũ → query mới

Mỗi mục có `old_query`, `new_query` và `added_values` (các giá trị đã được thêm vào).

- `meaning_preserved`: `"y"` nếu query mới giữ đúng ý định và mọi thông tin của query cũ,
  chỉ **thêm** các giá trị trong `added_values`; `"n"` nếu làm mất, đổi nghĩa, hoặc thêm
  thông tin ngoài danh sách.
- `natural`: mức tự nhiên của query mới như người Việt gõ cho trợ lý:
  - `3` = tự nhiên, không có gì gượng;
  - `2` = hiểu được nhưng hơi gượng (liệt kê kiểu điền form, lặp từ, trật tự lạ);
  - `1` = không tự nhiên (lộ mã kỹ thuật, tên tham số tiếng Anh, câu sai ngữ pháp).

## 2. v11_spotcheck (30 trajectory v1.1 ngẫu nhiên)

Mỗi mục có `query`, và với từng bước: lời gọi gold + observation của các bước trước.
Câu hỏi: **còn giá trị nào trong lời gọi gold mà người dùng chưa nói, không suy ra được
từ query, không lấy từ observation trước, và không phải lựa chọn enum hiển nhiên?**
(Cùng tiêu chí với `ANNOTATION_GUIDE.md` của Step 0; số lượng không nêu trong query
tính là chưa grounded.)

- `fabricated_value_found`: `"y"` hoặc `"n"`.
- `fabricated_values`: nếu `"y"`, ghi danh sách dạng `["add_to_cart.quantity=2", ...]`.
- `note`: tuỳ chọn (vd query mới có vấn đề khác).
"""


def write_pair(name, items, fields, intro):
    for a in (1, 2):
        out = {"instructions": intro, "fill_fields": fields,
               "items": [{**it, **{f: None for f in fields}, "note": ""} for it in items]}
        (V6 / f"{name}_annotator{a}.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
    cols = list(items[0]) + [f"{f}_annotator_{a}" for a in (1, 2) for f in fields + ["note"]]
    with open(V6 / f"{name}.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for it in items:
            w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
                        for k, v in it.items()})


def main():
    diffs = [json.loads(l) for l in open(V6 / "query_diffs.jsonl", encoding="utf-8")]
    acc = [d for d in diffs if d["new_query"]]
    rng = random.Random(SEED)
    pick = sorted(rng.sample(range(len(acc)), min(50, len(acc))))
    rng.shuffle(pick)
    rw = [{"item_id": f"R{k + 1:02d}", "traj_id": acc[i]["traj_id"], "split": acc[i]["split"],
           "old_query": acc[i]["old_query"], "new_query": acc[i]["new_query"],
           "added_values": [f"{b['tool']}.{b['param']}={json.dumps(b['value'], ensure_ascii=False)}"
                            for b in acc[i]["missing_values"]]}
          for k, i in enumerate(pick)]
    write_pair("rewrite_check", rw, ["meaning_preserved", "natural"],
               "Xem eval/v6/CHECKS_GUIDE.md mục 1.")

    trajs = [t for name in ("sft_train_v1.1", "pilot_eval_v1.1")
             for t in load_eval(ROOT / f"data/vitool/{name}.jsonl")]
    split = {t.traj_id: s for s, name in (("train", "sft_train_v1.1"), ("eval", "pilot_eval_v1.1"))
             for t in load_eval(ROOT / f"data/vitool/{name}.jsonl")}
    rewritten = {d["traj_id"] for d in acc}
    rng = random.Random(SEED + 1)
    sel = rng.sample(trajs, 30)
    sp = []
    for k, t in enumerate(sel):
        steps = []
        full, _ = gold_prefix(t, t.num_steps)  # [(call, observation)] for every gold step
        for i, s in enumerate(t.steps):
            steps.append({"step": i, "gold_call": {"tool_name": s.expected_tool, "params": s.expected_params},
                          "observation_of_this_step": full[i][1]})
        sp.append({"item_id": f"S{k + 1:02d}", "traj_id": t.traj_id, "split": split[t.traj_id],
                   "domain": t.domain, "tier": t.steps[0].difficulty.value,
                   "query_was_rewritten": t.traj_id in rewritten, "query": t.query, "steps": steps})
    write_pair("v11_spotcheck", sp, ["fabricated_value_found", "fabricated_values"],
               "Xem eval/v6/CHECKS_GUIDE.md mục 2.")
    (V6 / "CHECKS_GUIDE.md").write_text(GUIDE)
    print(f"rewrite_check: {len(rw)} items; v11_spotcheck: {len(sp)} items "
          f"({sum(x['query_was_rewritten'] for x in sp)} rewritten, "
          f"{sum(x['split'] == 'eval' for x in sp)} eval)")


if __name__ == "__main__":
    main()
