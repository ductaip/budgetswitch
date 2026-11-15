"""eval/v6/export_v12_spotcheck.py — 20 random ViTool v1.2 trajectories (seed 42) for the two authors.
Same task and fields as v11_spotcheck (CHECKS_GUIDE.md §2); trajectories whose query changed in
v1.2 also show the v1.1 query. Writes v12_spotcheck.csv and v12_spotcheck_annotator{1,2}.json."""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from eval.v5.common_v5 import gold_prefix, load_eval  # noqa: E402
from eval.v6.export_checks_v6 import write_pair  # noqa: E402

v11 = {t.traj_id: t.query for n in ("sft_train_v1.1", "pilot_eval_v1.1") for t in load_eval(ROOT / f"data/vitool/{n}.jsonl")}
split, trajs = {}, []
for s, n in (("train", "sft_train_v1.2"), ("eval", "pilot_eval_v1.2")):
    for t in load_eval(ROOT / f"data/vitool/{n}.jsonl"):
        split[t.traj_id] = s
        trajs.append(t)
sel = random.Random(42).sample(trajs, 20)
items = []
for k, t in enumerate(sel):
    full, _ = gold_prefix(t, t.num_steps)
    items.append({"item_id": f"T{k + 1:02d}", "traj_id": t.traj_id, "split": split[t.traj_id], "domain": t.domain,
                  "tier": t.steps[0].difficulty.value, "query_changed_in_v1.2": t.query != v11[t.traj_id],
                  "query_v1.1": v11[t.traj_id] if t.query != v11[t.traj_id] else None, "query": t.query,
                  "steps": [{"step": i, "gold_call": {"tool_name": s.expected_tool, "params": s.expected_params},
                             "observation_of_this_step": full[i][1]} for i, s in enumerate(t.steps)]})
write_pair("v12_spotcheck", items, ["fabricated_value_found", "fabricated_values"],
           "Xem eval/v6/CHECKS_GUIDE.md mục 2 (giờ cả lựa chọn enum như phương thức thanh toán, sàn, "
           "cách hoàn tiền cũng phải được người dùng nói ra).")
print(len(items), "items;", sum(x["query_changed_in_v1.2"] for x in items), "changed in v1.2;",
      sum(x["split"] == "eval" for x in items), "eval")
