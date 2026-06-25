"""
eval/error_analysis.py — Trích 2 thứ cho paper (phương án 6 trang):
  (1) Bảng SUCCESS theo DOMAIN × mode  -> Table IV trong paper
  (2) 2-3 VÍ DỤ LỖI THẬT (mode đúng vs sai trên cùng 1 step) -> minh hoạ §V-D

KHÔNG bịa gì: mọi số/ví dụ đọc trực tiếp từ pilot_results_v4.jsonl.

CHẠY (local, ~5 giây, không cần GPU):
  python eval/error_analysis.py \
      --results pilot_results_v4.jsonl \
      --eval    data/vitool/pilot_eval.jsonl \
      --out     eval/error_analysis.json

Rồi GỬI output (cả phần in ra màn hình + file json) cho mình để điền vào paper.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODES = ["M0", "M1", "M2", "M3"]


def _ok(step, mode):
    r = step.get("results", {}).get(mode)
    return bool(r and r.get("success"))


def _pred(step, mode):
    """Lấy tool/params model M dự đoán (nếu pilot có lưu)."""
    r = step.get("results", {}).get(mode, {})
    # tuỳ field pilot lưu — thử vài tên hay gặp
    for k in ("prediction", "pred", "output", "call"):
        if k in r:
            return r[k]
    return r.get("pred_tool"), r.get("pred_params")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--eval", default="data/vitool/pilot_eval.jsonl")
    ap.add_argument("--out", default="eval/error_analysis.json")
    args = ap.parse_args()

    steps = [json.loads(l) for l in open(args.results)]

    # map traj_id -> (domain, query) từ eval set
    meta = {}
    try:
        from data.vitool.schema import load_trajectories
        for t in load_trajectories(args.eval):
            meta[t.traj_id] = {"domain": getattr(t, "domain", None) or "?",
                               "query": t.query}
    except Exception as e:
        print(f"[info] không dùng schema ({e}); đọc eval jsonl trực tiếp")
    # fallback / bổ sung: đọc thẳng jsonl để chắc có query + domain
    try:
        for line in open(args.eval, encoding="utf-8"):
            d = json.loads(line)
            tid = d.get("traj_id") or d.get("id")
            if not tid:
                continue
            if tid not in meta or meta[tid].get("query") in (None, "?"):
                meta[tid] = {"domain": d.get("domain", "?"),
                             "query": d.get("query") or d.get("user_request") or "?"}
    except Exception as e:
        print(f"[warn] không đọc được {args.eval} trực tiếp: {e}")

    # ---------- (1) success theo domain × mode ----------
    # đếm theo domain (suy từ traj_id prefix nếu thiếu meta)
    def domain_of(step):
        tid = step["traj_id"]
        if tid in meta and meta[tid]["domain"] != "?":
            return meta[tid]["domain"]
        # fallback: prefix trước dấu _ đầu (ecommerce_0008 -> ecommerce; public_services_... -> public)
        return step.get("domain") or tid.split("_")[0]

    dom_tot = defaultdict(int)
    dom_ok = defaultdict(lambda: defaultdict(int))
    for s in steps:
        d = domain_of(s)
        dom_tot[d] += 1
        for m in MODES:
            dom_ok[d][m] += _ok(s, m)

    print("\n" + "=" * 70)
    print("(1) SUCCESS RATE theo DOMAIN × mode  (điền vào Table IV)")
    print("=" * 70)
    header = f"{'Domain':<20}{'n':>5}" + "".join(f"{m:>8}" for m in MODES)
    print(header)
    print("-" * len(header))
    domain_rows = {}
    for d in sorted(dom_tot, key=lambda x: -dom_tot[x]):
        n = dom_tot[d]
        rates = {m: dom_ok[d][m] / n for m in MODES}
        domain_rows[d] = {"n": n, **{m: round(rates[m], 3) for m in MODES}}
        print(f"{d:<20}{n:>5}" + "".join(f"{rates[m]:>8.2f}" for m in MODES))

    # ---------- (2) ví dụ lỗi thật ----------
    # case thú vị: M0 ĐÚNG nhưng M3 SAI (minh hoạ defer thua)  + step mọi mode SAI (unsolvable)
    m0ok_m3fail = []
    all_fail = []
    for s in steps:
        tid = s["traj_id"]
        q = meta.get(tid, {}).get("query", "?")
        if _ok(s, "M0") and not _ok(s, "M3"):
            m0ok_m3fail.append({
                "traj_id": tid, "step_idx": s.get("step_idx"),
                "difficulty": s.get("difficulty"), "query": q,
                "M3_pred": _pred(s, "M3"),
            })
        if all(not _ok(s, m) for m in MODES):
            all_fail.append({
                "traj_id": tid, "step_idx": s.get("step_idx"),
                "difficulty": s.get("difficulty"), "query": q,
            })

    print("\n" + "=" * 70)
    print(f"(2) VÍ DỤ LỖI THẬT")
    print("=" * 70)
    print(f"\n[A] M0 ĐÚNG nhưng M3 (defer 120B) SAI — có {len(m0ok_m3fail)} case. 3 ví dụ:")
    for c in m0ok_m3fail[:3]:
        print(f"  - [{c['difficulty']}/{c['traj_id']} step{c['step_idx']}] {str(c['query'])[:90]}")
        if c["M3_pred"] and any(c["M3_pred"]):
            print(f"      M3 dự đoán: {c['M3_pred']}")
    print(f"\n[B] KHÔNG mode nào giải được (unsolvable) — có {len(all_fail)} case. 3 ví dụ:")
    for c in all_fail[:3]:
        print(f"  - [{c['difficulty']}/{c['traj_id']} step{c['step_idx']}] {str(c['query'])[:90]}")

    out = {
        "domain_success": domain_rows,
        "n_m0ok_m3fail": len(m0ok_m3fail),
        "examples_m0ok_m3fail": m0ok_m3fail[:5],
        "n_all_fail": len(all_fail),
        "examples_all_fail": all_fail[:5],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved → {args.out}")
    print("→ GỬI mình: phần in trên + file json. Mình sẽ điền Table IV + ví dụ vào paper.")
    if not any(c.get("M3_pred") and any(c["M3_pred"]) for c in m0ok_m3fail[:5]):
        print("\n[lưu ý] pilot_results không lưu prediction của từng mode → ví dụ sẽ chỉ có")
        print("        query + tier (không có 'M3 sinh X thay vì Y'). Vẫn đủ để minh hoạ.")


if __name__ == "__main__":
    main()