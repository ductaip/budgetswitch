"""
scripts/rerun_m3.py — Thay NGUYÊN cột M3 trong pilot_results bằng 1 model ổn định.

VÌ SAO: 2 run liên tiếp, cột M3 (Groq free) đều chết dần vì quota theo THỨ TỰ chạy
(0.80→0.00) → heatmap M3 vô nghĩa. M0/M1/M2 là model local, đã chuẩn → giữ nguyên.
Chỉ gọi lại 284 call M3 với gpt-4o-mini (~$0.06), KHÔNG cần GPU, chạy local.

Methodology: thay TOÀN BỘ cột (mọi step, kể cả step trước đó success) để cột M3
là MỘT model duy nhất — không trộn 2 model trong 1 cột.

CHẠY (máy local, từ gốc repo):
  export OPENAI_API_KEY=...
  python scripts/rerun_m3.py \
      --eval data/vitool/pilot_eval.jsonl \
      --in  pilot_results.jsonl \
      --out pilot_results_v4.jsonl \
      --model gpt-4o-mini
  python eval/oracle.py  --in pilot_results_v4.jsonl --out eval/oracle_report_v4.json
  python eval/figures.py --in eval/oracle_report_v4.json --outdir eval/figs_v4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.vitool.schema import load_trajectories  # noqa: E402
from pilot.simulate_modes import _replay_history, clean_catalog  # noqa: E402
from verifiers.universal.schema_verifier import verify_step_outcome  # noqa: E402
from modes.m3_defer.m3_defer import m3_defer_generate  # noqa: E402

M3_COST = 15.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", dest="eval_path", default="data/vitool/pilot_eval.jsonl")
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument("--catalog", default="data/vitool/tool_catalog.json")
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--sleep", type=float, default=0.2, help="nghỉ giữa call (Groq free: dùng 12)")
    ap.add_argument("--resume", action="store_true",
                    help="đọc file --out cũ, bỏ qua step đã thay bằng đúng model này (chạy tiếp sau khi cạn quota ngày)")
    ap.add_argument("--client", default=None, help="(test) bỏ qua, dùng nội bộ")
    args = ap.parse_args()

    catalog = clean_catalog({k: v for k, v in json.load(open(args.catalog)).items()})
    trajs = {t.traj_id: t for t in load_trajectories(args.eval_path)}
    rows = [json.loads(l) for l in open(args.in_path)]

    done = {}
    if args.resume and Path(args.out_path).exists():
        for prev in (json.loads(l) for l in open(args.out_path)):
            m3 = prev.get("results", {}).get("M3", {})
            if m3.get("m3_model") == args.model:
                done[(prev["traj_id"], prev["step_idx"])] = prev
        print(f"[resume] {len(done)} steps đã xong từ lần trước → bỏ qua")

    client = _make_client(args.model)
    print(f"Rerun M3 = {args.model} cho {len(rows) - len(done)}/{len(rows)} steps (M0/M1/M2 GIỮ NGUYÊN)…")

    n_ok = 0
    n_done = 0
    quota_dead = False
    t_start = time.time()
    for i, r in enumerate(rows):
        key = (r["traj_id"], r["step_idx"])
        if key in done:
            rows[i] = done[key]
            n_ok += bool(done[key]["results"]["M3"]["success"])
            n_done += 1
            continue
        if quota_dead:
            continue  # giữ nguyên M3 cũ (KHÔNG tag) → lần --resume sau xử nốt
        traj = trajs.get(r["traj_id"])
        if traj is None:
            raise SystemExit(f"traj_id {r['traj_id']} không có trong {args.eval_path}")
        step_idx = r["step_idx"]
        gt = traj.steps[step_idx]
        gt_dict = {
            "expected_tool": gt.expected_tool,
            "expected_params": gt.expected_params,
            "fuzzy_keys": gt.fuzzy_keys,
        }
        history = _replay_history(traj, step_idx)

        t0 = time.time()
        try:
            out = m3_defer_generate(traj.query, history, catalog,
                                    llm_client=client, model_name=args.model)
            success = bool(verify_step_outcome(out, gt_dict, catalog))
            ctk = out.get("completion_tokens", 0)
            usd = out.get("api_cost_usd", 0.0)
        except RuntimeError as e:
            if "cạn quota" in str(e):
                print(f"\n⛔ Quota ngày cạn tại step {i}/{len(rows)}. Lưu tiến độ — mai chạy lại"
                      f" CÙNG lệnh + thêm --resume để tiếp tục từ đây.")
                quota_dead = True
                continue
            print(f"  [warn] M3 fail {r['traj_id']} step{step_idx}: {str(e)[:120]}")
            success, ctk, usd = False, 0, 0.0
        except Exception as e:
            print(f"  [warn] M3 fail {r['traj_id']} step{step_idx}: {str(e)[:120]}")
            success, ctk, usd = False, 0, 0.0
        lat = (time.time() - t0) * 1000

        r["results"]["M3"] = {
            "mode": "M3", "success": success, "cost": M3_COST,
            "latency_ms": lat, "completion_tokens": ctk, "api_cost_usd": usd,
            "m3_model": args.model,  # tag provenance + để --resume nhận diện
        }
        n_done += 1
        # oracle_mode trong row là của run cũ → xoá để khỏi gây hiểu lầm
        # (oracle.py tự tính lại từ results, không đọc field này)
        r.pop("oracle_mode", None)
        n_ok += success
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(rows)} | M3 success tạm: {n_ok/(i+1):.2f}")
        time.sleep(args.sleep)

    with open(args.out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    mins = (time.time() - t_start) / 60
    pending = len(rows) - n_done
    status = "✅ HOÀN TẤT" if pending == 0 else f"⏸ CÒN {pending} steps (chạy lại với --resume khi quota reset)"
    print(f"\n{status} — {n_done}/{len(rows)} steps, {mins:.1f} phút | M3 success (phần đã thay): {n_ok/max(n_done,1):.3f}")
    print(f"   → {args.out_path}")
    print("   Kiểm sức khỏe: success theo THỨ TỰ chạy phải PHẲNG (không decay):")
    m3 = [1 if r["results"]["M3"]["success"] else 0 for r in rows]
    print("  ", [round(sum(m3[i:i+50]) / len(m3[i:i+50]), 2) for i in range(0, len(m3), 50)])


def _make_client(model: str):
    from openai import OpenAI
    if model.startswith("gemini"):
        # FREE không cần thẻ: lấy key tại https://aistudio.google.com/apikey
        # flash-lite free ~15 RPM / 1000 req/ngày → 284 call OK trong 1 lần chạy
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise SystemExit("Cần GEMINI_API_KEY (free, không cần thẻ: aistudio.google.com/apikey).")
        return OpenAI(api_key=key,
                      base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    if "/" in model:
        # model dạng provider/name (vd openai/gpt-oss-120b) = open model trên GROQ.
        # LƯU Ý: "openai/gpt-oss-120b" KHÔNG phải API OpenAI — bug cũ startswith("o")
        # từng route nhầm sang api.openai.com → 400 invalid model ID.
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise SystemExit("Model dạng provider/name chạy trên Groq → cần GROQ_API_KEY.")
        return OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
    if model.startswith(("gpt-", "o1", "o3", "o4")):
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise SystemExit("Cần OPENAI_API_KEY (model OpenAI). ~$0.06 cho 284 call gpt-4o-mini.")
        return OpenAI(api_key=key)
    # model khác → Groq (không khuyến nghị cho run chính thức vì quota)
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        raise SystemExit("Model không phải OpenAI → cần GROQ_API_KEY.")
    print("⚠️ Dùng Groq cho M3 — quota ngày đã giết 2 run trước. Cân nhắc gpt-4o-mini.")
    return OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")


if __name__ == "__main__":
    main()