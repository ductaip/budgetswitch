"""
data/verify_cli.py — CLI human-verify tool cho ViTool (Cổng 3, v10.0).

VÌ SAO: verify 600–700 trajectory bằng cách mở file JSON sửa tay = chậm + dễ sai.
Tool này hiện từng trajectory gọn gàng (query + steps + final-state đã resolve),
cho phép accept / edit / reject / skip bằng 1 phím, LƯU TIẾN ĐỘ (resume được),
và ghi log để tính Cohen's kappa khi 2 người verify chồng.

Cổng 1+2 đã đảm bảo trajectory CHẠY ĐƯỢC. Việc của con người ở đây là chất lượng
NGỮ NGHĨA mà gate không bắt được:
  - Query có tự nhiên như người Việt gõ không? (không phải "customer_id: không xác định")
  - Tool sequence có HỢP LÝ với ý định query không?
  - Độ khó gán có đúng không? (easy/medium/hard)
  - Params có khớp ý query không (số tiền, ngày, địa danh)?

DÙNG:
  python data/verify_cli.py --in data/vitool/merged.jsonl \
         --out data/vitool/verified.jsonl --reviewer A
  # resume: chạy lại cùng lệnh, tool bỏ qua cái đã xử lý (theo progress file)
  # tính kappa sau khi A và B cùng verify một overlap set:
  python data/verify_cli.py --kappa data/vitool/verified_A.jsonl data/vitool/verified_B.jsonl

PHÍM:
  [a] accept    [e] edit (mở JSON trong $EDITOR)   [r] reject
  [d] đổi độ khó   [s] skip (để sau)   [q] lưu & thoát
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.vitool.schema import load_trajectories, save_trajectories, Difficulty  # noqa: E402
from env.mock_tools import is_executable, compute_final_state  # noqa: E402

C = {"g": "\033[92m", "y": "\033[93m", "r": "\033[91m", "b": "\033[94m",
     "d": "\033[2m", "bold": "\033[1m", "x": "\033[0m"}


def _progress_path(out_path: str) -> Path:
    return Path(out_path).with_suffix(".progress.json")


def _load_progress(out_path: str) -> dict:
    p = _progress_path(out_path)
    if p.exists():
        return json.loads(p.read_text())
    return {"decisions": {}}  # traj_id -> "accept"/"reject"/"edited"


def _save_progress(out_path: str, prog: dict) -> None:
    _progress_path(out_path).write_text(json.dumps(prog, ensure_ascii=False, indent=2))


def _show(traj, idx, total, prog) -> None:
    diff = ",".join(sorted({s.difficulty.value for s in traj.steps}))
    done = len(prog["decisions"])
    print("\n" + "=" * 64)
    print(f"{C['bold']}[{idx + 1}/{total}]{C['x']} id={traj.traj_id}  "
          f"domain={C['b']}{traj.domain}{C['x']}  difficulty={C['y']}{diff}{C['x']}  "
          f"{C['d']}(đã xong {done}){C['x']}")
    print(f"{C['bold']}Query:{C['x']} {traj.query}")
    print(f"{C['bold']}Steps ({len(traj.steps)}):{C['x']}")
    for i, s in enumerate(traj.steps, 1):
        print(f"  {i}. {C['g']}{s.expected_tool}{C['x']}  {json.dumps(s.expected_params, ensure_ascii=False)}")
    if traj.expected_final_state:
        effects = {k: v for k, v in traj.expected_final_state.items() if v}
        print(f"{C['d']}final-state (mutation): {json.dumps(effects, ensure_ascii=False) or '{}'}{C['x']}")
    ok, reason = is_executable([{"tool_name": s.expected_tool, "params": s.expected_params}
                               for s in traj.steps])
    flag = f"{C['g']}executable ✓{C['x']}" if ok else f"{C['r']}KHÔNG chạy được: {reason}{C['x']}"
    print(f"  [{flag}]")


def _edit_traj(traj):
    """Mở step JSON trong $EDITOR; re-validate sau khi sửa."""
    editor = os.getenv("EDITOR", "nano")
    payload = {"query": traj.query,
               "steps": [{"expected_tool": s.expected_tool,
                          "expected_params": s.expected_params,
                          "difficulty": s.difficulty.value} for s in traj.steps]}
    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        tmp = f.name
    subprocess.call([editor, tmp])
    try:
        edited = json.loads(Path(tmp).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print(f"{C['r']}JSON lỗi, bỏ chỉnh sửa.{C['x']}")
        return traj, False
    finally:
        os.unlink(tmp)
    from data.vitool.schema import Step
    traj.query = edited.get("query", traj.query)
    traj.steps = [Step(expected_tool=s["expected_tool"], expected_params=s["expected_params"],
                       difficulty=Difficulty(s.get("difficulty", "medium")))
                  for s in edited["steps"]]
    steps_d = [{"tool_name": s.expected_tool, "params": s.expected_params} for s in traj.steps]
    ok, reason = is_executable(steps_d)
    if ok:
        traj.expected_final_state = compute_final_state(steps_d)
        print(f"{C['g']}✓ sửa xong, vẫn executable, final-state cập nhật.{C['x']}")
    else:
        print(f"{C['r']}⚠ sau khi sửa KHÔNG chạy được: {reason} — vẫn lưu nhưng nên xem lại.{C['x']}")
    return traj, True


def run_verify(in_path, out_path, reviewer):
    trajs = load_trajectories(in_path)
    prog = _load_progress(out_path)
    # load đã-accept từ trước (resume)
    accepted = {t.traj_id: t for t in (load_trajectories(out_path) if Path(out_path).exists() else [])}

    todo = [t for t in trajs if t.traj_id not in prog["decisions"]]
    print(f"{C['bold']}Verify {len(trajs)} traj — còn {len(todo)} chưa xử lý "
          f"(reviewer={reviewer}).{C['x']}  Phím: a/e/r/d/s/q")

    for idx, traj in enumerate(trajs):
        if traj.traj_id in prog["decisions"]:
            continue
        while True:
            _show(traj, idx, len(trajs), prog)
            cmd = input(f"{C['bold']}[a]ccept [e]dit [r]eject [d]iff [s]kip [q]uit > {C['x']}").strip().lower()
            if cmd in ("a", ""):
                accepted[traj.traj_id] = traj
                prog["decisions"][traj.traj_id] = "accept"
                break
            elif cmd == "e":
                traj, _ = _edit_traj(traj)
                accepted[traj.traj_id] = traj
                prog["decisions"][traj.traj_id] = "edited"
                break
            elif cmd == "r":
                prog["decisions"][traj.traj_id] = "reject"
                accepted.pop(traj.traj_id, None)
                break
            elif cmd == "d":
                nd = input("  độ khó mới (easy/medium/hard): ").strip()
                if nd in ("easy", "medium", "hard"):
                    for s in traj.steps:
                        s.difficulty = Difficulty(nd)
                    print(f"  → đổi thành {nd}")
            elif cmd == "s":
                break  # để lại, không ghi decision
            elif cmd == "q":
                _flush(accepted, out_path, prog, reviewer)
                print(f"{C['y']}Đã lưu. Chạy lại lệnh để tiếp tục.{C['x']}")
                return
        # lưu sau mỗi quyết định (an toàn nếu crash)
        _flush(accepted, out_path, prog, reviewer)

    _flush(accepted, out_path, prog, reviewer)
    n_acc = sum(1 for v in prog["decisions"].values() if v in ("accept", "edited"))
    n_rej = sum(1 for v in prog["decisions"].values() if v == "reject")
    print(f"\n{C['g']}✅ Xong. Accept/edited {n_acc}, reject {n_rej} → {out_path}{C['x']}")


def _flush(accepted, out_path, prog, reviewer):
    save_trajectories(list(accepted.values()), out_path)
    _save_progress(out_path, prog)


# ============================================================
# Cohen's kappa (đo agreement 2 reviewer trên overlap set)
# ============================================================
def cohen_kappa(file_a, file_b):
    """
    So quyết định accept/reject của 2 reviewer trên cùng tập traj_id.
    Dùng progress file (.progress.json) cạnh mỗi output.
    """
    pa = _load_progress(file_a)["decisions"]
    pb = _load_progress(file_b)["decisions"]
    common = set(pa) & set(pb)
    if not common:
        print("Không có traj_id chung giữa 2 reviewer.")
        return
    # nhị phân hoá: accept/edited = 1 (giữ), reject = 0
    def b(v):
        return 1 if v in ("accept", "edited") else 0
    a = [b(pa[t]) for t in common]
    bb = [b(pb[t]) for t in common]
    n = len(common)
    po = sum(1 for x, y in zip(a, bb) if x == y) / n
    pa1 = sum(a) / n
    pb1 = sum(bb) / n
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    kappa = (po - pe) / (1 - pe) if (1 - pe) else 1.0
    print(f"Overlap: {n} traj | agreement quan sát po={po:.3f} | "
          f"kỳ vọng pe={pe:.3f} | {C['bold']}Cohen's κ = {kappa:.3f}{C['x']}")
    interp = ("gần như hoàn hảo" if kappa > 0.8 else "đáng kể" if kappa > 0.6
              else "vừa phải" if kappa > 0.4 else "yếu — cần thống nhất tiêu chí lại")
    print(f"  → mức đồng thuận: {interp}")


def main():
    ap = argparse.ArgumentParser(description="ViTool human-verify CLI (Cổng 3)")
    ap.add_argument("--in", dest="in_path")
    ap.add_argument("--out", dest="out_path")
    ap.add_argument("--reviewer", default="A")
    ap.add_argument("--kappa", nargs=2, metavar=("OUT_A", "OUT_B"),
                    help="tính Cohen's kappa giữa 2 output (dùng .progress.json)")
    args = ap.parse_args()

    if args.kappa:
        cohen_kappa(args.kappa[0], args.kappa[1])
        return
    if not (args.in_path and args.out_path):
        raise SystemExit("Cần --in và --out (hoặc --kappa A B).")
    run_verify(args.in_path, args.out_path, args.reviewer)


if __name__ == "__main__":
    main()