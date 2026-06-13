"""
scripts/split_data.py — Chia ViTool thành SFT-train / pilot-eval (v10.0).

VÌ SAO BẮT BUỘC: LoRA cũ (W3) SFT trên bộ 200 cũ, mà 172 cái trong đó nằm TRONG
vitool_clean_final.jsonl → pilot bằng LoRA cũ trên full 375 = train-on-test
(reviewer giết ngay). Fix: split sạch → SFT MỚI trên train 70% → pilot CHỈ trên
eval 30% held-out.

Split STRATIFIED theo (difficulty × domain) để eval 30% giữ đúng tỉ lệ các tầng.
Seed cố định (42) → reproducible, ghi vào paper.

Dùng:
  python scripts/split_data.py \
      --in data/vitool/vitool_clean_final.jsonl \
      --train-out data/vitool/sft_train.jsonl \
      --eval-out  data/vitool/pilot_eval.jsonl \
      --eval-ratio 0.3
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.vitool.schema import load_trajectories, save_trajectories  # noqa: E402

ORDER = {"easy": 0, "medium": 1, "hard": 2}


def traj_level(x) -> str:
    return max((s.difficulty.value for s in x.steps), key=lambda d: ORDER[d])


def split(in_path: str, train_out: str, eval_out: str, eval_ratio: float, seed: int) -> None:
    rng = random.Random(seed)
    trajs = load_trajectories(in_path)

    groups = defaultdict(list)
    for x in trajs:
        groups[(traj_level(x), x.domain)].append(x)

    train, evals = [], []
    for key in sorted(groups):  # sorted để deterministic
        pool = groups[key]
        rng.shuffle(pool)
        n_eval = max(1, round(len(pool) * eval_ratio)) if len(pool) > 1 else 0
        evals += pool[:n_eval]
        train += pool[n_eval:]

    # an toàn: không traj nào ở cả 2 bên
    tid_train = {x.traj_id for x in train}
    tid_eval = {x.traj_id for x in evals}
    assert not (tid_train & tid_eval), "LEAK: trajectory xuất hiện ở cả train lẫn eval!"

    save_trajectories(train, train_out)
    save_trajectories(evals, eval_out)

    def report(name, arr):
        c = Counter(traj_level(x) for x in arr)
        d = Counter(x.domain for x in arr)
        steps = sum(len(x.steps) for x in arr)
        print(f"{name}: {len(arr)} traj ({steps} steps) | tier {dict(c)} | domain {dict(d)}")

    print(f"Seed={seed}, eval_ratio={eval_ratio} (stratified theo tier×domain)")
    report("TRAIN (SFT)", train)
    report("EVAL  (pilot, held-out)", evals)
    print("✅ Không leak (train ∩ eval = ∅). Ghi seed vào paper §4 cho reproducibility.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/vitool/vitool_clean_final.jsonl")
    ap.add_argument("--train-out", default="data/vitool/sft_train.jsonl")
    ap.add_argument("--eval-out", default="data/vitool/pilot_eval.jsonl")
    ap.add_argument("--eval-ratio", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    split(args.in_path, args.train_out, args.eval_out, args.eval_ratio, args.seed)


if __name__ == "__main__":
    main()