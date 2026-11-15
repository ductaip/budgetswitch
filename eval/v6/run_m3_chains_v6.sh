#!/usr/bin/env bash
# Step 4 + 6 M3 runs on v1.1 eval. One Groq key (GROQ_API_KEY), retry/backoff inside run_m3_v5.py.
# Usage: bash eval/v6/run_m3_chains_v6.sh gptoss|qwen
set -u
PY=venv/bin/python; R=eval/v5/run_m3_v5.py
D=data/vitool/pilot_eval_v1.1.jsonl; TR=data/vitool/sft_train_v1.1.jsonl
if [ "$1" = gptoss ]; then
  for k in 1 2 3; do
    $PY $R --run $k --data $D --train $TR --resume --out eval/v6/m3/m3_run${k}_v6.jsonl || exit 1
  done
  $PY $R --run 1 --fewshot 3 --data $D --train $TR --resume --out eval/v6/m3_variants/m3_gptoss_3shot_v6.jsonl
else
  M=qwen/qwen3.8-27b
  $PY $R --run 1 --model $M --data $D --train $TR --resume --out eval/v6/m3_variants/m3_qwen27b_0shot_v6.jsonl || exit 1
  $PY $R --run 1 --model $M --fewshot 3 --data $D --train $TR --resume --out eval/v6/m3_variants/m3_qwen27b_3shot_v6.jsonl
fi
