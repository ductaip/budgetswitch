#!/usr/bin/env bash
# Groq queue after the v1.2 decision (one key, retry/backoff, no rotation):
#   [done] gpt-oss run 3 v1.1, gpt-oss 3-shot v1.1
#   1. v1.2: gpt-oss-120b 0-shot run 1  ||  Qwen3.8-27B 0-shot   (different models, own quotas)
#   2. Qwen3.8-27B 3-shot v1.1 (resume)
set -u
PY=venv/bin/python; R=eval/v5/run_m3_v5.py
D12=data/vitool/pilot_eval_v1.2.jsonl; TR12=data/vitool/sft_train_v1.2.jsonl
while [ ! -s eval/v6/rewrite_meta_v12.json ]; do sleep 60; done   # v1.2 assembled
mkdir -p eval/v6/m3_v12
$PY $R --run 1 --data $D12 --train $TR12 --resume --out eval/v6/m3_v12/m3_gptoss_0shot_run1_v12.jsonl &
P1=$!
$PY $R --run 1 --model qwen/qwen3.8-27b --data $D12 --train $TR12 --resume --out eval/v6/m3_v12/m3_qwen27b_0shot_v12.jsonl
wait $P1
$PY $R --run 1 --model qwen/qwen3.8-27b --fewshot 3 --data data/vitool/pilot_eval_v1.1.jsonl \
   --train data/vitool/sft_train_v1.1.jsonl --resume --out eval/v6/m3_variants/m3_qwen27b_3shot_v6.jsonl
echo QUEUE DONE
