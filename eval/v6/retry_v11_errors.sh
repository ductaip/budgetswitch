while pgrep -f "retry_api_errors|pilot_eval_v1.2.jsonl" >/dev/null; do sleep 60; done
P=venv/bin/python; R=eval/v5/run_m3_v5.py; D=data/vitool/pilot_eval_v1.1.jsonl; T=data/vitool/sft_train_v1.1.jsonl
$P $R --run 2 --data $D --train $T --resume --out eval/v6/m3/m3_run2_v6.jsonl
$P $R --run 3 --data $D --train $T --resume --out eval/v6/m3/m3_run3_v6.jsonl
$P $R --run 1 --fewshot 3 --data $D --train $T --resume --out eval/v6/m3_variants/m3_gptoss_3shot_v6.jsonl
echo DONE
