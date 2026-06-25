# ViTool: An Executable Vietnamese Tool-Use Benchmark

ViTool is the first **executable** Vietnamese tool-use benchmark for studying
**per-step compute-mode allocation** in small language models. Unlike
string/AST-matching benchmarks, every gold trajectory is verified by **running
it against a deterministic, stateful environment** and checking the resulting
final state.

- **375 trajectories** across 5 domains (banking, e-commerce, information
  retrieval, public services, travel) and **36 deterministic mock tools**.
- Stratified by structural difficulty: easy (142), medium (90), hard (143).
- Four-gate validation (schema → placeholder → structure → executable).
- Inter-annotator reliability on a stratified 51-sample subset:
  raw agreement 92.2%, Gwet's AC1 = 0.91.
- Evaluation harness for four compute modes (fast / think / search / defer)
  and an offline **oracle-gap** analysis.

> Anonymized release for peer review. Author and institution details are
> withheld during the review period.
>
> **Note on naming:** the repository directory is named `budgetswitch` for
> historical reasons; the benchmark and paper are named **ViTool**. They refer
> to the same project.

---

## Repository layout

```
data/vitool/
  vitool_clean_final.jsonl   # the benchmark (375 trajectories)
  tool_catalog.json          # 36 tool schemas
  tool_emb.npy               # precomputed tool embeddings (for M2)
env/
  mock_tools.py              # deterministic environment + executable verifier
data/
  build_dataset.py           # generation + 4-gate pipeline
  dedup.py                   # semantic dedup (multilingual-e5)
  verify_cli.py              # human-verification CLI + Cohen's kappa
scripts/
  split_data.py              # 70/30 stratified split (seed 42)
  rerun_m3.py                # re-run the defer column with one stable model
pilot/
  simulate_modes.py          # run the 4 modes per step
  run_pilot.py               # end-to-end pilot runner
modes/                        # m0_fast, m1_think, m2_think_search, m3_defer
eval/
  oracle.py                  # oracle-gap report
  figures.py                 # Fig 1 (heatmap) + Fig 2 (Pareto)
  run_baselines.py           # routing baselines vs oracle
DATASHEET.md                  # datasheet for datasets (Gebru et al.)
```

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install torch transformers peft bitsandbytes \
            sentence-transformers openai jsonschema datasets \
            matplotlib numpy
```
A CUDA GPU (e.g., a single T4, 16 GB) is sufficient for the 3B modes.
The defer mode (M3) calls an external API; see below.

## Quick start: load the benchmark

```python
from data.vitool.schema import load_trajectories
trajs = load_trajectories("data/vitool/vitool_clean_final.jsonl")
print(len(trajs), "trajectories")
for s in trajs[0].steps:
    print(s.expected_tool, s.expected_params)
```

## Reproduce the experiments

The full pipeline is deterministic given seed 42.

```bash
# 1) 70/30 stratified split -> SFT-train (263) / held-out eval (112)
python scripts/split_data.py

# 2) precompute tool embeddings for the search mode (M2)
python -c "from modes.m2_think_search.m2_think_search import build_tool_embeddings; \
           build_tool_embeddings('data/vitool/tool_catalog.json','data/vitool/tool_emb.npy')"

# 3) fine-tune the 3B LoRA adapter on the TRAIN split only
#    (see the training notebook; produces models/tool_use_lora)

# 4) run the pilot (4 modes per step) on the 112 held-out trajectories
export GROQ_API_KEY=...        # defer mode (M3) = gpt-oss-120b via Groq
python pilot/run_pilot.py \
    --data data/vitool/pilot_eval.jsonl \
    --lora models/tool_use_lora \
    --emb  data/vitool/tool_emb.npy \
    --out  pilot/pilot_results.jsonl

# 5) oracle-gap analysis + figures
python eval/oracle.py     --in pilot/pilot_results.jsonl --out eval/oracle_report.json
python eval/figures.py    --in eval/oracle_report.json   --outdir eval/figs
python eval/run_baselines.py --results pilot/pilot_results.jsonl \
                             --eval data/vitool/pilot_eval.jsonl --out eval/baselines.json
```

### Compute modes

| Mode | Name | Relative cost | Backend |
|---|---|---|---|
| M0 | fast (direct call) | 1× | 3B SLM + LoRA |
| M1 | think (CoT then call) | 3× | 3B SLM + LoRA |
| M2 | search (retrieve tools, then think) | 5× | 3B SLM + LoRA + e5 |
| M3 | defer (zero-shot, larger model) | 15× | gpt-oss-120b (Groq) |

The defer model is from a **different model family** than the data generator
to avoid contamination, and is run **zero-shot** to reflect realistic
deployment (no per-task fine-tuning of a large model).

## Headline result

A per-step oracle reaches TSR **0.634 at 1.27× cost**, versus **0.57 at 3×**
for always-think — a **57.7% compute reduction at higher quality**. Fixed
policies degrade monotonically as cost rises (0.588 → 0.570 → 0.535 → 0.486
for M0 → M1 → M2 → M3), and no realistic router closes the gap to the oracle,
motivating learned per-step routing.

See `DATASHEET.md` for dataset documentation and intended uses/limitations.

## Evaluating your own model

Implement a function that, given a query and history, returns
`{"tool_name": ..., "params": {...}}`, then reuse the executable verifier in
`env/mock_tools.py` (`verify_final_outcome`) or the per-step verifier to score
trajectories. A minimal adapter example is in `pilot/run_pilot.py`.

## License

- Data: [TODO — proposed CC BY 4.0]
- Code: [TODO — proposed MIT]

## Citation

```bibtex
@inproceedings{vitool2026,
  title     = {{ViTool}: An Executable Vietnamese Tool-Use Benchmark
               for Compute-Mode Allocation in Small Language Models},
  author    = {Anonymous},
  booktitle = {Under review},
  year      = {2026}
}
```
(Author and venue details will be added upon acceptance.)