# ViTool: An Executable Vietnamese Tool-Use Benchmark

ViTool is an **executable** Vietnamese tool-use benchmark for studying
**per-step compute-mode allocation** in small language models. Every gold
trajectory is verified by **running it against a deterministic, stateful
environment** and checking the final state, and — since v1.2 — every gold
argument is checked to be **grounded** in the user's request.

**Current release: ViTool v1.2** (`data/vitool/vitool_v1.2.jsonl`)

- **373 trajectories**, 5 domains, **36 deterministic mock tools**:
  e-commerce 145, travel 64, information retrieval 56, banking 55,
  public services 53.
- Fixed split: **261 train / 112 eval** (eval: 43 easy, 26 medium, 43 hard;
  284 steps).
- Validation gates: schema → placeholder → structure → executable →
  **grounding (G3)** → semantic dedup. All 373 trajectories pass every gate.
- Harness for four compute modes (fast / think / search / defer), measured
  cost (tokens, FLOPs, latency), oracle and router analysis.

> The repository directory is named `budgetswitch` for historical reasons; the
> benchmark and paper are named **ViTool**.

---

## Version history: v1 → v1.1 → v1.2

| | v1 (`vitool_clean_final.jsonl`) | v1.1 | **v1.2** |
|---|---|---|---|
| trajectories | 375 | 373 | **373** |
| gates | schema, placeholder, structure, executable | + G3 grounding (enum values accepted as given) | + G3 with enum values required to be stated |
| eval trajectories with an ungrounded gold value (G3 v1.2 rule) | 63 / 112 | 34 / 112 | **0 / 112** |
| what changed | — | 198 requests rewritten to state missing values; 2 train trajectories with placeholder gold removed | 104 requests rewritten to state enumerated choices (payment method, platform, refund method, …) |

**Why.** v1 passed execution checks and a two-annotator review, yet 203 of 375
trajectories (54%; 63 of 112 eval) expected argument values the user never
gave — account numbers such as `1234567890`, quantities, price caps, passenger
names, voucher codes, payment methods. Models were penalised for not guessing
them. Gate G3 (`data/grounding_gate.py`) catches this: every gold value must be
stated in the request (after Vietnamese normalisation of dates, numbers and
money such as "500.000đ"; enumerated values through a Vietnamese phrase table,
e.g. "thanh toán khi nhận hàng" → `cod`), returned by an earlier tool call, or a
schema default.

**How the repair was done and checked.** Only the *request* was rewritten
(gpt-oss-20b, temperature 0); gold calls, tier, domain and split are unchanged
and asserted identical. A rewrite was accepted only if all gates passed and ≥ 70%
of the original words were kept. Two annotators checked the repair:

- G3 agrees with adjudicated human labels on 60/60 sampled values
  (inter-annotator κ = 0.76).
- 50 random rewrites: meaning preserved 50/50, none rated unnatural.
- Random spot checks for remaining invented values: 3–4 / 30 in v1.1 (all
  enumerated choices, which led to v1.2), **0 / 20 in v1.2**.

Earlier versions stay in the repository for reproducibility. Details:
`eval/v6/REPORT_v6.md`.

## Headline results (v1.2, 112 eval trajectories)

LoRA-tuned Qwen2.5-3B (adapter trained on the v1 train split; retraining on the
repaired split changed no mode significantly). TSR = every call correct **and**
the final environment state matches gold.

| Mode | TSR [95% CI] | SSR teacher-forced | SSR rollout | FLOPs vs M0 |
|---|---|---|---|---|
| M0 fast (direct call) | **0.688** [0.60, 0.77] | 0.873 | 0.722 | 1.00× |
| M1 think | 0.607 [0.51, 0.69] | 0.824 | 0.665 | 1.09× |
| M2 search | 0.527 [0.43, 0.62] | 0.768 | 0.532 | 0.66× |
| M3 defer (gpt-oss-120b, zero-shot) | 0.598 [0.51, 0.68] | 0.771 | – | 4.35× |
| Per-step oracle (M0–M3) | 0.804 [0.72, 0.87] | 0.944 | – | – |

- Direct calling is the best fixed mode; it beats thinking (p = 0.019) and
  retrieval (p < 0.001) on TSR; vs. deferral the TSR gap (−0.089) is not
  significant (p = 0.051), the step-level gap is (McNemar p < 0.001).
- The oracle shows +0.116 headroom; the best deployable router (learned,
  cross-validated) reaches 0.705.
- Repairing the data raised TSR by 0.14–0.25 for every mode
  (v1 → v1.2: M0 0.438 → 0.688) and more than doubled the routing headroom
  (0.053 → 0.116).
- Measured cost matters: the oracle saves 46.7% vs. always-think under assumed
  multipliers (1/3/5/15) but 21.7% under measured FLOPs.

All numbers: `eval/v6/metrics_v6.json`, `routers_v6.json`, `cost_v6.json`.

---

## Repository layout

```
data/vitool/
  vitool_v1.2.jsonl            # ViTool v1.2, all 373 trajectories (field "split")
  sft_train_v1.2.jsonl         # v1.2 train split (261)
  pilot_eval_v1.2.jsonl        # v1.2 eval split (112)
  vitool_v1.1.jsonl, *_v1.1    # v1.1 (kept for reproducibility)
  vitool_clean_final.jsonl     # v1 (375), sft_train.jsonl / pilot_eval.jsonl = v1 split
  tool_catalog.json            # 36 tool schemas
  tool_emb.npy                 # precomputed tool embeddings (M2)
env/mock_tools.py              # deterministic environment + executable verifier
data/
  build_dataset.py             # generation + all gates; --gate-report, --selftest
  grounding_gate.py            # Gate G3 (grounding) + Vietnamese phrase table
  dedup.py                     # semantic dedup (multilingual-e5)
tests/test_grounding_gate.py   # unit tests for G3
modes/                         # m0_fast, m1_think, m2_think_search, m3_defer
eval/v5/                       # measured-cost rerun harness (local modes, M3, routers)
eval/v6/                       # grounding audit, query repair, v1.1/v1.2 analysis, REPORT_v6.md
DATASHEET.md                   # datasheet (Gebru et al.)
```

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```
A single 16 GB GPU (e.g., T4) is enough for the 3B modes. The defer mode (M3)
calls the Groq API: copy `.env.example` to `.env` and set `GROQ_API_KEY`
(`.env` is git-ignored; never commit keys).

## Quick start

```python
from data.vitool.schema import load_trajectories
trajs = load_trajectories("data/vitool/pilot_eval_v1.2.jsonl")   # 112 eval trajectories
for s in trajs[0].steps:
    print(s.expected_tool, s.expected_params)
```

Check every gate (including G3) on any file, and run the tests:

```bash
python data/build_dataset.py --gate-report data/vitool/sft_train_v1.2.jsonl \
       data/vitool/pilot_eval_v1.2.jsonl --out gate_report.json
python -m pytest tests/test_grounding_gate.py -q
```

## Reproduce the experiments (v1.2)

Greedy decoding, seed 42.

```bash
# local modes M0/M1/M2, teacher-forced and rollout (GPU)
python eval/v5/run_local_modes_v5.py --task tf      --data data/vitool/pilot_eval_v1.2.jsonl \
       --lora <adapter_dir> --out eval/v6/raw/local_tf_v4adapter_v12.jsonl
python eval/v5/run_local_modes_v5.py --task rollout --modes M0,M1,M2 \
       --data data/vitool/pilot_eval_v1.2.jsonl --lora <adapter_dir> \
       --out eval/v6/raw/local_rollout_v4adapter_v12.jsonl
# defer mode M3 (Groq, one key, retry/backoff)
python eval/v5/run_m3_v5.py --run 1 --data data/vitool/pilot_eval_v1.2.jsonl \
       --out eval/v6/m3_v12/m3_gptoss_0shot_run1_v12.jsonl
# metrics, routers, report, figures
bash eval/v6/build_report_v6.sh
python eval/v6/figures_v6.py
```
Kaggle kernels used for the GPU runs: `eval/v6/kaggle/`.

### Compute modes

| Mode | Name | Assumed cost | Measured FLOPs vs M0 (v1.2) | Backend |
|---|---|---|---|---|
| M0 | fast (direct call) | 1× | 1.00× | Qwen2.5-3B + LoRA |
| M1 | think (CoT, then call) | 3× | 1.09× | Qwen2.5-3B + LoRA |
| M2 | search (retrieve tools, think, call) | 5× | 0.66× | Qwen2.5-3B + LoRA + e5 |
| M3 | defer (zero-shot larger model) | 15× | 4.35× | gpt-oss-120b (Groq) |

## Evaluating your own model

Implement a function that, given a query and history, returns
`{"tool_name": ..., "params": {...}}`, then score it with the step verifier and
`env/mock_tools.py::verify_final_outcome` (see `eval/v5/common_v5.py::score`).

## License

- **Code**: MIT — see [`LICENSE`](LICENSE).
- **Data** (`data/vitool/*.jsonl`, `data/vitool/tool_catalog.json`):
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — see
  [`data/vitool/LICENSE`](data/vitool/LICENSE). You may share and
  adapt the data with attribution (cite the paper below).

## Citation

```bibtex
@inproceedings{vitool2026,
  title     = {{ViTool}: An Executable Vietnamese Tool-Use Benchmark
               for Compute-Mode Allocation in Small Language Models},
  author    = {Phan, Duc Tai and Trinh Tran, Trung Duc},
  booktitle = {Proc. RIVF},
  year      = {2026}
}
```
