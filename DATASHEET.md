# Datasheet for ViTool v1.2

Following *Datasheets for Datasets* (Gebru et al., CACM 2021). Version: **v1.2**
(November 2026). Files: `data/vitool/vitool_v1.2.jsonl`, `sft_train_v1.2.jsonl`,
`pilot_eval_v1.2.jsonl`, `tool_catalog.json`.

## Motivation

- **Purpose.** Measure how a (small) language model should spend test-time
  compute per step of a tool-use task — direct call, chain of thought, tool
  retrieval, or deferral to a larger model — on executable Vietnamese tasks
  whose correctness can be checked by running them.
- **Gap.** Existing tool-use benchmarks are mostly English and verify calls by
  string or AST matching; no executable Vietnamese benchmark existed.
- **Creators / funding.** Duc Tai Phan and Trung Duc Trinh Tran, University of
  Transport Ho Chi Minh City. No external funding; compute from free Kaggle GPU
  quota and the Groq API free tier.

## Composition

- **Instances.** 373 *trajectories*. Each has a Vietnamese user request
  (`query`), an ordered list of gold tool calls (`steps`: `expected_tool`,
  `expected_params`, `difficulty`), the gold final environment state
  (`expected_final_state`), `domain`, `traj_id`, `source`, and in
  `vitool_v1.2.jsonl` the field `split`.
- **Domains.** e-commerce 145, travel 64, information retrieval 56,
  banking 55, public services 53. 36 tools (`tool_catalog.json`).
- **Difficulty** (structural). easy = 1 call; medium = several independent
  calls; hard = ≥ 3 calls with at least one data dependency (an ID returned by
  one call and consumed by a later one).
- **Splits.** 261 train / 112 eval, fixed (stratified, seed 42 in v1). Eval:
  43 easy, 26 medium, 43 hard; 284 steps. The split is inherited from v1; the
  2 trajectories removed in v1.1 were both in train.
- **Labels.** The gold calls are the labels; correctness is defined by strict
  argument match plus reproducing `expected_final_state` in the environment.
- **Synthetic data.** Requests and calls are LLM-drafted (see Collection);
  names, account numbers, IDs, addresses and codes are fictional. No real
  personal data. Some fictional values are deliberately conventional
  (e.g. account `1234567890`, customer `KH001`) and recur across splits.
- **Self-contained.** Yes. Tools are deterministic mock functions over a shared
  `WorldState` (`env/mock_tools.py`); no external API is needed to score.
- **Known issues.**
  - E-commerce is over-represented (44 of 112 eval trajectories) and has the
    longest dependency chains.
  - Requests that were repaired are longer and sometimes list-like.
  - Some gold oddities from v1 remain because gold calls were never re-derived
    (annotators flagged 4 of 50 sampled rewrites: an unrelated extra step, an
    unrelated expression, an implausible price cap).
  - The grounding check is automatic; its phrase table for enumerated values is
    hand-written (listed in `eval/v6/REPORT_v6.md`, table T7).

## Collection process

1. **Drafting (v1).** An LLM (Llama-3.3-70B via Groq) drafted requests and
   call sequences per domain and difficulty from the tool catalog.
2. **Gates (v1).** schema (Gate 1) → placeholder (Gate 1.5) → structure →
   dependent-ID resolution from real tool outputs → executable (Gate 2, must
   reach a valid final state) → semantic dedup (multilingual-e5, cos > 0.93).
   Human review of 51 stratified trajectories: 45 accepted, 6 minor edits
   (agreement 92.2%, Gwet's AC1 0.91).
3. **Grounding audit (v1 → v1.1).** Gate G3 (`data/grounding_gate.py`) requires
   every gold value to be stated in the request, returned by an earlier call,
   or a schema default. Validation: two annotators labelled 60 sampled values
   (κ = 0.76); after adjudicating 8, G3 matched 60/60. Under G3, 203/375 v1
   trajectories had at least one ungrounded value.
4. **Repair v1.1.** 198 requests rewritten by gpt-oss-20b (temperature 0) to
   state the missing values; only the request changed. Accepted only if all
   gates passed and ≥ 70% of the original words were kept. Two train
   trajectories with placeholder gold values (`order_id = "id_đơn_hàng"`,
   `seat_class = ""`) were removed.
5. **Repair v1.2.** G3 tightened: an enumerated value (payment method,
   platform, refund method, seat class, …) counts only if the request names it
   (phrase table). 104 further requests rewritten with the same model and
   prompt; all 104 accepted automatically.
6. **Human checks.** 50 random rewrites: meaning preserved 50/50 (both
   annotators), none rated unnatural. Spot checks for remaining invented
   values: 3–4 / 30 in v1.1, 0 / 20 in v1.2.

All prompts, model IDs, acceptance criteria, logs and per-trajectory old/new
requests are in `eval/v6/` (`rewrite_meta*.json`, `query_diffs*.jsonl`,
`rewrite_stats*.json`).

## Preprocessing

Dates use `dd/mm/yyyy`, money is in VND, phone numbers use `0…`. Dependent IDs
(flight, hotel, product, cart, order, booking) are filled from the environment's
real outputs, so gold calls are always executable. No other preprocessing; the
raw v1 files are kept (`vitool_clean_final.jsonl`, `sft_train.jsonl`,
`pilot_eval.jsonl`, and the v1.1 files).

## Uses

- **Intended.** Evaluating tool-calling models in Vietnamese; studying
  per-step compute allocation (mode routing, cascades, deferral) with measured
  cost; studying the effect of grounding errors in synthetic data.
- **Not intended.** Measuring performance on live services (tools are mocks);
  drawing conclusions about real banking, government or commerce systems;
  training models to fabricate personal data.
- **Caveat on comparisons.** v1, v1.1 and v1.2 share eval trajectories but not
  requests; report the version. Scores on v1 are depressed by ungrounded gold
  values (e.g. M0 TSR 0.438 on v1 vs 0.688 on v1.2).

## Distribution

GitHub: https://github.com/ductaip/budgetswitch. Data CC BY 4.0, code MIT.

## Maintenance

Maintained by the authors via GitHub issues. Fixes will be released as new
versions (v1.3, …) with the previous files kept and the change documented here
and in `README.md`.
