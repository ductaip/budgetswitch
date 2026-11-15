# ViTool camera-ready — experiment report v5

Status: P0 items 1–7 done. P1 item 8 is partly done (gpt-oss 3-shot finished;
Qwen3.8-27B 0-/3-shot and gpt-oss 3-shot with clean demos still running on a
single rate-limited Groq key). Backbones (item 9) are reported separately in
`eval/v5/journal/` and are not part of this report.

**Headline metric: TSR.** A trajectory counts as a success only if every step
is strictly correct and running the model's calls in order reproduces the gold
final environment state. Every SSR below is labelled with how its history was
built: `SSR_tf` uses the gold history (teacher-forced, the v4 protocol);
`SSR_rollout` uses the model's own history. 95% Wilson CIs are in brackets.
112 trajectories / 284 steps (held-out split).

All numbers come from files in `eval/v5/` and can be regenerated with:
`analyze_v5.py` → `relaxed_v5.py` → `routers_v5.py` → `figures_v5.py`.
No v4 artifact was modified.

---

## 0. Findings that change the paper's story

1. **v4 TSR already measures task success.** Under rollout, where each model
   uses its own previous calls and observations, TSR equals teacher-forced TSR
   for every one of the 112 trajectories (M0 0.438, M1 0.420). The reason: a
   trajectory only succeeds if every step is right, and then the model's own
   history is identical to the gold history. **SSR, however, is inflated by
   gold history:** M0 0.588 teacher-forced vs 0.468 rollout (−12 pts); M1 0.570
   vs 0.433 (−14 pts).
2. **Many gold parameters are invented.** In 209/375 trajectories (56%), at
   least one gold value cannot be derived from the query, an earlier
   observation, or a tool enum. On the eval split this is 61/112 trajectories,
   and **39/43 hard trajectories**. This explains most of the hard-tier TSR of
   about 0.05. On trajectories with no invented values, M0 TSR is 0.90 (n=51,
   mostly easy tier; see §3c).
3. **The cost ordering assumed in v4 (1/3/5/15) does not hold.** Measured in
   tokens or FLOPs, M2 is the *cheapest* mode (0.64× M0), because it puts only
   10 retrieved tools in the prompt; M1 costs only 1.10× M0. The claim "oracle
   saves X% vs always-think (M1)" survives but shrinks: **51% (assumed costs)
   → 22% (tokens) → 17% (FLOPs)**. Against always-M0, the oracle saves 14%
   (tokens) / 9% (FLOPs) while raising TSR from 0.438 to 0.482.
4. **No deployable router clearly beats always-M0.** Best cross-validated
   learned router: TSR 0.446 vs 0.438. The cascade that escalates on
   execution errors fires on only 1% of steps.
5. **The relaxed metric adds no truly-correct steps.** Format normalization
   rescues 0 steps. "Execution equivalence" rescues 15–22 steps per mode, but
   only because the mock environment ignores most optional parameters; a
   manual check of 20 cases found 1 correct.

---

## 1. Audit of the v4 evaluation protocol

**Gold history (teacher-forced) or model's own history?** Gold history.
`pilot/simulate_modes.py:120-135` replays the gold prefix through the mock env:

```python
for s in traj.steps[:step_idx]:
    call = {"tool_name": s.expected_tool, "params": s.expected_params}
    obs = _env_run(s.expected_tool, dict(s.expected_params), world)
    hist.append((call, obs))
```

This is called at `simulate_modes.py:154` (M0–M2) and `scripts/rerun_m3.py:92`
(M3). 0/284 gold steps fail during replay. Every step receives the whole
trajectory query (`traj.query`).

**Were raw outputs or parsed calls saved for v4?** No. `ModeResult`
(`simulate_modes.py:53-60`) keeps only `mode, success, cost, latency_ms,
completion_tokens, api_cost_usd`. v4 artifacts: `pilot_results.jsonl` (before
the M3 rerun), `pilot_results_v4.jsonl`, `eval/oracle_report_v4.json`,
`eval/error_analysis.json`. The 284-step run was the Kaggle kernel
`ductaiphan/vitool-pilot`; its output is byte-identical to
`pilot_results.jsonl`. It also contains the v4 adapter `tool_use_lora_v2`
(sha256 `c704f9d5…`), which v5 reuses.

**What is compared for `success`?** The call itself, not the env state.
`verifiers/universal/schema_verifier.py:167-187`:
- tool name: case-insensitive;
- params: jsonschema validation against the gold tool's schema;
- every gold key must be present: strings compared after `strip().lower()`,
  other types with `==`;
- `fuzzy_keys`: only separators and currency marks are normalized; day/month
  order is not;
- extra params are allowed.

**Other protocol facts found during the audit**
- M0–M2 prompts list only tool names and descriptions. The M3 prompt also lists
  param names, types, required flags and enums
  (`m0_fast.py:92-95` vs `m3_defer.py:160-180`).
- M2 retrieval ignores history (`m2_think_search.py:58-65`), so every step of a
  trajectory gets the same top-10 tools.
- Why M0 `completion_tokens` was always 0: `m0_fast_generate` never sets it,
  and the harness defaults to 0.
- v4 latency was `time.time()` around generate + parse + verify (M2 also
  included retrieval; M3 included network time and backoff). The 3 negative M3
  latencies come from wall-clock adjustments.
- The v4 log has 117 swallowed M3 exceptions (Groq daily quota). They are all
  in the original M3 column, which `rerun_m3.py` later replaced entirely; v4
  M0–M2 are unaffected.
- `eval/oracle.py` labels **step-level** success as "TSR". The v4 figure "oracle
  saves 57.7% vs M1" is step-level.
- Difficulty is a trajectory property: all steps of a trajectory share one tier.

---

## 2. v5 rerun with full logging (teacher-forced, same protocol as v4)

M0/M1/M2: Qwen2.5-3B-Instruct in 4-bit nf4 + the v4 LoRA (same sha). Greedy
decoding, seed 42, 2× Tesla T4 (Kaggle). Package versions differ from v4
(transformers 5.16.1, torch 2.11; logged in `raw/local_tf_v5.meta.json`).
M3: `openai/gpt-oss-120b` on Groq, same prompt as v4, temperature 0,
reasoning_effort=low, run 3 times.

| Mode | SSR_tf v4 | SSR_tf v5 | Per-step agreement | Flips (v4✓v5✗ / v4✗v5✓) | parse_ok | exec_error |
|---|---|---|---|---|---|---|
| M0 | 0.588 | 0.588 | 1.000 | 0 / 0 | 1.000 | 0.011 |
| M1 | 0.570 | 0.570 | 1.000 | 0 / 0 | 0.993 | 0.028 |
| M2 | 0.535 | 0.535 | 1.000 | 0 / 0 | 0.993 | 0.046 |
| M3 (run 1) | 0.486 | 0.507 | 0.951 | 4 / 10 (McNemar p=0.18) | 0.968 | 0.046 |

M0–M2 reproduce v4 exactly, step by step. M3 repeats:
- SSR_tf: 0.507 / 0.496 / 0.496 (mean 0.500 ± 0.006);
- TSR: 0.393 / 0.375 / 0.375 (mean 0.381 ± 0.010);
- pairwise step agreement between runs: 0.95–0.97; 0 API errors.

Using any of the three M3 runs gives the same oracle TSR (0.491).

About 9 M3 parse failures per run are clarification questions (for example
"please provide your account number") on steps whose gold value is invented
(see §3c).

---

## 3. Trajectory-level metrics (`metrics_v5.json`)

### 3a. Overall, teacher-forced

| | TSR | SSR_tf |
|---|---|---|
| M0 | **0.438** [0.35, 0.53] | 0.588 [0.53, 0.64] |
| M1 | 0.420 [0.33, 0.51] | 0.570 [0.51, 0.63] |
| M2 | 0.384 [0.30, 0.48] | 0.535 [0.48, 0.59] |
| M3 | 0.393 [0.31, 0.49] | 0.507 [0.45, 0.56] |
| Oracle (cheapest correct, assumed cost) | 0.491 [0.40, 0.58] | 0.648 [0.59, 0.70] |

The state check changes one result: M2 loses `public_services_0003`. Strict
matching ignores case ("Cấp giấy phép…" vs "cấp giấy phép…"), but the stored
env state does not. For all other modes TSR (strict + state) equals strict-only
TSR.

**By tier**

| TSR / SSR_tf | M0 | M1 | M2 | M3 | Oracle |
|---|---|---|---|---|---|
| easy (n=43) | 0.93 / 0.93 | 0.84 / 0.84 | 0.79 / 0.81 | 0.86 / 0.86 | 0.93 / 0.93 |
| medium (n=26) | 0.27 / 0.53 | 0.31 / 0.53 | 0.31 / 0.55 | 0.23 / 0.47 | 0.38 / 0.62 |
| hard (n=43) | 0.05 / 0.53 | 0.07 / 0.52 | 0.02 / 0.47 | 0.02 / 0.44 | 0.12 / 0.59 |

**By domain** (TSR; n = number of trajectories)

| | n | M0 | M1 | M2 | M3 | Oracle |
|---|---|---|---|---|---|---|
| banking | 16 | 0.56 | 0.63 | 0.63 | 0.56 | 0.63 |
| ecommerce | 44 | 0.18 | 0.20 | 0.16 | 0.18 | 0.25 |
| info_retrieval | 17 | 0.71 | 0.71 | 0.65 | 0.59 | 0.77 |
| public_services | 16 | 0.75 | 0.63 | 0.69 | 0.56 | 0.75 |
| travel | 19 | 0.42 | 0.32 | 0.21 | 0.42 | 0.47 |
| **domain macro-avg** | | 0.52 | 0.50 | 0.47 | 0.46 | 0.57 |

**Paired tests vs M0**

| | SSR_tf, exact McNemar (M0 wins / other wins, p) | TSR difference, paired bootstrap by trajectory (10k) |
|---|---|---|
| M1 | 13 / 8, p=0.38 | −0.018 [−0.071, +0.036], p=0.63 |
| M2 | 21 / 6, p=0.006 | −0.054 [−0.107, 0.000], p=0.07 |
| M3 | 35 / 12, p=0.001 | −0.045 [−0.089, 0.000], p=0.06 |

### 3b. End-to-end rollout (M0, M1)

| | SSR_rollout | TSR rollout | TSR teacher-forced | Per-trajectory agreement | Mean SSR_tf − SSR_rollout |
|---|---|---|---|---|---|
| M0 | 0.468 [0.41, 0.53] | 0.438 [0.35, 0.53] | 0.438 | **112/112** | +0.120 |
| M1 | 0.433 [0.38, 0.49] | 0.420 [0.33, 0.51] | 0.420 | **112/112** | +0.137 |

Hard-tier rollout TSR: M0 0.05, M1 0.07. "Final state matches" on its own is
0.58 for both modes, because read-only trajectories match trivially; do not use
it alone.

### 3c. Grounding audit (`grounding_v5.py`, all 375 trajectories)

A gold value is **grounded** if it appears in the normalized query (also
checking diacritic-free forms, date formats, and number forms such as
"1,5 triệu" or "500k"), appears in an earlier step's observation, is a tool
enum value, or is a boolean. A **lenient** sensitivity check also accepts an
implicit `quantity`/`passengers` = 1 and currency/language/city names mapped to
codes (for example "đô la Mỹ" → USD).

| | Steps ungrounded (strict / lenient) | Trajectories ungrounded (strict / lenient) |
|---|---|---|
| all 375 trajectories | 420 / 338 of 1003 | **209 / 191** of 375 |
| train | 301 / 241 of 719 | 148 / 133 of 263 |
| eval | 119 / 97 of 284 | **61 / 58** of 112 |
| eval easy | 5 / 4 of 43 | 5 / 4 of 43 |
| eval medium | 23 / 23 of 53 | 17 / 17 of 26 |
| eval hard | 91 / 70 of 188 | **39 / 37** of 43 |

By domain (all splits, strict): ecommerce 115/146 trajectories, travel 36/65,
info_retrieval 21/56, public_services 19/53, banking 18/55.

Most frequent ungrounded parameters: `add_to_cart.quantity`,
`search_product.max_price`, `book_flight.passenger_names`,
`apply_voucher.voucher_code`, `checkout.shipping_address`. Most frequent values:
"1", "2", "1234567890", "500000", "Lê Văn C", "Nguyễn Văn A".

The full list is in `grounding_ungrounded_v5.jsonl`. A manual check of 18
random flagged steps found 13 with truly invented values (IDs, vouchers,
addresses, prices, names, a text to translate), 4 that can be inferred
(implicit quantity/passengers 1, currency codes), and 1 unclear (a hotel
checkout date). The lenient column covers the inferable kind.

**Eval TSR under three views (strict audit)**

| TSR (SSR_tf) | (a) all 112 | (b) grounded only, n=51 | (c) clarification counts as correct on ungrounded steps |
|---|---|---|---|
| M0 | 0.438 (0.588) | 0.902 (0.917) | 0.438 (0.588) |
| M1 | 0.420 (0.570) | 0.824 (0.875) | 0.420 (0.570) |
| M2 | 0.384 (0.535) | 0.784 (0.833) | 0.384 (0.535) |
| M3 | 0.393 (0.507) | 0.824 (0.847) | **0.438** (0.528), 6 credited steps |
| Oracle | 0.491 (0.648) | 0.941 (0.958) | 0.491 (0.648) |

Caveats:
- View (b) is not tier-balanced: its 51 trajectories are 38 easy, 9 medium and
  only 4 hard.
- With the lenient audit, (b) has n=54 and M0 TSR 0.870
  (`metrics_v5.json → grounding_views.lenient_audit`).
- Only M3 ever asks clarification questions; the fine-tuned local modes always
  output a call.

---

## 4. Relaxed / execution-equivalence metric (`relaxed_v5.json`)

| | SSR_tf strict | SSR_tf relaxed | TSR strict | TSR relaxed | Rescued by normalization only | Rescued by exec-equivalence only |
|---|---|---|---|---|---|---|
| M0 | 0.588 | 0.655 | 0.438 | 0.482 | 0 | 19 |
| M1 | 0.570 | 0.644 | 0.420 | 0.455 | 0 | 21 |
| M2 | 0.535 | 0.613 | 0.393 | 0.429 | 0 | 22 |
| M3 | 0.507 | 0.560 | 0.393 | 0.420 | 0 | 13 |

(TSR in this table is strict-only, without the state check; it equals the
headline TSR in §3a except for M2, 0.393 vs 0.384.)

Steps that no mode solves strictly: 100. Of these, 20 become solvable under the
relaxed metric.

**Manual check of the 20 sampled strict-fail / relaxed-pass cases**
(`relaxed_examples_v5.jsonl`): 14 omit an optional gold param, 4 use a wrong
value (for example `max_price` 800000 vs 500000), 1 misspells a param key, and
**1 is correct**. All gains come from execution equivalence, and they happen
because the mock tools ignore `max_price`, `platform`, `shipping_address`,
`guests`, `room_type` and similar params, so the observations come out
identical. **Recommendation:** keep strict as the metric; mention the relaxed
check as a robustness test that found no format false-negatives.

---

## 5. Measured cost models (`cost_v5.json`, `routers_v5.json`)

Mean per step:

| | Prompt tok | Completion tok | FLOPs | Rel. to M0: assumed / tokens / FLOPs | Latency median | Hardware |
|---|---|---|---|---|---|---|
| M0 | 502 | 38.6 | 3.34e12 | 1 / 1.00 / 1.00 | 5.5 s | 2×T4, 4-bit |
| M1 | 527 | 66.2 | 3.66e12 | 3 / 1.10 / 1.10 | 8.9 s | 2×T4, 4-bit |
| M2 | 288 | 57.1 | 2.15e12 (+e5) | 5 / 0.64 / 0.64 | 7.9 s | 2×T4, 4-bit |
| M3 | 1371 | 105.1 | 1.51e13 | 15 / 2.73 / 4.51 | 0.48 s | Groq API (client-side clock) |

- FLOPs = 2·N_active·(prompt + completion). N_active: 3.09e9 for Qwen2.5-3B;
  5.1e9 for gpt-oss-120b (model card: "117B parameters with 5.1B active
  parameters"). M2 also adds 2·278e6·25 FLOPs for the e5 query embedding.
- M3 cost: $0.081 for 284 calls at Groq list prices.
- Latency is measured with `perf_counter` around `generate` only. It is not
  comparable across the two hardware platforms.

**Policies × cost models** (cost relative to always-M0)

| Policy | TSR | SSR_tf | assumed | tokens | FLOPs |
|---|---|---|---|---|---|
| Fixed M0 | 0.438 | 0.588 | 1.00 | 1.00 | 1.00 |
| Fixed M1 | 0.420 | 0.570 | 3.00 | 1.10 | 1.10 |
| Fixed M2 | 0.384 | 0.535 | 5.00 | 0.64 | 0.64 |
| Fixed M3 | 0.393 | 0.507 | 15.00 | 2.73 | 4.51 |
| Random (uniform) | 0.410 | 0.550 | 6.00 | 1.37 | 1.81 |
| Length router | 0.420 | 0.567 | 3.13 | 0.89 | 0.89 |
| Difficulty escalation (uses gold tier) | 0.438 | 0.528 | 10.64 | 2.17 | 3.39 |
| Oracle, cheapest by assumed cost | 0.491 | 0.648 | 1.46 | 1.05 | 1.10 |
| Oracle, cheapest by tokens | 0.482 | 0.648 | 3.56 | 0.86 | 0.91 |
| Oracle, cheapest by FLOPs | 0.482 | 0.648 | 3.56 | 0.86 | 0.91 |
| Cascade M0→M1 on parse/exec error | 0.446 | 0.592 | 1.03 | 1.01 | 1.01 |
| Cascade M0→M3 on parse/exec error | 0.438 | 0.588 | 1.16 | 1.03 | 1.05 |
| Best learned router (pre-generation, λ=0.05, FLOPs) | 0.446 | 0.581 | 2.42 | 0.96 | 0.97 |

**Answer to the key question.** "Oracle saves X% vs always-think (M1)" holds
under measured costs, with X = **21.8% (tokens) / 16.9% (FLOPs)**, down from
51.2% under the assumed 1/3/5/15 (and v4's step-level 57.7%).
- Under measured costs the oracle mostly picks M0 (43% of steps) or M2 (54%),
  because M2 is the cheapest mode.
- The oracle's gain over always-M0 is +0.044 TSR at 0.86× the tokens.
- Pareto front (TSR vs FLOPs): M2 → learned router → oracle; M0, M1 and M3 are
  dominated. See `fig2_pareto_v5.pdf`.

---

## 6. Execution-feedback cascade

- Trigger rate: **1.1%** of steps (M0 output fails to parse or execute).
- When it fires, M0 is always wrong (precision 1.0), but it catches only
  **2.6%** of M0's errors.
- After escalation, M1 is right on 1/3 of triggered steps and M3 on 0/3.
- Result: M0→M1 gives TSR 0.446 at 1.01× FLOPs; M0→M3 gives 0.438 at 1.05×.

Why so weak: the mock env rejects only unknown tools, missing required params,
and unknown IDs. Wrong values in optional params execute without error.

The "cascade + learned" variant adds M0's `parse_ok`/`exec_error` as features.
Best TSR 0.446 (τ=0.3), at a cost higher than M0 (1.32× tokens). Adding the
execution features leaves AUC unchanged (M0 success AUC 0.795 → 0.797).

## 7. Learned router (GroupKFold(5) by trajectory, 284 eval steps only)

Features: char TF-IDF of the query, multilingual-e5 query embedding, step
index, previous-step tool predicted by M0, and (post-M0 variant) M0's
parse/exec signals. Per-mode logistic regression; decisions use fold-train mean
cost per mode, with λ swept.

- Cross-validated AUC per mode: M0 0.80, M1 0.74, M2 0.79, M3 0.76.
- Best TSR: **0.446** (vs always-M0 0.438), at 0.96× FLOPs.
- The cheaper Pareto points trade TSR for cost along the line toward M2
  (λ=0.5: 0.411 TSR at 0.66×).
- Conclusion: the router learns which steps will fail (largely tier and domain),
  not which mode will rescue them. Only 17/284 steps are failed by M0 and solved
  by another mode (13 in v4; the difference comes from the M3 rerun).

## 8. Fairer deferral (P1)

| Variant | TSR | SSR_tf | Prompt / completion tokens | $ / 284 steps | Status |
|---|---|---|---|---|---|
| gpt-oss-120b 0-shot (= M3 run 1) | 0.393 | 0.507 | 1371 / 105 | 0.081 | done |
| gpt-oss-120b 3-shot (seed-42 TRAIN demos) | 0.402 [0.32, 0.49] | 0.514 | 1774 / 109 | 0.099 | done |
| gpt-oss-120b 3-shot, clean demos (no ungrounded values) | — | — | — | — | pending |
| Qwen3.8-27B 0-shot | — | — | — | — | pending (158/284) |
| Qwen3.8-27B 3-shot | — | — | — | — | pending |

- The original 3-shot demos include an invented account number ("1234567890").
  The clean-demo variant checks whether few-shot gains come from copying the
  dataset's conventions.
- The alternative model is Qwen3.8-27B (dense, 27B params, Groq list price
  $0.80 / $4.00 per 1M input/output tokens).
- Llama-3.3-70B is no longer on Groq, and it generated the gold data, so it
  would have been contaminated anyway.

---

## Deviations from the brief, and why

1. **M3 calls pass `seed=42`.** v4 did not. Results still vary across runs (§2).
2. **The headline TSR adds an env final-state check to strict step success.**
   On these data it changes one trajectory (M2; a case difference that strict
   matching ignores).
3. **Step 4, execution equivalence also requires an identical observation.**
   For read-only tools the state delta is always empty, so a delta-only check
   would accept any executable call.
4. **Step 4, normalized matching skips the JSON-schema check** that strict uses.
5. **Router TSR is computed on teacher-forced step outcomes**, composed per
   trajectory: all four modes exist only under that protocol. §3b shows that
   teacher-forced TSR equals rollout TSR for M0 and M1.
6. **Groq keys.** M3 runs 1–3 and gpt-oss 3-shot used a runner that rotated
   over several Groq keys when one was rate-limited (same model and settings;
   each row records which key it used). From the Qwen 0-shot run onwards, by
   request, a single key with retry/backoff is used. This slows the remaining
   P1 runs considerably.
7. **M3-alt is Qwen3.8-27B** (see §8). One run per M3 variant, not three.
8. **Package versions differ from v4** (Kaggle image update). M0–M2 still
   reproduce v4 step by step, so the effect is nil.
9. **The prompt asymmetry from the audit is kept as in v4** for comparability:
   M3 sees param signatures, M0–M2 do not. A symmetric-prompt variant was not
   run.

## Files

- `pilot_results_v5.jsonl`: per step, all modes, with raw output, parsed call,
  parse_ok, exec_error, state_delta, tokens, latency.
- `raw/`: Kaggle outputs and metadata (adapter sha, package versions, prompt
  template hashes); `m3/`: three M3 runs; `m3_variants/`: item 8.
- `metrics_v5.json`, `relaxed_v5.json`, `relaxed_examples_v5.jsonl` (with
  manual verdicts), `cost_v5.json`, `routers_v5.json`, `grounding_v5.json`,
  `grounding_ungrounded_v5.jsonl`.
- `fig1_heatmap_v5.pdf` (TSR and SSR_tf by tier × mode), `fig2_pareto_v5.pdf`
  (TSR vs FLOPs).
