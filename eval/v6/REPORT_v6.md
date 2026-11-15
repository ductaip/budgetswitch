# ViTool v6 — grounding repair → ViTool v1.1 (REPORT_v6)

Status 2026-11-15. All runs complete (0 API errors after reruns). Steps 0–5 done; Step 6 partly done (Groq quota, see §6); Step 7 figures done
(`fig1_heatmap_v6_v12`, `fig2_pareto_v6_v12`, PDF + PNG). Every table in the appendix is
generated from the JSON outputs by `eval/v6/report_tables_v6.py`.
Headline metric: **TSR** = every step strictly correct AND the executed calls
reproduce the gold final environment state. SSR is always labelled teacher-forced
(SSR_tf) or rollout (SSR_rollout). 112 eval trajectories / 284 steps, greedy
decoding, seed 42.

## Summary

1. **The grounding audit is valid.** On 60 hand-labelled values (two authors,
   κ = 0.76), Gate G3 agrees with the adjudicated labels on 60/60 items
   (precision 1.0, in-sample recall 1.0).
2. **One grounding number (G3, adjudicated rule):** 200/375 v1 trajectories
   contain a gold value the user never stated; **60/112 eval** (easy 4/43,
   medium 17/26, hard 39/43). Only 11 trajectories (2 eval) are ungrounded
   *only* because of an unstated quantity.
3. **ViTool v1.1:** queries of the 198 repairable trajectories were rewritten
   (gold calls untouched); 2 train trajectories with placeholder gold values were
   removed → 373 trajectories (261 train / 112 eval), **373/373 pass every gate
   including G3**.
4. **Every mode gains 0.125–0.143 TSR on v1.1** (paired bootstrap, all p ≤ 0.004).
   M0 0.438 → 0.562, M1 0.420 → 0.562, M2 0.384 → 0.518, M3 0.393 → 0.527,
   oracle 0.491 → 0.688. Hard-tier TSR rises from ≤ 0.07 to 0.16–0.28.
5. **The gain comes from the queries, not from retraining.** The v4 adapter on
   v1.1 queries already reaches M0 TSR 0.598; the adapter retrained on v1.1 train
   gives 0.562 (Δ −0.036 [−0.098, +0.027], p = 0.297). No mode changes
   significantly between the two adapters.
6. **Mode ranking:** on v1.1 M0 = M1 (0.562) > M3 (0.527) > M2 (0.518); none of
   the differences to M0 is significant on TSR. The routing headroom grows:
   oracle − best fixed mode = +0.13 TSR (v1: +0.05), and the oracle saves
   39.0% (assumed) / 22.6% (tokens) / 14.4% (FLOPs) of the cost of always-M1.
7. **Residual grounding gap (new finding, needs a decision):** G3 accepts any
   enum value as grounded. In v1.1 eval, 26/30 gold `payment_method` and 21/39
   gold `platform` values are not stated in the query (31 eval trajectories).
   M0 gets `platform` right on 16/18 steps where the query names it but on 6/21
   where it does not. This is the main remaining error source in e-commerce
   (TSR 0.30). See §7.

## Step 0 — validation of the grounding audit

- Sample: 40 flagged + 20 unflagged values (stratified, seed 42), two authors,
  independent labels (`audit_sample_annotator{1,2}.json`), guide
  `ANNOTATION_GUIDE.md`.
- Agreement: 86.7% on 4 labels (κ = 0.764), 88.3% binary (κ = 0.757).
- Adjudication (`audit_sample_adjudicated.json`, 8 items): A28 =
  from_prior_observation, A11 = ungrounded, A26 = inferable_from_query, and the
  five `quantity = 1` items = ungrounded (rule: a quantity absent from the query
  is not grounded; the schema declares no default).
- Audit precision vs adjudicated labels: strict 0.975 [0.87, 1.00], lenient 1.000
  [0.89, 1.00]. With annotator 2 alone the strict audit falls to 0.825 — all of
  the gap is the quantity rule, which the adjudication settles.
- G3 (lenient minus implicit quantity) vs adjudicated labels: 60/60
  (`g3_vs_adjudicated.json`).
- Conventional values (`audit_value_frequency.json`): 58 values repeat ≥ 2 times
  and cover 327/502 flagged occurrences; 83/139 flagged eval occurrences use a
  value that also appears in train (e.g. `1`, `500000`, `1234567890`, `Lê Văn C`).

## Step 1 — Gate G3 in the pipeline

- Code: `data/grounding_gate.py`, called from `data/build_dataset.accept_trajectory`
  after Gate 2 (also used by `--revalidate` and the LLM builder). Report command:
  `python data/build_dataset.py --gate-report <jsonl...> --out <json>`.
- A value passes if it is in the query after normalisation (accents, spacing,
  date formats, Vietnamese number formats, number words with a counter such as
  "hai chiếc"), maps through a small lexicon (currencies, languages, city
  abbreviations), appears in an earlier step's real observation, or is an enum
  value / schema default.
- Unit tests: `tests/test_grounding_gate.py` (18 tests, all pass), including
  money written as "500.000đ", "2000000đ", "500.000 VNĐ", "2 triệu đồng".
- Bugs found and fixed while building G3 (inherited from the v5 audit):
  (1) money with a glued "đ" ("500.000đ") was not read as a number;
  (2) "500.000" was also read as the decimal 500. Neither changes any v1 count
  (no v1 query writes money that way), so v5 numbers stand.
- Gate 1.5 (placeholder) now also rejects `id_…` values and empty strings.

**Counts on all 375 v1 trajectories** (`g3_report_v1.json`, `g3_summary.json`):
175 pass every gate; 200 fail G3 (198 fail G3 first, 2 fail Gate 1.5 first and
also fail G3); 484 ungrounded values. Most frequent: `add_to_cart.quantity` 94,
`search_product.max_price` 69, `book_flight.passenger_names` 60,
`apply_voucher.voucher_code` 54.

**Why 209 (v5 audit) → 200 (G3).** The 9 trajectories flagged by the v5 strict
audit but not by G3 are all info_retrieval tasks whose only "ungrounded" values
are currency or language codes the query names in words (USD for "đô la Mỹ",
VND for "tiền Việt Nam", fr for "tiếng Pháp"); G3's lexicon accepts them, the
strict audit did not. G3 flags no trajectory that the v5 audit missed. (The
"199" quoted in an earlier status message was the number whose *first* failing
gate is G3; the G3 total is 200.)

## Step 2 — query repair → ViTool v1.1

- Files: `data/vitool/vitool_v1.1.jsonl` (373, field `split`),
  `sft_train_v1.1.jsonl` (261), `pilot_eval_v1.1.jsonl` (112);
  `eval/v6/query_diffs.jsonl`, `rewrite_meta.json` (prompt, hash `b7abbd9847d26c04`).
- Rewriter: `openai/gpt-oss-20b` on Groq, temperature 0, seed 42,
  reasoning_effort low. Only the query changes; an assertion checks that gold
  calls, traj_id, domain, tier, final state and split are identical.
- Acceptance: all gates (structure, resolve, 1, 1.5, 2, G3) on the new query, no
  copied system ids (PROD…, FL…), no English parameter names, ≥ 70% of the old
  query's words kept; otherwise up to 3 retries with the failure reason fed back.
- Result: 198/200 accepted automatically (188 at the first attempt, 10 at the
  second). The other 2 had placeholder gold values, not a query problem:
  - `ecommerce_0009` (train): `order_id = "id_đơn_hàng"`; single-step task, no
    earlier observation produces an order id → **removed from v1.1**.
  - `travel_0018` (train): `seat_class = ""`; not derivable from any observation
    → **removed from v1.1**. (Same rule; logged in `rewrite_meta.json`.)
- Edit distance old → new (`rewrite_stats.json`, 198 rewrites): median 46
  characters (p10 17, p90 94); relative to the old query length median 0.56
  (p10 0.19, p90 1.48); median 9 words and 2 values added; 97.5% of the old
  query's words kept on average; 70% of rewrites are pure insertions. Eval
  rewrites are slightly smaller than train ones (median relative edit 0.51 vs 0.60).
- **Manual check of 50 random rewrites** (`rewrite_check_annotator{1,2}.json`,
  two authors, `checks_validation.json`): meaning preserved 50/50 for both
  authors (Wilson CI [0.93, 1.00]; agreement 100%). Naturalness: no rewrite rated
  1 (unnatural); mean 2.48 and 2.54 on the 1–3 scale (3 = fully natural: 24 and
  27 items). Agreement on 2 vs 3 is low (58% exact, weighted κ = 0.16): the
  authors agree that rewrites are acceptable but not on which ones read slightly
  stiff. Annotator notes flag long, list-like queries (R01, R07, R10) and four
  cases where the rewrite faithfully exposes an oddity of the **gold**, not of
  the rewrite: an unrelated extra step (R26 `check_balance`, R33 BHXH lookup),
  an unrelated expression (R27) and an implausible price cap (R46
  `max_price = 15000` for a phone).
- **Spot check of 30 random v1.1 trajectories** (`v11_spotcheck_annotator{1,2}.json`,
  10 eval / 20 train, 14 rewritten): a fabricated value remains in 3/30
  (annotator 1) and 4/30 (annotator 2) trajectories, Wilson CI [0.04, 0.26] and
  [0.05, 0.30]; agreement 96.7%, κ = 0.84. **Every reported value is an enum
  choice the query does not make**: `checkout.payment_method` (cod, momo,
  bank_transfer), `search_product.platform` (shopee, lazada) and
  `return_request.refund_method` (original). No non-enum fabricated value was
  found. The automatic enum check of §7 flags exactly the same 4 trajectories.

## Step 3 — retrained adapter

Kaggle kernel `budgetswitch-v6-retrain` (2× T4): v4 recipe (Unsloth 2026.6.2,
4-bit Qwen2.5-3B-Instruct, LoRA r 16 / α 32 on 7 projections, 3 epochs, bs 4 ×
acc 4, lr 2e-4 cosine, seed 42, 10% validation split) on `sft_train_v1.1.jsonl`
(sha `8c97679772e1…`). 1,432 examples (v4: 1,438; the 3 removed steps × 2
example types), 47.5 min, eval loss 0.041 / 0.029 / 0.025 by epoch. Adapter
`eval/v6/raw/tool_use_lora_v6`, sha256 `61b838dbf038…`. The v4 adapter
(`c704f9d562b6…`) is kept and was rerun on v1.1 (kernel `budgetswitch-v6-v4adapter`).

## Step 4 — runs on v1.1 eval

| run | status |
|---|---|
| M0/M1/M2 teacher-forced + rollout (M0/M1/M2), retrained adapter | done |
| M0/M1/M2 teacher-forced + rollout, v4 adapter on v1.1 queries | done |
| M3 gpt-oss-120b zero-shot run 1 | done |
| M3 run 2 | done |
| M3 run 3 | done |

Logging as in v5 (`*.meta.json`: adapter and data sha, packages, template
hashes, hardware). Rollout now also covers M2 (v5 rollout had M0/M1 only).
Rollout and teacher-forced TSR flags agree on 111–112/112 trajectories per mode;
SSR_tf − SSR_rollout = 0.16–0.18.

## Step 5 — results (tables T1–T5 in the appendix)

- **v1 vs v1.1 (T1, T3):** all four modes gain +0.125 to +0.143 TSR (v1 → v1.1,
  p ≤ 0.004), and all of it is already there with the v4 adapter (v1 →
  v1.1_v4adapter: +0.125 to +0.161). Retraining on v1.1 changes no mode
  significantly (|Δ| ≤ 0.036, p ≥ 0.29). Interpretation: on v1 the adapter was
  not failing for lack of skill but because the gold expected values it could not
  know; the memorised conventions in the v4 train split (`1234567890`, …) did not
  give it a measurable edge either way.
- **Per tier (T2):** easy stays ≈ 0.95 for every mode; medium 0.27–0.42; hard
  0.16–0.28 (v1: 0.02–0.07). Hard tasks are no longer near zero, so the tier
  analysis is now informative.
- **Per domain (T2):** e-commerce remains the weak domain (TSR 0.23–0.30;
  oracle 0.43) because of the enum issue in §7; banking, info retrieval, public
  services and travel are 0.59–0.88.
- **Mode comparison (T2):** M0 vs M1 tie on TSR (0.562 each; SSR_tf McNemar 3 vs
  8 steps, p = 0.23). M3 is worse on steps (46 vs 22, p = 0.005) but not on
  trajectories (−0.036, p = 0.48). M2 is the weakest local mode.
- **Routers (T5):** on v1.1 the learned pre-generation router reaches TSR 0.580
  (best fixed mode 0.562) at 1.08× M0 tokens; the M0→M3 error cascade 0.571;
  the length router 0.545. The oracle at 0.688–0.696 shows +0.13 headroom (v1:
  +0.05). Learned-router AUCs dropped (M0 0.80 → 0.64): on v1 the router could
  predict failure from "task contains values the user never gave", a signal that
  v1.1 removes.
- **Oracle savings vs always-M1 (T5):** 39.0% assumed, 22.6% tokens, 14.4% FLOPs
  (v1: 51.2 / 21.8 / 16.9). M2 remains the cheapest mode under measured cost
  (0.65× M0 tokens) despite its higher assumed cost.
- **Oracle TSR differs by cost model** (0.688 assumed vs 0.696 tokens/FLOPs): the
  oracle picks the cheapest strictly correct mode per step, and different correct
  calls can differ in ways the strict step check ignores but the final-state check
  does not (e.g. letter case; trajectories listed under `state_check_changes` in
  `metrics_v6.json`). All tables use M3 run 1 for M3 and for the oracle.
- **Clarification (T4):** on v1 only M3 asked questions (7/284 steps, 6 of them on
  G3-ungrounded steps, i.e. arguably correct behaviour). On v1.1 no mode asks
  (0/284), so the "clarification counted as correct" view is moot.

## Step 6 — M3 variants on v1.1 (and the Groq quota)

One key (`GROQ_API_KEY`), retry/backoff, no key rotation. Measured limits for
this key: 8,000 tokens/min and 1,000 requests/day per model, plus a daily token
budget that, once used, forces 5–12 minute waits (both models). Throughput
varied from 3–4 steps/hour per model (night of 7–8 Nov) to much faster on
later days; the jobs also stopped once when the machine restarted and were
resumed from the last complete line (no step was run twice).

| run (v1.1 eval, 284 steps) | status on 2026-11-13 |
|---|---|
| gpt-oss-120b zero-shot run 1 | done |
| gpt-oss-120b zero-shot run 2 | done |
| gpt-oss-120b zero-shot run 3 | done |
| gpt-oss-120b 3-shot (demos from v1.1 train, all G3-clean, seed 42) | done |
| Qwen3.8-27B 0-shot | done |
| Qwen3.8-27B 3-shot | 232/284, queued after Qwen 0-shot on v1.2 |

First results (T6):
- **Qwen3.8-27B 0-shot reaches TSR 0.625 [0.53, 0.71] on v1.1**, above every
  fixed mode in T2 (M0/M1 0.562, gpt-oss 0.527), with hard-tier TSR 0.279. It
  is dense (27B active), so its FLOPs per step are ~5× gpt-oss-120b's (7.6e13
  vs 1.5e13) and its token price higher ($0.35 vs $0.08 for 284 steps). It is
  not yet part of the oracle/router tables, which keep the brief's M3
  (gpt-oss-120b zero-shot, run 1).
- gpt-oss-120b zero-shot runs 1 and 2 give the same TSR (0.527); SSR_tf 0.683
  vs 0.659. Run 2 has 2 steps whose API call failed after all retries; they
  are counted as failures.

Results of finished runs are in T6 and in the M3-repeat line of T2; unfinished
runs are listed as incomplete and fill in when `build_report_v6.sh` is rerun.

## 7. Residual grounding gap: enum choices the user never made

G3 (as approved in Step 0) treats any enum value as grounded. The annotators'
`enum_justified` field already hinted at the problem (2 of 4 and 2 of 3 enum
labels marked "not justified"). Measured on v1.1 with a word list per value
("chuyển khoản" for bank_transfer, "khi nhận hàng" for cod, platform names):

| gold value not named in the query | eval | train |
|---|---|---|
| `checkout.payment_method` | 26/30 | 70/78 |
| `search_product.platform` | 21/39 | 31/97 |
| trajectories affected (either param) | 31/112 | 67/261 |

The two authors' spot check (§ Step 2) confirms this independently: all
fabricated values left in 30 random v1.1 trajectories are of this kind (plus
`return_request.refund_method`, not counted in the table), and the word-list
check flags the same 4 trajectories as the humans.

M0 accuracy on these steps: `platform` 16/18 when named vs 6/21 when not;
`payment_method` 13/26 when not named. A looser count over all enum params
(news category, utility type, …) gives 66 unstated enum values in eval, but that
count needs a full Vietnamese synonym list and is an upper bound.

Options: (a) report it as a known limitation of v1.1; (b) tighten G3 so a
non-default enum value must be stated (or derivable), rewrite the ~98 affected
queries (same procedure, 1–2 hours) → v1.2, and rerun Kaggle (≈ 5 h) — the M3
reruns are then quota-bound as in §6.

## v1.2 — enum choices must be stated (authors' decision, 2026-11-11)

- **G3 v1.2** (`data/grounding_gate.py`, `enum_rule="v1.2"`, now the pipeline
  default): an enum value is grounded only if the query states it — literally or
  through the Vietnamese phrase table T7 — or it comes from an earlier
  observation (or equals a schema default; the catalog has none). The v1.1 rule
  stays available (`enum_rule="v1.1"`); rebuilding v1.1 with it reproduces the
  v1.1 files byte for byte. 31 unit tests pass.
- Under G3 v1.2, **104/373 v1.1 trajectories fail** (34 eval, 70 train): mostly
  `checkout.payment_method` (96 values) and `search_product.platform` (52), then
  `return_request.refund_method`, `civil_status_lookup.document_type`, etc.
  (`g3v12_report_v1.1.json`). On v1 the strict rule flags 203/375 (63 eval).
- **Rewrite → v1.2** (same model gpt-oss-20b, same system prompt, hash
  `b7abbd9847d26c04`): 104/104 accepted automatically (103 at the first
  attempt). **v1.2: 373 trajectories (261 / 112), 373/373 pass every gate
  including G3 v1.2** (`g3_report_v12.json`). Deviation: the missing-value list
  in the user message now carries a Vietnamese wording hint for enum codes
  (e.g. `cod` → "thanh toán khi nhận hàng"); without it the model copied the
  codes into the query.
- Edit distance v1.1 → v1.2 (`rewrite_stats_v12.json`): median 30 characters,
  0.25 of the query length (p90 0.51); 99.6% of words kept; 91% pure insertions.
- **Spot check of 20 random v1.2 trajectories** (two authors, enum choices now
  count as fabricated): **0/20 by both** (Wilson upper bound 0.16; agreement
  100%). On v1.1 the same check found 3–4/30, all enum choices.
- **Runs:** v4 adapter only (as decided): M0/M1/M2 teacher-forced + rollout
  (kernel `budgetswitch-v6-v4adapter-v12`, adapter sha `c704f9d562b6…`, data sha
  `ac91473853a2…`). M3 on v1.2: gpt-oss-120b 0-shot run 1 **done**
  (2026-11-13); Qwen3.8-27B 0-shot on v1.2 done (2026-11-15): **TSR 0.723**
  [0.63, 0.80], SSR_tf 0.884, 22× M0 FLOPs, $0.35 / 284 steps — above M0 (0.688)
  but not significantly (+0.036, paired bootstrap p = 0.45; steps 20 vs 23).
- **Results (T1, T3):** TSR v1 → v1.1 → v1.2 (v4 adapter): M0 0.438 → 0.598 →
  **0.688**, M1 0.420 → 0.554 → **0.607**, M2 0.384 → 0.509 → **0.527**.
  v1.1 → v1.2: M0 +0.089 (p < 0.001), M1 +0.054 (p = 0.065), M2 +0.018 (p = 0.59).
  E-commerce M0 TSR 0.30 → 0.48; hard tier M0 0.42, medium 0.62, easy 1.00.
  Rollout TSR M0 0.679, M1 0.607.
- **M3 and oracle on v1.2:** M3 (gpt-oss-120b 0-shot) TSR 0.598 (v1.1: 0.527,
  +0.071, p = 0.054). 5 steps of the first v1.2 M3 run ended in
  `APITimeoutError` after all retries and were scored as wrong (TSR 0.571); they
  were rerun on 2026-11-13 (original kept as `*.with_api_errors.jsonl.bak`).
  Oracle over M0–M3 0.804 (assumed costs) / 0.795 (tokens, FLOPs). On v1.2 M0
  is the best fixed mode: vs M1 −0.080 (p = 0.019), vs M2 −0.161 (p < 0.001),
  vs M3 −0.089 (**p = 0.051, not significant on TSR**; on steps McNemar 46 vs 17,
  p < 0.001). Oracle headroom over M0: +0.116.
- **Routers on v1.2 (T5-v1.2, fig2):** best learned pre-generation router TSR
  0.705 at 1.04× M0 tokens (M0 0.688); cascade M0→M1 0.696; length router 0.562;
  random 0.598. Learned-router AUCs are 0.56–0.62, i.e. step success is hard
  to predict before generation once queries are grounded. Oracle saves 46.7%
  (assumed) / 27.9% (tokens) / 21.7% (FLOPs) vs always-M1.

## Deviations from the brief

1. **Rewriter model:** `openai/gpt-oss-20b` instead of the original generator
   (llama-3.3-70b-versatile, no longer served by Groq). It is a different model
   from M3 (gpt-oss-120b) but from the same family. Approved by the authors;
   edit-distance distribution reported above.
2. **Two train trajectories removed** (placeholder gold values; rule approved by
   the authors), so v1.1 has 373 trajectories and the retrained adapter saw 1,432
   instead of 1,438 examples. The eval split is unchanged (112).
3. **G3 definition** = Step-0 lenient audit minus implicit quantity (authors'
   decision), plus two parser fixes; Gate 1.5 extended to `id_…` and empty
   strings.
4. **Rollout for M2** added (brief asks for M0/M1/M2 rollout; v5 had M0/M1).
5. **v1.1_v4adapter uses the v1.1 M3 runs** (M3 does not depend on the adapter),
   so its M3 column equals the main condition's.
6. **Step 6 incomplete** at writing time because of the Groq quota (§6): gpt-oss
   Qwen 3-shot on v1.1 and Qwen 0-shot on v1.2 still running.
7. **Figures** are drawn for v1.2 (main) and v1.1 (v4 adapter); fig2 uses a
   numbered key instead of free labels so no text overlaps.
9. **v1.2** (beyond the v6 brief, authors' decision): strict enum rule, enum
   wording hint in the rewrite user message, v4 adapter only, M3 one run per
   model on v1.2.
8. **Extra manual check** (`v11_spotcheck`, 30 trajectories) added at the
   authors' request, beyond the brief's 50-item rewrite check.

## Figures (Step 7)

- `fig1_heatmap_v6_v12.{pdf,png}`: TSR | SSR teacher-forced | SSR rollout by tier
  × mode, one colour scale (0–1). M3 and the oracle have no rollout (hatched,
  "n/a"). Oracle = M0–M3 with M3 run 1, same as the tables.
- `fig2_pareto_v6_v12.{pdf,png}`: TSR vs FLOPs per step relative to always-M0
  (log x), y-axis 0–1. Every point is numbered and listed with its TSR; legend
  for marker types. The oracle is the FLOPs oracle (cheapest strictly correct
  mode per step over M0–M3, M3 run 1), so its TSR (0.795) differs slightly from
  the assumed-cost oracle in T1 (0.804): different correct calls can differ in
  the final-state check. The v5 figure's oracle (0.482) was likewise the
  tokens/FLOPs oracle on v1 while the v4 table used assumed costs; both use M3
  run 1.
- `*_v11_v4adapter` versions show v1.1 with the Step-6 M3 variants (gpt-oss
  3-shot, Qwen 0-shot) as diamonds. Rerun `figures_v6.py` to refresh.

## Files

`data/grounding_gate.py`, `data/build_dataset.py` (G3 + `--gate-report`),
`tests/test_grounding_gate.py`; `data/vitool/{vitool,sft_train,pilot_eval}_v1.1.jsonl`;
`eval/v6/`: `audit_*`, `g3_*`, `rewrite_*`, `query_diffs.jsonl`,
`rewrite_check*`, `v11_spotcheck*`, `CHECKS_GUIDE.md`, `kaggle/`, `raw/`, `m3/`,
`m3_variants/`, `metrics_v6.json`, `cost_v6.json`, `routers_v6.json`,
`pilot_results_v6.jsonl`, `report_tables_v6.md`.

# Appendix — generated tables
### T1. Main table — v1 / v1.1 / v1.2 (all with the v4 adapter; retrained v1.1 for reference)

| metric | v1 | v1.1 | **v1.2** | Δ v1.2 − v1 | v1.1, retrained adapter |
|---|---|---|---|---|---|
| TSR M0 | 0.438 | 0.598 | **0.688** | +0.250 | 0.562 |
| SSR_tf M0 | 0.588 | 0.753 | **0.873** | +0.285 | 0.768 |
| TSR M0 domain-macro | 0.524 | 0.704 | **0.757** | +0.233 | 0.647 |
| TSR M0 easy | 0.930 | 1.000 | **1.000** | +0.070 | 0.954 |
| TSR M0 medium | 0.269 | 0.538 | **0.615** | +0.346 | 0.385 |
| TSR M0 hard | 0.046 | 0.233 | **0.419** | +0.372 | 0.279 |
| TSR M1 | 0.420 | 0.554 | **0.607** | +0.188 | 0.562 |
| SSR_tf M1 | 0.570 | 0.747 | **0.824** | +0.254 | 0.785 |
| TSR M1 domain-macro | 0.495 | 0.645 | **0.672** | +0.177 | 0.652 |
| TSR M1 easy | 0.837 | 0.907 | **0.907** | +0.070 | 0.954 |
| TSR M1 medium | 0.308 | 0.423 | **0.462** | +0.154 | 0.385 |
| TSR M1 hard | 0.070 | 0.279 | **0.395** | +0.326 | 0.279 |
| TSR M2 | 0.384 | 0.509 | **0.527** | +0.143 | 0.518 |
| SSR_tf M2 | 0.535 | 0.697 | **0.768** | +0.232 | 0.725 |
| TSR M2 domain-macro | 0.466 | 0.606 | **0.616** | +0.150 | 0.604 |
| TSR M2 easy | 0.791 | 0.861 | **0.861** | +0.070 | 0.954 |
| TSR M2 medium | 0.308 | 0.462 | **0.462** | +0.154 | 0.269 |
| TSR M2 hard | 0.023 | 0.186 | **0.233** | +0.209 | 0.233 |
| TSR M3 | 0.393 | 0.527 | **0.598** | +0.205 | 0.527 |
| SSR_tf M3 | 0.507 | 0.683 | **0.771** | +0.264 | 0.683 |
| TSR M3 domain-macro | 0.463 | 0.623 | **0.690** | +0.227 | 0.623 |
| TSR M3 easy | 0.861 | 0.954 | **0.954** | +0.093 | 0.954 |
| TSR M3 medium | 0.231 | 0.423 | **0.538** | +0.308 | 0.423 |
| TSR M3 hard | 0.023 | 0.163 | **0.279** | +0.256 | 0.163 |
| TSR oracle | 0.491 | 0.732 | **0.804** | +0.312 | 0.688 |
| SSR_tf oracle | 0.648 | 0.859 | **0.944** | +0.296 | 0.866 |
| TSR oracle domain-macro | 0.573 | 0.827 | **0.864** | +0.291 | 0.765 |
| TSR oracle easy | 0.930 | 1.000 | **1.000** | +0.070 | 0.977 |
| TSR oracle medium | 0.385 | 0.769 | **0.808** | +0.423 | 0.615 |
| TSR oracle hard | 0.116 | 0.442 | **0.605** | +0.488 | 0.442 |
| TSR_rollout M0 | 0.438 | 0.598 | **0.679** | +0.241 | 0.554 |
| SSR_rollout M0 | 0.468 | 0.630 | **0.722** | +0.254 | 0.606 |
| TSR_rollout M1 | 0.420 | 0.554 | **0.607** | +0.188 | 0.562 |
| SSR_rollout M1 | 0.433 | 0.592 | **0.665** | +0.232 | 0.623 |
| TSR M3 mean of runs | 0.381 | 0.521 | **0.598** | +0.217 | 0.521 |
| clarify rate M0 (tf) | 0.000 | 0.000 | **0.000** | +0.000 | 0.000 |
| clarify rate M1 (tf) | 0.000 | 0.000 | **0.000** | +0.000 | 0.000 |
| clarify rate M2 (tf) | 0.000 | 0.000 | **0.000** | +0.000 | 0.000 |
| clarify rate M3 (tf) | 0.025 | 0.000 | **0.004** | -0.021 | 0.000 |

### T2-v1.2. TSR (strict + state) and SSR, Wilson 95% CI — v1.2

| mode | TSR (tf) | SSR_tf | TSR_rollout | SSR_rollout | TSR domain-macro |
|---|---|---|---|---|---|
| M0 | 0.688 [0.60, 0.77] | 0.873 [0.83, 0.91] | 0.679 [0.59, 0.76] | 0.722 [0.67, 0.77] | 0.757 |
| M1 | 0.607 [0.51, 0.69] | 0.824 [0.78, 0.86] | 0.607 [0.51, 0.69] | 0.665 [0.61, 0.72] | 0.672 |
| M2 | 0.527 [0.43, 0.62] | 0.768 [0.72, 0.81] | 0.518 [0.43, 0.61] | 0.532 [0.47, 0.59] | 0.616 |
| M3 | 0.598 [0.51, 0.68] | 0.771 [0.72, 0.82] | – | – | 0.690 |
| oracle | 0.804 [0.72, 0.87] | 0.944 [0.91, 0.96] | – | – | 0.864 |

By tier (TSR, tf) — v1.2

| mode | easy (n=43) | medium (n=26) | hard (n=43) |
|---|---|---|---|
| M0 | 1.000 | 0.615 | 0.419 |
| M1 | 0.907 | 0.462 | 0.395 |
| M2 | 0.861 | 0.462 | 0.233 |
| M3 | 0.954 | 0.538 | 0.279 |
| oracle | 1.000 | 0.808 | 0.605 |

By domain (TSR, tf) — v1.2

| mode | banking (n=16) | ecommerce (n=44) | info_retrieval (n=17) | public_services (n=16) | travel (n=19) |
|---|---|---|---|---|---|
| M0 | 0.938 | 0.477 | 0.824 | 0.812 | 0.737 |
| M1 | 0.875 | 0.409 | 0.706 | 0.688 | 0.684 |
| M2 | 0.938 | 0.273 | 0.706 | 0.688 | 0.474 |
| M3 | 0.875 | 0.318 | 0.647 | 0.875 | 0.737 |
| oracle | 1.000 | 0.614 | 0.882 | 0.875 | 0.947 |

Paired tests vs M0 — v1.2

| comparison | SSR_tf McNemar (M0 wins / other wins, p) | TSR paired bootstrap diff [95% CI], p |
|---|---|---|
| M0_vs_M1 | 19 / 5, p=0.00661 | -0.080 [-0.152, -0.018], p=0.0186 |
| M0_vs_M2 | 37 / 7, p=5e-06 | -0.161 [-0.241, -0.089], p=0 |
| M0_vs_M3 | 46 / 17, p=0.000337 | -0.089 [-0.179, +0.000], p=0.0514 |

M3 repeats (v1.2): TSR per run [0.5982], mean 0.5982, sd None; SSR_tf per run [0.7711]; API errors [0].

### T2-v1.1_v4adapter. TSR (strict + state) and SSR, Wilson 95% CI — v1.1_v4adapter

| mode | TSR (tf) | SSR_tf | TSR_rollout | SSR_rollout | TSR domain-macro |
|---|---|---|---|---|---|
| M0 | 0.598 [0.51, 0.68] | 0.753 [0.70, 0.80] | 0.598 [0.51, 0.68] | 0.630 [0.57, 0.68] | 0.704 |
| M1 | 0.554 [0.46, 0.64] | 0.747 [0.69, 0.79] | 0.554 [0.46, 0.64] | 0.592 [0.53, 0.65] | 0.645 |
| M2 | 0.509 [0.42, 0.60] | 0.697 [0.64, 0.75] | 0.500 [0.41, 0.59] | 0.532 [0.47, 0.59] | 0.606 |
| M3 | 0.527 [0.43, 0.62] | 0.683 [0.63, 0.73] | – | – | 0.623 |
| oracle | 0.732 [0.64, 0.81] | 0.859 [0.81, 0.89] | – | – | 0.827 |

By tier (TSR, tf) — v1.1_v4adapter

| mode | easy (n=43) | medium (n=26) | hard (n=43) |
|---|---|---|---|
| M0 | 1.000 | 0.538 | 0.233 |
| M1 | 0.907 | 0.423 | 0.279 |
| M2 | 0.861 | 0.462 | 0.186 |
| M3 | 0.954 | 0.423 | 0.163 |
| oracle | 1.000 | 0.769 | 0.442 |

By domain (TSR, tf) — v1.1_v4adapter

| mode | banking (n=16) | ecommerce (n=44) | info_retrieval (n=17) | public_services (n=16) | travel (n=19) |
|---|---|---|---|---|---|
| M0 | 0.938 | 0.273 | 0.824 | 0.750 | 0.737 |
| M1 | 0.875 | 0.273 | 0.706 | 0.688 | 0.684 |
| M2 | 0.938 | 0.227 | 0.706 | 0.688 | 0.474 |
| M3 | 0.875 | 0.227 | 0.588 | 0.688 | 0.737 |
| oracle | 1.000 | 0.432 | 0.882 | 0.875 | 0.947 |

Paired tests vs M0 — v1.1_v4adapter

| comparison | SSR_tf McNemar (M0 wins / other wins, p) | TSR paired bootstrap diff [95% CI], p |
|---|---|---|
| M0_vs_M1 | 14 / 12, p=0.845 | -0.045 [-0.107, +0.018], p=0.231 |
| M0_vs_M2 | 25 / 9, p=0.00904 | -0.089 [-0.161, -0.027], p=0.0124 |
| M0_vs_M3 | 45 / 25, p=0.0225 | -0.071 [-0.152, +0.009], p=0.0818 |

M3 repeats (v1.1_v4adapter): TSR per run [0.5268, 0.5357, 0.5], mean 0.5208, sd 0.0186; SSR_tf per run [0.6831, 0.662, 0.6479]; API errors [0, 0, 0].

### T2-v1.1. TSR (strict + state) and SSR, Wilson 95% CI — v1.1

| mode | TSR (tf) | SSR_tf | TSR_rollout | SSR_rollout | TSR domain-macro |
|---|---|---|---|---|---|
| M0 | 0.562 [0.47, 0.65] | 0.768 [0.72, 0.81] | 0.554 [0.46, 0.64] | 0.606 [0.55, 0.66] | 0.647 |
| M1 | 0.562 [0.47, 0.65] | 0.785 [0.73, 0.83] | 0.562 [0.47, 0.65] | 0.623 [0.57, 0.68] | 0.652 |
| M2 | 0.518 [0.43, 0.61] | 0.725 [0.67, 0.77] | 0.518 [0.43, 0.61] | 0.546 [0.49, 0.60] | 0.604 |
| M3 | 0.527 [0.43, 0.62] | 0.683 [0.63, 0.73] | – | – | 0.623 |
| oracle | 0.688 [0.60, 0.77] | 0.866 [0.82, 0.90] | – | – | 0.765 |

By tier (TSR, tf) — v1.1

| mode | easy (n=43) | medium (n=26) | hard (n=43) |
|---|---|---|---|
| M0 | 0.954 | 0.385 | 0.279 |
| M1 | 0.954 | 0.385 | 0.279 |
| M2 | 0.954 | 0.269 | 0.233 |
| M3 | 0.954 | 0.423 | 0.163 |
| oracle | 0.977 | 0.615 | 0.442 |

By domain (TSR, tf) — v1.1

| mode | banking (n=16) | ecommerce (n=44) | info_retrieval (n=17) | public_services (n=16) | travel (n=19) |
|---|---|---|---|---|---|
| M0 | 0.812 | 0.295 | 0.765 | 0.625 | 0.737 |
| M1 | 0.750 | 0.273 | 0.824 | 0.625 | 0.789 |
| M2 | 0.750 | 0.250 | 0.647 | 0.688 | 0.684 |
| M3 | 0.875 | 0.227 | 0.588 | 0.688 | 0.737 |
| oracle | 0.875 | 0.432 | 0.882 | 0.688 | 0.947 |

Paired tests vs M0 — v1.1

| comparison | SSR_tf McNemar (M0 wins / other wins, p) | TSR paired bootstrap diff [95% CI], p |
|---|---|---|
| M0_vs_M1 | 3 / 8, p=0.227 | +0.000 [-0.036, +0.036], p=1 |
| M0_vs_M2 | 23 / 11, p=0.0576 | -0.045 [-0.098, +0.009], p=0.109 |
| M0_vs_M3 | 46 / 22, p=0.0049 | -0.036 [-0.125, +0.054], p=0.479 |

M3 repeats (v1.1): TSR per run [0.5268, 0.5357, 0.5], mean 0.5208, sd 0.0186; SSR_tf per run [0.6831, 0.662, 0.6479]; API errors [0, 0, 0].

### T2-v1. TSR (strict + state) and SSR, Wilson 95% CI — v1

| mode | TSR (tf) | SSR_tf | TSR_rollout | SSR_rollout | TSR domain-macro |
|---|---|---|---|---|---|
| M0 | 0.438 [0.35, 0.53] | 0.588 [0.53, 0.64] | 0.438 [0.35, 0.53] | 0.468 [0.41, 0.53] | 0.524 |
| M1 | 0.420 [0.33, 0.51] | 0.570 [0.51, 0.63] | 0.420 [0.33, 0.51] | 0.433 [0.38, 0.49] | 0.495 |
| M2 | 0.384 [0.30, 0.48] | 0.535 [0.48, 0.59] | – | – | 0.466 |
| M3 | 0.393 [0.31, 0.49] | 0.507 [0.45, 0.56] | – | – | 0.463 |
| oracle | 0.491 [0.40, 0.58] | 0.648 [0.59, 0.70] | – | – | 0.573 |

By tier (TSR, tf) — v1

| mode | easy (n=43) | medium (n=26) | hard (n=43) |
|---|---|---|---|
| M0 | 0.930 | 0.269 | 0.046 |
| M1 | 0.837 | 0.308 | 0.070 |
| M2 | 0.791 | 0.308 | 0.023 |
| M3 | 0.861 | 0.231 | 0.023 |
| oracle | 0.930 | 0.385 | 0.116 |

By domain (TSR, tf) — v1

| mode | banking (n=16) | ecommerce (n=44) | info_retrieval (n=17) | public_services (n=16) | travel (n=19) |
|---|---|---|---|---|---|
| M0 | 0.562 | 0.182 | 0.706 | 0.750 | 0.421 |
| M1 | 0.625 | 0.204 | 0.706 | 0.625 | 0.316 |
| M2 | 0.625 | 0.159 | 0.647 | 0.688 | 0.210 |
| M3 | 0.562 | 0.182 | 0.588 | 0.562 | 0.421 |
| oracle | 0.625 | 0.250 | 0.765 | 0.750 | 0.474 |

Paired tests vs M0 — v1

| comparison | SSR_tf McNemar (M0 wins / other wins, p) | TSR paired bootstrap diff [95% CI], p |
|---|---|---|
| M0_vs_M1 | 13 / 8, p=0.383 | -0.018 [-0.071, +0.036], p=0.627 |
| M0_vs_M2 | 21 / 6, p=0.00592 | -0.054 [-0.107, +0.000], p=0.0658 |
| M0_vs_M3 | 35 / 12, p=0.00109 | -0.045 [-0.089, +0.000], p=0.0618 |

M3 repeats (v1): TSR per run [0.3929, 0.375, 0.375], mean 0.381, sd 0.0103; SSR_tf per run [0.507, 0.4965, 0.4965]; API errors [0, 0, 0].

### T3. Paired TSR bootstrap between conditions (same 112 trajectories)

| from → to | mode | Δ TSR [95% CI] | p |
|---|---|---|---|
| v1 -> v1.1_v4adapter | M0 | +0.161 [+0.098, +0.232] | 0 |
| v1 -> v1.1_v4adapter | M1 | +0.134 [+0.071, +0.196] | 0 |
| v1 -> v1.1_v4adapter | M2 | +0.125 [+0.071, +0.188] | 0 |
| v1 -> v1.1_v4adapter | M3 | +0.134 [+0.062, +0.205] | 0.0004 |
| v1.1_v4adapter -> v1.1 | M0 | -0.036 [-0.098, +0.027] | 0.308 |
| v1.1_v4adapter -> v1.1 | M1 | +0.009 [-0.071, +0.089] | 0.905 |
| v1.1_v4adapter -> v1.1 | M2 | +0.009 [-0.062, +0.080] | 0.899 |
| v1.1_v4adapter -> v1.1 | M3 | +0.000 [+0.000, +0.000] | 1 |
| v1 -> v1.1 | M0 | +0.125 [+0.045, +0.205] | 0.0028 |
| v1 -> v1.1 | M1 | +0.143 [+0.054, +0.232] | 0.0012 |
| v1 -> v1.1 | M2 | +0.134 [+0.054, +0.214] | 0.0014 |
| v1 -> v1.1 | M3 | +0.134 [+0.062, +0.205] | 0.0004 |
| v1.1_v4adapter -> v1.2 | M0 | +0.089 [+0.045, +0.143] | 0 |
| v1.1_v4adapter -> v1.2 | M1 | +0.054 [+0.000, +0.107] | 0.0612 |
| v1.1_v4adapter -> v1.2 | M2 | +0.018 [-0.027, +0.071] | 0.58 |
| v1.1_v4adapter -> v1.2 | M3 | +0.071 [+0.000, +0.143] | 0.0536 |
| v1 -> v1.2 | M0 | +0.250 [+0.170, +0.330] | 0 |
| v1 -> v1.2 | M1 | +0.188 [+0.107, +0.268] | 0 |
| v1 -> v1.2 | M2 | +0.143 [+0.080, +0.214] | 0 |
| v1 -> v1.2 | M3 | +0.205 [+0.125, +0.286] | 0 |

### T4. Clarification questions (no call, asks the user), teacher-forced

| mode | v1: all steps | v1: on G3-ungrounded steps | v1: on grounded steps | v1.1: all steps (all grounded → wrong) |
|---|---|---|---|---|
| M0 | 0/284 | 0/143 | 0/141 | 0/284 |
| M1 | 0/284 | 0/143 | 0/141 | 0/284 |
| M2 | 0/284 | 0/143 | 0/141 | 0/284 |
| M3 | 7/284 | 7/143 | 0/141 | 0/284 |

### T5-v1. Policies (teacher-forced outcomes), cost relative to always-M0

| policy | TSR | SSR_tf | assumed | tokens | FLOPs |
|---|---|---|---|---|---|
| Fixed M0 | 0.438 | 0.588 | 1.00× | 1.00× | 1.00× |
| Fixed M1 | 0.420 | 0.570 | 3.00× | 1.10× | 1.10× |
| Fixed M2 | 0.384 | 0.535 | 5.00× | 0.64× | 0.64× |
| Fixed M3 | 0.393 | 0.507 | 15.00× | 2.73× | 4.51× |
| Random (uniform, 1000 draws) | 0.410 | 0.550 | 6.00× | 1.37× | 1.81× |
| Length router (terciles → M0/M1/M2) | 0.420 | 0.567 | 3.13× | 0.89× | 0.89× |
| Difficulty escalation (gold tier; semi-oracle) | 0.438 | 0.528 | 10.64× | 2.17× | 3.39× |
| Oracle (cheapest-correct by assumed) | 0.491 | 0.648 | 1.46× | 1.05× | 1.10× |
| Oracle (cheapest-correct by tokens) | 0.482 | 0.648 | 3.56× | 0.86× | 0.91× |
| Oracle (cheapest-correct by flops) | 0.482 | 0.648 | 3.56× | 0.86× | 0.91× |
| Cascade M0→M1 on parse/exec error | 0.446 | 0.592 | 1.03× | 1.01× | 1.01× |
| Cascade M0→M3 on parse/exec error | 0.438 | 0.588 | 1.16× | 1.03× | 1.05× |
| Learned pre-gen λ=0.01 [assumed] (best TSR) | 0.446 | 0.585 | 1.88× | 1.03× | 1.05× |
| Learned pre-gen λ=0.05 [tokens] (best TSR) | 0.446 | 0.585 | 2.56× | 0.98× | 1.01× |
| Learned pre-gen λ=0.05 [flops] (best TSR) | 0.446 | 0.581 | 2.42× | 0.96× | 0.97× |

Oracle savings — v1

| cost model | oracle cost (×M0) | M1 cost (×M0) | oracle saves vs M1 | oracle TSR | M1 TSR |
|---|---|---|---|---|---|
| assumed | 1.465 | 3.000 | 51.2% | 0.491 | 0.420 |
| tokens | 0.858 | 1.097 | 21.8% | 0.482 | 0.420 |
| flops | 0.911 | 1.097 | 16.9% | 0.482 | 0.420 |

Cascade trigger stats — v1: {"trigger_rate": 0.0106, "P(M0 wrong | trigger)": 1.0, "P(trigger | M0 wrong)": 0.0256, "P(escalated mode right | trigger)": {"M1": 0.3333, "M3": 0.0}}

Learned-router AUC (CV) — v1: {"pre-generation": {"M0": 0.795, "M1": 0.739, "M2": 0.786, "M3": 0.764}, "post-M0 (cascade+learned)": {"M0": 0.797, "M1": 0.739, "M2": 0.786, "M3": 0.765}}

Oracle with each M3 repeat — v1: [{"m3_run": 1, "oracle_TSR": 0.4911, "oracle_SSR_tf": 0.6479, "M3_TSR": 0.3929}, {"m3_run": 2, "oracle_TSR": 0.4911, "oracle_SSR_tf": 0.6479, "M3_TSR": 0.375}, {"m3_run": 3, "oracle_TSR": 0.4911, "oracle_SSR_tf": 0.6444, "M3_TSR": 0.375}]

### T5-v1.1_v4adapter. Policies (teacher-forced outcomes), cost relative to always-M0

| policy | TSR | SSR_tf | assumed | tokens | FLOPs |
|---|---|---|---|---|---|
| Fixed M0 | 0.598 | 0.753 | 1.00× | 1.00× | 1.00× |
| Fixed M1 | 0.554 | 0.747 | 3.00× | 1.10× | 1.10× |
| Fixed M2 | 0.509 | 0.697 | 5.00× | 0.65× | 0.66× |
| Fixed M3 | 0.527 | 0.683 | 15.00× | 2.66× | 4.39× |
| Random (uniform, 1000 draws) | 0.549 | 0.720 | 6.00× | 1.35× | 1.79× |
| Length router (terciles → M0/M1/M2) | 0.545 | 0.729 | 3.56× | 0.88× | 0.88× |
| Difficulty escalation (gold tier; semi-oracle) | 0.545 | 0.694 | 10.64× | 2.13× | 3.32× |
| Oracle (cheapest-correct by assumed) | 0.732 | 0.859 | 1.87× | 1.09× | 1.18× |
| Oracle (cheapest-correct by tokens) | 0.723 | 0.859 | 4.57× | 0.84× | 0.94× |
| Oracle (cheapest-correct by flops) | 0.723 | 0.859 | 4.57× | 0.84× | 0.94× |
| Cascade M0→M1 on parse/exec error | 0.607 | 0.757 | 1.01× | 1.00× | 1.00× |
| Cascade M0→M3 on parse/exec error | 0.598 | 0.753 | 1.05× | 1.01× | 1.02× |
| Learned pre-gen λ=0 [assumed] (best TSR) | 0.625 | 0.771 | 3.98× | 1.18× | 1.42× |
| Learned pre-gen λ=0.02 [tokens] (best TSR) | 0.625 | 0.771 | 3.58× | 1.07× | 1.23× |
| Learned pre-gen λ=0.01 [flops] (best TSR) | 0.625 | 0.771 | 3.42× | 1.08× | 1.22× |

Oracle savings — v1.1_v4adapter

| cost model | oracle cost (×M0) | M1 cost (×M0) | oracle saves vs M1 | oracle TSR | M1 TSR |
|---|---|---|---|---|---|
| assumed | 1.866 | 3.000 | 37.8% | 0.732 | 0.554 |
| tokens | 0.844 | 1.098 | 23.1% | 0.723 | 0.554 |
| flops | 0.942 | 1.098 | 14.2% | 0.723 | 0.554 |

Cascade trigger stats — v1.1_v4adapter: {"trigger_rate": 0.0035, "P(M0 wrong | trigger)": 1.0, "P(trigger | M0 wrong)": 0.0143, "P(escalated mode right | trigger)": {"M1": 1.0, "M3": 0.0}}

Learned-router AUC (CV) — v1.1_v4adapter: {"pre-generation": {"M0": 0.707, "M1": 0.693, "M2": 0.728, "M3": 0.741}, "post-M0 (cascade+learned)": {"M0": 0.707, "M1": 0.694, "M2": 0.728, "M3": 0.74}}

Oracle with each M3 repeat — v1.1_v4adapter: [{"m3_run": 1, "oracle_TSR": 0.7321, "oracle_SSR_tf": 0.8592, "M3_TSR": 0.5268}, {"m3_run": 2, "oracle_TSR": 0.7143, "oracle_SSR_tf": 0.8451, "M3_TSR": 0.5357}, {"m3_run": 3, "oracle_TSR": 0.6964, "oracle_SSR_tf": 0.8345, "M3_TSR": 0.5}]

### T5-v1.2. Policies (teacher-forced outcomes), cost relative to always-M0

| policy | TSR | SSR_tf | assumed | tokens | FLOPs |
|---|---|---|---|---|---|
| Fixed M0 | 0.688 | 0.873 | 1.00× | 1.00× | 1.00× |
| Fixed M1 | 0.607 | 0.824 | 3.00× | 1.09× | 1.09× |
| Fixed M2 | 0.527 | 0.768 | 5.00× | 0.65× | 0.66× |
| Fixed M3 | 0.598 | 0.771 | 15.00× | 2.64× | 4.35× |
| Random (uniform, 1000 draws) | 0.598 | 0.809 | 6.00× | 1.35× | 1.78× |
| Length router (terciles → M0/M1/M2) | 0.562 | 0.778 | 3.77× | 0.84× | 0.84× |
| Difficulty escalation (gold tier; semi-oracle) | 0.598 | 0.775 | 10.64× | 2.12× | 3.30× |
| Oracle (cheapest-correct by assumed) | 0.804 | 0.944 | 1.60× | 1.05× | 1.12× |
| Oracle (cheapest-correct by tokens) | 0.795 | 0.944 | 4.58× | 0.79× | 0.86× |
| Oracle (cheapest-correct by flops) | 0.795 | 0.944 | 4.58× | 0.79× | 0.86× |
| Cascade M0→M1 on parse/exec error | 0.696 | 0.877 | 1.01× | 1.00× | 1.00× |
| Cascade M0→M3 on parse/exec error | 0.688 | 0.873 | 1.05× | 1.01× | 1.01× |
| Learned pre-gen λ=0 [assumed] (best TSR) | 0.705 | 0.863 | 2.57× | 1.10× | 1.24× |
| Learned pre-gen λ=0.01 [tokens] (best TSR) | 0.705 | 0.873 | 2.23× | 1.04× | 1.13× |
| Learned pre-gen λ=0 [flops] (best TSR) | 0.705 | 0.863 | 2.57× | 1.10× | 1.24× |

Oracle savings — v1.2

| cost model | oracle cost (×M0) | M1 cost (×M0) | oracle saves vs M1 | oracle TSR | M1 TSR |
|---|---|---|---|---|---|
| assumed | 1.599 | 3.000 | 46.7% | 0.804 | 0.607 |
| tokens | 0.789 | 1.094 | 27.9% | 0.795 | 0.607 |
| flops | 0.857 | 1.094 | 21.7% | 0.795 | 0.607 |

Cascade trigger stats — v1.2: {"trigger_rate": 0.0035, "P(M0 wrong | trigger)": 1.0, "P(trigger | M0 wrong)": 0.0278, "P(escalated mode right | trigger)": {"M1": 1.0, "M3": 0.0}}

Learned-router AUC (CV) — v1.2: {"pre-generation": {"M0": 0.614, "M1": 0.561, "M2": 0.619, "M3": 0.579}, "post-M0 (cascade+learned)": {"M0": 0.615, "M1": 0.561, "M2": 0.619, "M3": 0.579}}

Oracle with each M3 repeat — v1.2: [{"m3_run": 1, "oracle_TSR": 0.8036, "oracle_SSR_tf": 0.9437, "M3_TSR": 0.5982}]

### T5-v1.1. Policies (teacher-forced outcomes), cost relative to always-M0

| policy | TSR | SSR_tf | assumed | tokens | FLOPs |
|---|---|---|---|---|---|
| Fixed M0 | 0.562 | 0.768 | 1.00× | 1.00× | 1.00× |
| Fixed M1 | 0.562 | 0.785 | 3.00× | 1.07× | 1.07× |
| Fixed M2 | 0.518 | 0.725 | 5.00× | 0.65× | 0.66× |
| Fixed M3 | 0.527 | 0.683 | 15.00× | 2.66× | 4.39× |
| Random (uniform, 1000 draws) | 0.543 | 0.740 | 6.00× | 1.34× | 1.78× |
| Length router (terciles → M0/M1/M2) | 0.545 | 0.764 | 3.56× | 0.86× | 0.87× |
| Difficulty escalation (gold tier; semi-oracle) | 0.518 | 0.687 | 10.64× | 2.12× | 3.31× |
| Oracle (cheapest-correct by assumed) | 0.688 | 0.866 | 1.83× | 1.08× | 1.16× |
| Oracle (cheapest-correct by tokens) | 0.696 | 0.866 | 4.61× | 0.82× | 0.91× |
| Oracle (cheapest-correct by flops) | 0.696 | 0.866 | 4.61× | 0.82× | 0.91× |
| Cascade M0→M1 on parse/exec error | 0.562 | 0.768 | 1.04× | 1.01× | 1.01× |
| Cascade M0→M3 on parse/exec error | 0.571 | 0.775 | 1.21× | 1.04× | 1.06× |
| Learned pre-gen λ=0 [assumed] (best TSR) | 0.571 | 0.785 | 4.23× | 1.20× | 1.41× |
| Learned pre-gen λ=0.02 [tokens] (best TSR) | 0.580 | 0.792 | 3.47× | 1.08× | 1.18× |
| Learned pre-gen λ=0.01 [flops] (best TSR) | 0.580 | 0.792 | 3.45× | 1.08× | 1.19× |

Oracle savings — v1.1

| cost model | oracle cost (×M0) | M1 cost (×M0) | oracle saves vs M1 | oracle TSR | M1 TSR |
|---|---|---|---|---|---|
| assumed | 1.831 | 3.000 | 39.0% | 0.688 | 0.562 |
| tokens | 0.825 | 1.066 | 22.6% | 0.696 | 0.562 |
| flops | 0.913 | 1.066 | 14.4% | 0.696 | 0.562 |

Cascade trigger stats — v1.1: {"trigger_rate": 0.0141, "P(M0 wrong | trigger)": 1.0, "P(trigger | M0 wrong)": 0.0606, "P(escalated mode right | trigger)": {"M1": 0.0, "M3": 0.5}}

Learned-router AUC (CV) — v1.1: {"pre-generation": {"M0": 0.644, "M1": 0.619, "M2": 0.666, "M3": 0.738}, "post-M0 (cascade+learned)": {"M0": 0.68, "M1": 0.656, "M2": 0.675, "M3": 0.737}}

Oracle with each M3 repeat — v1.1: [{"m3_run": 1, "oracle_TSR": 0.6875, "oracle_SSR_tf": 0.8662, "M3_TSR": 0.5268}, {"m3_run": 2, "oracle_TSR": 0.6786, "oracle_SSR_tf": 0.8627, "M3_TSR": 0.5357}, {"m3_run": 3, "oracle_TSR": 0.6518, "oracle_SSR_tf": 0.8486, "M3_TSR": 0.5}]

### T6. Step 6 — M3 variants on v1.1 eval (teacher-forced)

| variant | model | TSR | SSR_tf | TSR hard | clarify steps | mean tokens (in/out) | FLOPs/step | USD/284 | API errors |
|---|---|---|---|---|---|---|---|---|---|
| m3_gptoss_0shot_run1_v6 | openai/gpt-oss-120b | 0.527 [0.43, 0.62] | 0.683 [0.63, 0.73] | 0.163 | 0 | 1384/94 | 1.51e+13 | 0.0790 | 0 |
| m3_gptoss_3shot_v6 | openai/gpt-oss-120b | 0.589 [0.50, 0.68] | 0.722 [0.67, 0.77] | 0.186 | 0 | 1769/101 | 1.91e+13 | 0.0968 | 0 |
| m3_qwen27b_0shot_v6 | qwen/qwen3.8-27b | 0.625 [0.53, 0.71] | 0.757 [0.70, 0.80] | 0.279 | 0 | 1375/36 | 7.62e+13 | 0.3531 | 0 |
| m3_qwen27b_3shot_v6 | qwen/qwen3.8-27b | 0.616 [0.52, 0.70] | 0.753 [0.70, 0.80] | 0.209 | 0 | 1822/41 | 1.01e+14 | 0.4601 | 0 |
| m3_gptoss_0shot_run1_v12 (v1.2 eval) | openai/gpt-oss-120b | 0.598 [0.51, 0.68] | 0.771 [0.72, 0.82] | 0.279 | 1 | 1388/93 | 1.51e+13 | 0.0790 | 0 |
| m3_qwen27b_0shot_v12 (v1.2 eval) | qwen/qwen3.8-27b | 0.723 [0.63, 0.80] | 0.884 [0.84, 0.92] | 0.512 | 0 | 1379/36 | 7.64e+13 | 0.3547 | 0 |

### T7. G3 v1.2 phrase table (Vietnamese wording → enum value)

An enum value is grounded only if the query matches one of these patterns (regex, on the lower-cased query and on its accent-free form), or contains the value itself, or the value comes from an earlier observation. Source: `data/grounding_gate.py::ENUM_PHRASES`.

| tool.param | value | query patterns |
|---|---|---|
| block_card.reason | `lost` | `\bmất\b`, `đánh rơi`, `thất lạc`, `\blost\b` |
| block_card.reason | `stolen` | `bị trộm`, `bị cắp`, `bị móc`, `\btrộm\b`, `\bstolen\b` |
| block_card.reason | `damaged` | `\bhỏng\b`, `\bhư\b`, `\bgãy\b`, `không (dùng\|sử dụng\|quẹt) được`, `\bdamaged\b` |
| block_card.reason | `fraud` | `gian lận`, `lừa đảo`, `giao dịch lạ`, `bị hack`, `đáng ngờ`, `\bfraud\b` |
| loan_inquiry.loan_type | `home` | `mua nhà`, `\bnhà\b`, `bất động sản`, `nhà ở` |
| loan_inquiry.loan_type | `car` | `ô tô`, `xe hơi`, `mua xe`, `\boto\b` |
| loan_inquiry.loan_type | `personal` | `tiêu dùng`, `cá nhân`, `tín chấp` |
| loan_inquiry.loan_type | `business` | `kinh doanh`, `doanh nghiệp` |
| get_interest_rates.product_type | `savings` | `tiết kiệm`, `gửi tiền`, `tiền gửi` |
| get_interest_rates.product_type | `loan` | `\bvay\b`, `cho vay`, `khoản vay` |
| get_interest_rates.product_type | `credit_card` | `thẻ tín dụng`, `credit card` |
| find_branch.type | `branch` | `chi nhánh`, `phòng giao dịch` |
| find_branch.type | `atm` | `\batm\b`, `cây rút tiền`, `máy rút tiền` |
| civil_status_lookup.document_type | `birth` | `khai sinh` |
| civil_status_lookup.document_type | `marriage` | `kết hôn`, `hôn nhân` |
| civil_status_lookup.document_type | `death` | `khai tử` |
| civil_status_lookup.document_type | `residence` | `cư trú`, `tạm trú`, `thường trú`, `hộ khẩu` |
| utility_bill_inquiry.utility_type | `electricity` | `\bđiện\b(?!\s*(thoại\|tử\|máy\|ảnh))`, `\bevn\b` |
| utility_bill_inquiry.utility_type | `water` | `(tiền\|hóa đơn\|hoá đơn\|cấp\|cước\|điện)\s*nước`, `điện\s*(và\|,)\s*nước`, `nước\s*(sạch\|sinh hoạt\|máy)` |
| search_product.platform | `shopee` | `shopee` |
| search_product.platform | `lazada` | `lazada` |
| search_product.platform | `tiki` | `\btiki\b` |
| search_product.platform | `sendo` | `sendo` |
| checkout.payment_method | `cod` | `\bcod\b`, `khi nhận hàng`, `lúc nhận hàng`, `tiền mặt`, `nhận hàng (rồi\|mới) (trả\|thanh toán)` |
| checkout.payment_method | `momo` | `momo` |
| checkout.payment_method | `vnpay` | `vn\s?pay` |
| checkout.payment_method | `bank_transfer` | `chuyển khoản` |
| return_request.refund_method | `original` | `phương thức (thanh toán )?ban đầu`, `cách thanh toán ban đầu`, `hoàn (tiền )?về (nguồn\|phương thức)`, `như lúc thanh toán` |
| return_request.refund_method | `wallet` | `\bví\b` |
| return_request.refund_method | `bank` | `ngân hàng`, `tài khoản` |
| book_flight.seat_class | `economy` | `phổ thông`, `\beconomy\b`, `hạng thường` |
| book_flight.seat_class | `business` | `thương gia`, `\bbusiness\b` |
| book_flight.seat_class | `first` | `hạng nhất`, `first class` |
| news_search.category | `politics` | `chính trị`, `thời sự` |
| news_search.category | `economy` | `kinh tế`, `tài chính`, `cổ phiếu`, `chứng khoán` |
| news_search.category | `sports` | `thể thao`, `bóng đá` |
| news_search.category | `tech` | `công nghệ` |
| news_search.category | `entertainment` | `giải trí`, `showbiz` |
| open_savings_account.term_months | `12` | `(1\|một) năm` |
| open_savings_account.term_months | `24` | `(2\|hai) năm` |
| open_savings_account.term_months | `36` | `(3\|ba) năm` |
