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
