# Number check — paper/vitool_camera_ready.tex vs eval/v6 outputs (2026-11-13)

The .tex was NOT modified. Sources: `metrics_v6.json` (M), `routers_v6.json` (R), `cost_v6.json` (C),
`checks_validation.json`, `audit_validation.json`, `g3v12_report_v1.json`, data files.

**Why most mismatches exist.** The v1.2 M3 run (gpt-oss-120b 0-shot) had 5 steps that failed with
`APITimeoutError` after all retries and were scored as wrong. They were rerun on 2026-11-13 (original
file kept as `m3_gptoss_0shot_run1_v12.with_api_errors.jsonl.bak`). M3 v1.2 TSR moves from 0.571 to
**0.598**, and with it every number that depends on M3 on v1.2. The most important consequence:
**M0 vs M3 on TSR is no longer significant (−0.089, p = 0.051)**; the paper's claims that direct calling
"significantly" beats deferral must be softened (step level is still significant: McNemar 46 vs 17,
p = 0.0003).

## A. Mismatches (paper number → correct number)

| # | line(s) | paper | correct | source |
|---|---|---|---|---|
| 1 | 58–60 (abstract) | "significantly outperforms … zero-shot deferral to gpt-oss-120b (0.571)" | M3 TSR **0.598**; M0 − M3 = −0.089, **p = 0.051 (not significant)** on TSR | M.conditions.v1.2.teacher_forced.overall / paired_tests_vs_M0.M0_vs_M3 |
| 2 | 102–103 (contrib.) | "thinking, retrieval and zero-shot deferral are significantly worse" | deferral not significant on TSR (p = 0.051); thinking p = 0.019, retrieval p < 0.001 | same |
| 3 | 271 (Tab. I, M3) | 0.571 [.48,.66] | **0.598 [.51,.68]** | M…overall.M3.TSR |
| 4 | 271 | SSR_tf 0.757 | **0.771** | M…overall.M3.SSR_tf |
| 5 | 271 | easy 0.88 | **0.95** | M…by_difficulty.easy.M3 |
| 6 | 271 | FLOPs 4.28× | **4.35×** | C.v1.2.modes.M3.relative_to_M0.flops |
| 7 | 291–292 | "deferral (−0.116, p = 0.016)" significant | **−0.089, p = 0.051** | paired_tests_vs_M0.M0_vs_M3 |
| 8 | 293 | McNemar "50 vs. 17 against M3" | **46 vs 17** (p = 0.0003) | M0_vs_M3.SSR_tf_mcnemar |
| 9 | 303 | "reaches 0.571 on v1.2 at 4.3× M0's FLOPs" | **0.598 at 4.35×** (rounds to 4.4×) | M.m3_on_v12 / C |
| 10 | 328 (Tab. II) | v1.2 gpt-oss 0-shot 0.571, 4.3×, 0.078 | **0.598, 4.4×** (4.35), USD **0.079** | M.m3_on_v12.m3_gptoss_0shot_run1_v12 |
| 11 | 329 (Tab. II) | Qwen v1.2: pending, 22×, – | TSR **0.688** (277/284 steps valid; 6 timed-out steps being rerun, 5 left at writing), FLOPs **21.6×**, USD **0.35** — final value after rerun, see §C | M.m3_on_v12 (bak) |
| 12 | 359 (Tab. III, M3 v1.2) | 0.571 | **0.598** | M.v1_vs_v11_table "TSR M3" |
| 13 | 392 (Tab. IV, Fixed M3) | 0.571 / 15.00 / 2.59 / 4.28 | **0.598** / 15.00 / **2.64** / **4.35** | R.v1.2.policies "Fixed M3" |
| 14 | 394 (Random) | 0.591 / 6.00 / 1.33 / 1.76 | **0.598** / 6.00 / **1.35** / **1.78** | R "Random" |
| 15 | 397 (Learned best) | 0.705 / 2.04 / 1.02 / 1.09 | 0.705 (best now λ = 0.01 [tokens]) / **2.23 / 1.04 / 1.13** | R learned-tokens, max TSR |
| 16 | 398 (Gold-tier) | 0.598 / 10.64 / 2.10 / 3.27 | 0.598 / 10.64 / **2.12 / 3.30** | R "Difficulty escalation" |
| 17 | 400 (Oracle row) | 0.804 / 1.60 / 0.79 / 0.86 | the row mixes two oracles: assumed-cost oracle = 0.804 / 1.60 / **1.05 / 1.12**; tokens/FLOPs oracle = **0.795** / 4.58 / 0.79 / 0.86. Either split the row or caption it ("cost of the oracle chosen under each cost model") | R "Oracle (cheapest-correct by …)" |
| 18 | 309 / 323 | v1.1 3 runs "0.518 ± 0.016", "±0.016" | correct for the files before rerun (runs 0.527 / 0.527 / 0.500; sd 0.0155), but runs 2–3 had 2 + 1 timed-out steps now being rerun → may change slightly | M3 run files (.bak) |
| 19 | 305, 324 | v1.1 gpt-oss 3-shot 0.571, 5.5×, 0.096 | correct before rerun; 3 timed-out steps being rerun → may change | m3_gptoss_3shot_v6 (.bak) |
| 20 | 187–188 | "at least 39 of the 43 hard ones" | **40 of 43** under G3 v1.2 (39/43 was the v1.1 rule) | g3v12_report_v1.json |
| 21 | 407–410 (Fig. 2) | `fig2_pareto_v6_v12.pdf`, "Points are numbered" | new single-column figure `fig2_pareto_v6_v12_col.pdf` has short labels, no numbers; caption must change. Both figures must be regenerated after the M3 rerun (Fixed M3, random, oracle points move) | figures_v6.py |
| 22 | 4, 223 | REPORT/adapter: "p = 0.31" | p = 0.308 now (0.297 in the earlier REPORT: bootstrap p depends on RNG order) — fine to keep "0.31" | M.paired…v1.1_v4adapter → v1.1 |

## B. Checked and correct

36 tools; 373 trajectories (e-commerce 145, travel 64, info 56, banking 55, public services 53);
v1 train 263 trajectories / 719 steps; 54% = 203/375 ungrounded, 63/112 eval; audit 60 values (40/20),
86.7%, κ = 0.76, 8 adjudicated, 60/60; repair median 30 chars, 91% insertions; rewrite check 50/50,
naturalness 2.48 / 2.54, weighted κ = 0.16; spot checks 3 and 4 of 30 (v1.1), 0/20 (v1.2, Wilson
upper 0.16); Tab. I rows M0, M1, M2, Oracle (all values incl. CIs, SSR_ro 0.722 / 0.665 / 0.532, tiers,
562 tokens, 3.5e12 FLOPs, 1.09× / 0.66×); M0 vs M1 −0.080 p = 0.019, McNemar 19 vs 5; M0 vs M2
−0.161 p < 0.001; easy: M1 loses 4, M2 loses 6; public services M3 0.875 vs M0 0.812; e-commerce 0.48
vs 0.74–0.94; v1 → v1.2 gains 0.14–0.25 (p < 0.001; M3 now +0.205); M0 hard 0.05 → 0.42; v1 modes not
significant (p ≥ 0.06); headroom 0.053 / 0.134 / 0.116; M3 asked 7/284 on v1, all on ungrounded steps;
Tab. III except M3 v1.2; SSR 0.873 vs 0.722 (v1 0.588 vs 0.468); TSR 0.688 vs 0.679, M1 0.607 both;
Tab. IV rows Fixed M0/M1/M2, length, cascade; savings 46.7 / 27.9 / 21.7%, 0.197 higher, v1 51.2 vs
16.9%; latency medians 6.0 / 9.9 / 8.3 s, M3 0.9 s; best router +0.017 (2 trajectories); AUC
0.56–0.62 (v1 0.74–0.80); cascade trigger 0.4%, +0.009; Qwen v1.1 0.625 vs 0.598, 22×, 4.5× price
(0.353 / 0.079); e-commerce 44/112; 4 of 50 gold oddities; relaxed metric 1 of 20.

## C. Not verifiable from v6 outputs

- Lines 149–151: v1 human review (51 trajectories, 45 accepted, 92.2%, AC1 = 0.91) — from the original
  paper / earlier pipeline, not in eval/v6.
- Qwen3.8-27B 0-shot on v1.2 final value: rerun of the timed-out steps still running; rerun
  `bash eval/v6/build_report_v6.sh` and read `m3_on_v12.m3_qwen27b_0shot_v12` when complete.

## D. Final values after all reruns (2026-11-15) — supersede rows 11, 18, 19 above

All M3 runs are complete with 0 API errors (timed-out steps rerun; originals kept as `*.with_api_errors.jsonl.bak`).

| item | paper | final | source |
|---|---|---|---|
| Tab. II, Qwen3.8-27B 0-shot on v1.2 (`\QwenVtwo`) | pending | **0.723** [0.63, 0.80]; SSR_tf 0.884; FLOPs **22×**; USD **0.355**; hard 0.512 | M.m3_on_v12.m3_qwen27b_0shot_v12 |
| Qwen v1.2 vs M0 (same 112 trajectories) | – | +0.036 TSR, paired bootstrap **p = 0.45**; steps McNemar 20 vs 23, p = 0.76 (not significant) | computed from pilot_results_v12 + Qwen file |
| Tab. II, v1.1 gpt-oss 0-shot, 3 runs | 0.518 ± 0.016 | **0.521 ± 0.019** (runs 0.527 / 0.536 / 0.500) | M.conditions.v1.1_v4adapter.M3_repeats |
| line 309 "varied by ±0.016" | ±0.016 | **±0.019** (sd) | same |
| Tab. II / line 305, v1.1 gpt-oss 3-shot | 0.571, 5.5×, 0.096 | **0.589**, **5.6×**, **0.097** | M.step6_m3_variants_v11.m3_gptoss_3shot_v6 |
| v1.1 Qwen 3-shot (not in paper) | – | 0.616, 29×, 0.460 | m3_qwen27b_3shot_v6 |
| line 306–307 "Qwen … 0.625 … about 22× … 4.5× price" (v1.1) | unchanged | 0.625, 22×, 0.353 / 0.079 = 4.5× ✓ | |

Wording to revisit: on v1.2 a dense 27B model (zero-shot) scores **above** the 3B M0 (0.723 vs 0.688, not
significant) at 22× FLOPs — the claims "spending more compute per step does not help on average" and
"larger models therefore can help, at a cost far above …" should mention this number.
