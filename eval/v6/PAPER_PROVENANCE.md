# Provenance of every number in paper/vitool_camera_ready.tex

All paths are in this repository. "Rebuild" = `bash eval/v6/build_report_v6.sh`
(runs `eval/v6/analyze_v6.py` → `metrics_v6.json` + `pilot_results_*.jsonl`,
`eval/v6/routers_v6.py` → `routers_v6.json` + `cost_v6.json`, and
`eval/v6/report_tables_v6.py` → `REPORT_v6.md`). Per-step inputs are listed in §B.
`eval/learned_router_cv.py` (an older script on v4/v1 data) is **not** used by the paper.

## A. Paper → script → result file

| Paper item | Numbers | Script | Result file / key |
|---|---|---|---|
| Abstract, §III-A | 373 trajectories, 5 domains, 36 tools; domain counts | — (data) | `data/vitool/vitool_v1.2.jsonl`, `data/vitool/tool_catalog.json` |
| Abstract, §III-C "Extent" | 54% = 203/375; 63/112 eval; 40/43 hard | `data/build_dataset.py --gate-report` (G3 = `data/grounding_gate.py`) | `eval/v6/g3v12_report_v1.json` |
| §III-B | v1 review: 51 trajectories, 45 accepted, 92.2%, AC1 0.91 | `data/verify_cli.py` (v1 era) | `data/feedback/member{1,2}.jsonl`; `docs/13_06.md` |
| Alg. 1 | gates | `data/build_dataset.py::accept_trajectory`, `data/grounding_gate.py` | `tests/test_grounding_gate.py` |
| §III-C "Validation of the gate" | 60 values (40/20), 86.7%, κ 0.76, 8 adjudicated, 60/60 | `eval/v6/audit_sample_v6.py`, `eval/v6/score_audit_v6.py` | `eval/v6/audit_validation.json`, `audit_sample_adjudicated.json`, `g3_vs_adjudicated.json` |
| Table I (tab:examples) | 2 example trajectories | (export script inline in session; values reproducible from data) | `eval/v6/paper_examples.json` |
| §III-C "Repair" | 2 rounds, 2 removed, 373 pass, median 30 chars, 91% insertions | `eval/v6/rewrite_queries_v6.py --target v1.1 / v1.2` | `eval/v6/rewrite_meta.json`, `rewrite_meta_v12.json`, `g3_report_v12.json`, `rewrite_stats_v12.json`, `query_diffs*.jsonl` |
| §III-C "Human checks" | 50/50; 2.48 / 2.54; weighted κ 0.16; 3 and 4 / 30; 0/20, Wilson 0.16 | `eval/v6/score_checks_v6.py` | `eval/v6/checks_validation.json` (`rewrite_check`, `v11_spotcheck`, `v12_spotcheck`) |
| §IV-A | v1 train 263 / 719 steps | — | `data/vitool/sft_train.jsonl` |
| §IV-A retrain | Δ −0.036, p = 0.31 | `analyze_v6.py` | `metrics_v6.json` → `paired_TSR_bootstrap_between_conditions["v1.1_v4adapter -> v1.1"].M0` |
| Table II (tab:main) | TSR + CI, SSR_tf, tiers (M0–M3, oracle) | `analyze_v6.py` | `metrics_v6.json` → `conditions["v1.2"].teacher_forced.{overall,by_difficulty}` |
| Table II | SSR_ro | `analyze_v6.py` | `conditions["v1.2"].rollout.*.SSR_rollout` |
| Table II | FLOPs ×, 562 tokens, 3.5e12 FLOPs | `routers_v6.py` | `cost_v6.json` → `v1.2.modes.*` |
| Fig. 1 | heatmap | `eval/v6/figures_v6.py` (`fig1`) | `paper/fig1_heatmap_v6_v12.pdf` (copy of `eval/v6/…`) |
| §V-A | p-values vs M0 (0.019, <0.001, 0.051), McNemar 46/17, 19/5, domains, easy losses | `analyze_v6.py` | `conditions["v1.2"].teacher_forced.{paired_tests_vs_M0,by_domain,by_difficulty}` |
| §V-B, Table III (tab:defer) | gpt-oss / Qwen TSR, FLOPs ×, USD (v1.1 and v1.2) | `analyze_v6.py::m3_variants` | `metrics_v6.json` → `step6_m3_variants_v11`, `m3_on_v12`; M0 FLOPs from `cost_v6.json` |
| §V-B | 3 runs 0.521 ± 0.019 | `analyze_v6.py` | `conditions["v1.1_v4adapter"].M3_repeats` |
| §V-B | Qwen v1.2 vs M0: +0.036, p = 0.45; 23 vs 20, p = 0.76 | paired bootstrap / McNemar on the files below | `eval/v6/paper_number_check.md` §D (computed from `pilot_results_v12.jsonl` + `m3_v12/m3_qwen27b_0shot_v12.jsonl`) |
| §V-C, Table IV (tab:versions) | v1 / v1.1 / v1.2 TSR, hard tier, oracle − best | `analyze_v6.py` | `metrics_v6.json` → `v1_vs_v11_table`, `conditions[*]` |
| §V-C | +0.14–0.25, p < 0.001; v1 not significant; headroom 0.053 / 0.116 | `analyze_v6.py` | `paired_TSR_bootstrap_between_conditions["v1 -> v1.2"]`, `conditions["v1"].teacher_forced.paired_tests_vs_M0` |
| §V-C | M3 asked 7/284, all ungrounded | `analyze_v6.py` | `conditions["v1"].clarification.M3` |
| §V-D | SSR 0.873 vs 0.722 (v1 0.588 vs 0.468); TSR 0.688 vs 0.679 | `analyze_v6.py` | `conditions["v1.2"/"v1"].{teacher_forced,rollout}` |
| Table V (tab:routers) | all policies × 3 cost models, both oracles | `routers_v6.py` (policy code from `eval/v5/routers_v5.py`) | `routers_v6.json` → `v1.2.policies` |
| Fig. 2 | Pareto (single column) | `figures_v6.py::fig2_col` | `paper/fig2_pareto_v6_v12_col.pdf` |
| §VI | savings 46.7 / 27.9 / 21.7%, +0.197; v1 51.2 vs 16.9% | `routers_v6.py` | `routers_v6.json` → `{v1.2,v1}.oracle_savings` |
| §VI | latency medians 6.0 / 9.9 / 8.3 / 0.9 s | `routers_v6.py` | `cost_v6.json` → `v1.2.modes.*.latency_ms_median` |
| §VI | +0.017, AUC 0.56–0.62 (v1 0.74–0.80), trigger 0.4%, +0.009 | `routers_v6.py` | `routers_v6.json` → `{v1.2,v1}.learned_router_auc`, `v1.2.cascade_trigger_stats`, `v1.2.policies` |
| §VII | 44/112 e-commerce; 4 of 50 gold oddities; relaxed metric 1 of 20 | — / `score_checks_v6.py` / `eval/v5/relaxed_v5.py` | `pilot_eval_v1.2.jsonl`; `rewrite_check_annotator2.json` notes; `eval/v5/relaxed_examples_v5.jsonl` |

## B. Per-step result files (inputs to the scripts above)

| Condition | Mode / run | File |
|---|---|---|
| v1.2 | M0/M1/M2 teacher-forced (v4 adapter) | `eval/v6/raw/local_tf_v4adapter_v12.jsonl` (+ `.meta.json`) |
| v1.2 | M0/M1/M2 rollout | `eval/v6/raw/local_rollout_v4adapter_v12.jsonl` (+ `.meta.json`) |
| v1.2 | M3 gpt-oss-120b 0-shot | `eval/v6/m3_v12/m3_gptoss_0shot_run1_v12.jsonl` (+ `.meta.json`) |
| v1.2 | Qwen3.8-27B 0-shot | `eval/v6/m3_v12/m3_qwen27b_0shot_v12.jsonl` (+ `.meta.json`) |
| v1.2 | merged | `eval/v6/pilot_results_v12.jsonl` |
| v1.1 | M0–M2 tf / rollout, v4 adapter | `eval/v6/raw/local_{tf,rollout}_v4adapter_v11.jsonl` |
| v1.1 | M0–M2 tf / rollout, retrained adapter | `eval/v6/raw/local_{tf,rollout}_v6adapter_v11.jsonl` |
| v1.1 | M3 runs 1–3 | `eval/v6/m3/m3_run{1,2,3}_v6.jsonl` |
| v1.1 | gpt-oss 3-shot, Qwen 0-/3-shot | `eval/v6/m3_variants/m3_{gptoss_3shot,qwen27b_0shot,qwen27b_3shot}_v6.jsonl` |
| v1 | M0–M2 tf / rollout, M3 runs 1–3 | `eval/v5/raw/local_{tf,rollout}_v5.jsonl`, `eval/v5/m3/m3_run{1,2,3}_v5.jsonl` |

Rows of the M3 files that originally ended in an API timeout were rerun; the
pre-rerun files are not in the repository (git-ignored `*.bak`).
