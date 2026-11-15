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
