# BudgetSwitch

> Learned Adaptive Compute Allocation for Tool-Using SLM Agents via Outcome-Verifiable GRPO

**Project status:** Week 1 of 24, Foundation phase.
**Team:** 2 third-year AI Engineering undergraduates + 1 PhD advisor (TBD)
**Target:** RIVF 2026 or SOICT 2026 submission
**Resource:** Kaggle T4×2, 4 accounts, ~50 GPU-hours total

---

## Quick Start for New Contributors / New Chat Tabs

If you're joining this project (or this is a new Claude/LLM tab), read these documents IN ORDER:

1. **[`docs/README_BudgetSwitch_v9.2_FINAL.md`](docs/README_BudgetSwitch_v9.2_FINAL.md)** — Full proposal (19 sections, ~18,000 words). Single source of truth for the project.

2. **[`CRITICAL_RULES.md`](CRITICAL_RULES.md)** — 3 non-negotiable implementation rules. Read before writing any reward function code.

3. **[`docs/IMPLEMENTATION_PITFALLS.md`](docs/IMPLEMENTATION_PITFALLS.md)** — 2 technical pitfalls to verify at W3-W4 and W9.

4. **[`docs/BudgetSwitch_Proposal.pdf`](docs/BudgetSwitch_Proposal.pdf)** — Formal proposal sent to advisor.

---

## Folder Structure

```
budgetswitch/
├── CRITICAL_RULES.md          ← Read this FIRST
├── README.md                  ← This file
│
├── modes/                     ← 4 compute mode implementations
│   ├── m0_fast/              ← Direct generation
│   ├── m1_think/             ← Internal CoT
│   ├── m2_think_search/      ← CoT + retrieval
│   └── m3_defer/             ← LLM API defer
│
├── verifiers/                 ← Rule-based outcome verifiers (NO oracle)
│   └── universal/            ← Schema, tool, param, outcome checks
│
├── data/                      ← Dataset construction
│   ├── vitool/               ← ViTool-2.0K (to build, ~2000 trajectories)
│   ├── bfcl_vi/              ← BFCL-v3 → VN translations
│   ├── massive_vn/           ← MASSIVE-Agents VN subset
│   ├── cross_lingual/        ← EN/Thai/Indonesian transfer eval
│   └── tau_bench/            ← τ-bench Retail subset
│
├── pilot/                     ← §6 Pilot study scripts
│
├── training/                  ← SFT + GRPO pipelines
│   └── verification/         ← Sanity check scripts (run before training)
│
├── baselines/                 ← ReDAct, ATLAS, fixed-mode replicates
│
├── eval/                      ← Metrics, Pareto plots
│
├── embeddings/                ← Pre-computed tool embeddings (W3)
│
├── notebooks/                 ← Kaggle notebooks (one per task)
│
├── docs/                      ← All planning documents
│
├── tests/                     ← Unit tests
│
├── scripts/                   ← Bash utilities
│
└── paper/                     ← LaTeX source (W17+)
```

---

## Current Week: W1 (Foundation)

Daily tasks: see [`docs/W1_DAILY_PLAN.md`](docs/W1_DAILY_PLAN.md).

**Both members read this week:**
1. NVIDIA SLM Position (arXiv 2506.02153)
2. BCAE (arXiv 2604.19780)
3. ATLAS (arXiv 2603.06713)
4. Fission-GRPO (arXiv 2601.15625)
5. ReDAct (arXiv 2604.07036)
6. Vi-S1K (Vietnamese SLM Test-Time Scaling, Jan 2026)

Member A reads papers 1, 3, 5; Member B reads papers 2, 4, 6. Then swap weekly notes.

---

## Critical Rules (Summary)

See [`CRITICAL_RULES.md`](CRITICAL_RULES.md) for full details.

1. **Reward function MUST NOT reference oracle_mode** in any form
2. **Only mode-routing LoRA trains** during GRPO; tool-use LoRA frozen
3. **Mode tokens = regular text + constrained decoding** (no tokenizer modification)

---

## Contact

- **Member A:** [Name, email]
- **Member B:** [Name, email]
- **Advisor:** [TBD — email sent W1 Day 5]

---

*Last updated: W1 Day 1.*
