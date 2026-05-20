# Week 1 Retrospective

**Period:** [Start date] to [End date]
**Members present:** Member A, Member B (+ advisor if joined)
**Filled by:** _____ on _____

---

## What was completed

### Repository setup
- [ ] Repo created at [URL]
- [ ] Folder structure matches `docs/W1_DAILY_PLAN.md`
- [ ] `CRITICAL_RULES.md` reviewed and understood by both members
- [ ] `.gitignore` and `requirements.txt` in place

### Reading (6 Tier-1 papers)
- [ ] NVIDIA SLM Position — read by: _____ (notes in `docs/paper_notes/nvidia_slm.md`)
- [ ] BCAE — read by: _____ (notes in `docs/paper_notes/bcae.md`)
- [ ] ATLAS — read by: _____ (notes in `docs/paper_notes/atlas.md`)
- [ ] Fission-GRPO — read by: _____ (notes in `docs/paper_notes/fission_grpo.md`)
- [ ] ReDAct — read by: _____ (notes in `docs/paper_notes/redact.md`)
- [ ] Vi-S1K — read by: _____ (notes in `docs/paper_notes/vi_s1k.md`)

### Code
- [ ] M0 fast mode implemented (`modes/m0_fast/m0_fast.py`)
- [ ] Schema/outcome verifier implemented (`verifiers/universal/schema_verifier.py`)
- [ ] Tool catalog skeleton (5 tools) (`data/vitool/tool_catalog.json`)
- [ ] Test files: `tests/test_m0_prompt.py` passes all 5 tests
- [ ] Pre-commit hook: `scripts/verify_no_oracle_leak.sh` passes

### Pipeline verification
- [ ] Kaggle GRPO smoke test (`notebooks/W1_grpo_smoke_test.ipynb`) ran on T4
- [ ] All 5 GRPO steps completed without exception
- [ ] Peak VRAM: _____ GB (target: ≤ 14 GB)
- [ ] Estimated full training time: _____ hours

### Advisor outreach
- [ ] Email sent to advisor 1: [Name] on [date]
- [ ] Email sent to advisor 2: [Name] on [date]
- [ ] Email sent to advisor 3: [Name] on [date]
- [ ] Reply status: ___ confirmed, ___ pending, ___ declined

---

## What was harder than expected

(Be honest. This shapes the next week's planning.)

1. 
2. 
3. 

---

## What was easier than expected

(Calibrate confidence for upcoming weeks.)

1. 
2. 

---

## Open questions

### For the advisor (when meeting happens)
- 
- 

### To research independently
- 
- 

### Decisions deferred to W2 with rationale
- 

---

## Risks identified this week

(New risks not in original proposal §10)

| Risk | Severity | Mitigation idea |
|---|---|---|
| | | |

---

## Velocity check

**Hours worked per member:** _____ / _____ (target: ~15-20h each)

**Are we on track for W2?** Yes / No

**If No:** What needs to slip vs. what stays?

---

## Update to project state

### Critical Rules status
- [ ] Rule 1: All reward-related code verified clean (run `bash scripts/verify_no_oracle_leak.sh`)
- [ ] Rule 2: Not yet relevant (GRPO not running real LoRA freezing yet)
- [ ] Rule 3: Not yet relevant (constrained decoding implementation in W3-W4)

### Documentation status
- [ ] `docs/KAGGLE_NOTES.md` updated with environment-specific learnings
- [ ] `docs/decisions_log.md` updated with W1 design choices
- [ ] W2 plan ready (will draft `docs/W2_DAILY_PLAN.md` based on W1 retrospective)

---

## Action items for W2 prep

(To be completed before W2 Day 1 starts)

- [ ] Draft `docs/W2_DAILY_PLAN.md` based on patterns from W1
- [ ] If advisor hasn't replied, send follow-ups
- [ ] If smoke test failed, document fallback approach taken

---

## Quote of the week

(Something funny, frustrating, or insightful that happened. Keep team morale up.)

> 

---

*Template completed: ___ / ___ / 2026*
