# Week 1 Retrospective (Filled — Smoke Test Edition)

**Period:** W1
**Status:** Day 6 milestone (Kaggle GRPO smoke test) PASSED ✅

This is a partial retrospective focused on the smoke test result. Fill in the
rest (reading, advisor outreach) as those items complete.

---

## ✅ Day 6 Milestone: Kaggle GRPO Smoke Test — PASSED

| Criterion | Target | Actual | Status |
|---|---|---|---|
| 5 GRPO steps complete | Yes | Yes | ✅ |
| No exceptions | Yes | Yes | ✅ |
| Peak VRAM | ≤ 14 GB | 2.61 GB | ✅ (huge headroom) |
| Pipeline end-to-end | Working | Working | ✅ |

**Environment:** Kaggle T4 ×2, Unsloth 2026.5.5, transformers 5.5.0, trl 0.24.0,
torch 2.12.0+cu130, bitsandbytes 0.49.2.

---

## What was harder than expected

1. **Kaggle CUDA 13 environment broke 4-bit loading.** The default Kaggle image
   ships CUDA 13, but bitsandbytes expects `libnvJitLink.so.13` on the linker path.
   Required an OS-level symlink fix via `%%bash` + `ldconfig` (env vars / ctypes
   did NOT work because bitsandbytes uses `dlopen()`). **Fully documented in
   `docs/KAGGLE_NOTES.md` so we never debug this again.**

2. **vLLM unavailable on Kaggle CUDA 13.** Had to set `fast_inference=False`.
   GRPO rollouts are slower without vLLM but it works. Acceptable for our scale.

3. **Dependency version pinning.** unsloth-zoo requires transformers≤5.5.0,
   trl≤0.24.0, datasets<4.4.0. Upgrading any of these breaks the stack.
   Solution: only upgrade peft + accelerate, leave the rest pinned.

4. **adamw_8bit also needs bitsandbytes CUDA.** Switched to adamw_torch.

---

## What was easier than expected

1. **VRAM is a non-issue.** Peak was 2.61 GB out of 14.5 GB. We have enormous
   headroom — can increase batch size, group size, or sequence length freely
   in W11 without OOM concerns.

2. **Once env fixes were in place, GRPO "just worked."** TRL's GRPOTrainer
   integrated cleanly with the dummy reward function.

---

## Key correction: the "150 GPU-hours" scare

The smoke test's final cell extrapolated 150 GPU-hours for full training.
**This is a measurement artifact, not reality.** The 15s/step included
cold-start overhead, and the `max_completion 128→2048 = 16×` assumption is wrong
(generation scales with actual tokens produced, not the cap). Realistic estimate
is ~10-15 GPU-hours, consistent with Vi-S1K and GreenMind on the same hardware.
Cell 15 has been corrected. We will measure steady-state step time on the first
50 real GRPO steps in W10-W11 and re-extrapolate properly.

---

## Outstanding W1 items (to complete by end of week)

- [ ] Reading: 6 Tier-1 papers + notes in `docs/paper_notes/`
- [ ] Advisor outreach: 2-3 emails sent
- [ ] M1 think mode implementation (Day 4)
- [ ] Sunday team sync

---

## Risks identified this week

| Risk | Severity | Mitigation |
|---|---|---|
| Kaggle env may change/break between sessions | Medium | KAGGLE_NOTES.md documents exact fixes; re-run pre-flight checklist each session |
| Without vLLM, GRPO rollouts slower | Low | Still completes; measured 15s/step incl. cold start; will optimize if needed |
| Kaggle CUDA version may bump again | Low | Symlink fix is version-agnostic (finds any libnvJitLink.so*) |

---

## Velocity check

**Day 6 milestone hit on schedule.** Pipeline verification — the single most
important W1 deliverable — is done. We have confidence the training infrastructure
works before investing in W2 data construction.

**On track for W2?** Yes, provided data construction (the hard part) starts promptly.

---

## Update to project state

### Critical Rules status
- [x] Rule 1: Smoke test used dummy_reward (random) — no oracle_mode anywhere ✅
- [ ] Rule 2: Not yet relevant (single LoRA in smoke test; two-LoRA freezing in W9)
- [ ] Rule 3: Not yet relevant (constrained decoding in W3-W4)

### Documentation status
- [x] `docs/KAGGLE_NOTES.md` created with all env fixes ✅
- [x] Smoke test notebook saved to `notebooks/W1_grpo_smoke_test.ipynb` ✅
- [ ] `docs/W2_DAILY_PLAN.md` — drafted (see below)

---

## Quote of the week

> "libnvJitLink.so.13 — the four hours I'll never get back, but at least it's
> documented now so nobody else loses them."

---

*Updated: W1 Day 6, after smoke test success.*
