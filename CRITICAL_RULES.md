# CRITICAL_RULES.md — BudgetSwitch Implementation Rules

**These rules are NON-NEGOTIABLE. Violating them breaks methodology claims in the paper.**

Paste rules 1-3 as a comment block at the top of:
- `training/reward.py`
- `training/grpo_trainer.py`
- `verifiers/universal/*.py`

---

## Rule 1: Reward Function MUST NOT Reference oracle_mode

**Why:** The paper claims outcome-verifiable GRPO without oracle supervision. If `oracle_mode`, `expected_mode`, `correct_mode`, or any per-step mode label appears in the reward function, the methodology becomes SFT-disguised-as-RL. An RL-literate reviewer will catch this immediately and reject the novelty claim.

**Allowed in reward:**
- `step.tool_name == ground_truth['expected_tool']` ✅ (this is outcome verification)
- `validate_jsonschema(step.params, schema)` ✅
- `mode_cost[mode]` ✅ (cost lookup, not mode supervision)
- `verify_final_outcome(trajectory)` ✅

**FORBIDDEN in reward:**
- `oracle_mode` ❌
- `expected_mode` ❌
- `correct_mode` ❌
- `target_mode` ❌
- `mode == ground_truth['mode']` ❌
- Any comparison between actual mode and a "correct" mode label ❌

**Quick grep check (run before every commit):**
```bash
grep -rn "oracle_mode\|expected_mode\|correct_mode\|target_mode" training/ verifiers/
# Expected output: nothing
```

---

## Rule 2: Selective LoRA Freezing During GRPO

**Why:** Two-LoRA design (tool-use + mode-routing). During GRPO, only mode-routing LoRA trains. Tool-use LoRA must be frozen to preserve function-calling capability.

**Correct setup:**
```python
# Load tool-use LoRA as frozen
model = PeftModel.from_pretrained(base_model, "./tool_use_lora", is_trainable=False)

# Add NEW mode-routing LoRA (trainable)
model.add_adapter("mode_routing", LoraConfig(...))
model.set_adapter("mode_routing")  # ← CRITICAL: explicit activation

# Verify only mode_routing has trainable params
trainable = [n for n, p in model.named_parameters() if p.requires_grad]
assert all("mode_routing" in n for n in trainable), "Tool-use LoRA leaked!"
```

**Verification required after first GRPO step (see `training/verification/verify_lora_frozen.py`).**

---

## Rule 3: Mode Tokens = Regular Text + Constrained Decoding

**Why:** Modifying tokenizer breaks Unsloth compatibility. Use regular text tokens "M0", "M1", "M2", "M3" with constrained decoding (mask logits at mode position).

**Correct setup:**
```python
MODE_STRS = ["M0", "M1", "M2", "M3"]
MODE_TOKEN_IDS = []
for m in MODE_STRS:
    ids = tokenizer.encode(m, add_special_tokens=False)
    assert len(ids) == 1, f"'{m}' tokenized to {len(ids)} tokens, expected 1"
    MODE_TOKEN_IDS.append(ids[0])
```

**FORBIDDEN:**
- `tokenizer.add_special_tokens(...)` for mode tokens ❌
- `tokenizer.add_tokens(...)` for mode tokens ❌
- Modifying `tokenizer.vocab` ❌

---

## Implementation Sanity Checklist (run before W11 full GRPO)

```
[ ] grep -rn "oracle_mode\|expected_mode\|correct_mode\|target_mode" training/ → empty
[ ] training/verification/verify_lora_frozen.py → passes
[ ] training/verification/verify_log_prob.py → mode prob mass > 0.01
[ ] training/verification/verify_no_oracle_leak.py → passes
[ ] β-sensitivity sweep complete (4 short runs at β ∈ {0.01, 0.05, 0.10, 0.20})
[ ] Mode distribution monitoring callback registered
[ ] Pre-computed tool embeddings ready (~10ms retrieval verified)
[ ] Cold-start SFT model checkpoint exists
[ ] Validation set held out (10% of ViTool-2.0K)
[ ] Reward function unit tested with synthetic trajectories
```

If any item fails → STOP and fix before committing GPU hours to full training.

---

*Last updated: W1 Day 1 of BudgetSwitch project.*
*Reference: README_BudgetSwitch_v9.2_FINAL.md §17, IMPLEMENTATION_PITFALLS.md*
