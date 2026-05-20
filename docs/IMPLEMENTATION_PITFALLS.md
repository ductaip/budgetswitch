# IMPLEMENTATION_PITFALLS.md

**Companion file cho `README_BudgetSwitch_v9.2_FINAL.md`**

File này chứa 2 implementation pitfalls cần verify ở W3-W4 (GRPO setup phase). Đây KHÔNG phải methodology issues — chỉ là technical details có thể trip team nếu không check trước.

**Khi nào cần đọc file này:** Trước khi commit GRPO trial code (W4 trong roadmap).

---

## Pitfall 1 — PEFT Gradient Isolation cần Verify Thủ Công

### Vấn đề

Trong README §7.4, code:
```python
model = PeftModel.from_pretrained(base_model, "./tool_use_lora", is_trainable=False)
```

Dòng `is_trainable=False` **không đảm bảo hoàn toàn** gradient isolation trong GRPO backprop. Có khả năng (low probability nhưng non-zero) tool-use LoRA parameters vẫn nhận gradient updates từ GRPO step → catastrophic forgetting tool-calling capability.

### Verification Code (paste vào cuối W9)

```python
# === Verify tool-use LoRA frozen correctly ===
# Run this AFTER first trainer.train() call, BEFORE full training

import copy
import torch

# Snapshot tool-use LoRA params trước GRPO step
params_before = {
    k: v.clone().detach() 
    for k, v in model.named_parameters() 
    if "tool_use_lora" in k or "default" in k  # adjust based on adapter naming
}

# Run a few GRPO steps
trainer.train(max_steps=10)

# Snapshot after
params_after = {
    k: v.detach()
    for k, v in model.named_parameters() 
    if "tool_use_lora" in k or "default" in k
}

# Verify identical
all_frozen = True
for k in params_before:
    if not torch.allclose(params_before[k], params_after[k], atol=1e-7):
        print(f"❌ LEAK DETECTED: {k} changed during GRPO!")
        all_frozen = False

if all_frozen:
    print("✅ Tool-use LoRA frozen correctly.")
else:
    print("⚠️  Need to apply explicit fix below.")
```

### Fix nếu Assert Fail

```python
# Explicit gradient disable cho tool-use LoRA parameters
for name, param in model.named_parameters():
    if "tool_use_lora" in name or any(suffix in name for suffix in ["default.lora_A", "default.lora_B"]):
        param.requires_grad_(False)

# Verify only mode-routing LoRA has trainable params
trainable_params = [n for n, p in model.named_parameters() if p.requires_grad]
print(f"Trainable params: {len(trainable_params)}")
print(f"First few: {trainable_params[:5]}")
# Should only show mode_routing-related parameters
```

### Severity & Time Cost

- **Severity:** HIGH if leak occurs (catastrophic forgetting)
- **Time to verify:** ~10 minutes
- **Time to fix if needed:** ~30 minutes

→ **Mandatory verification** trước W11 full GRPO training.

---

## Pitfall 2 — GRPO log_prob và Constrained Decoding Phải Nhất Quán

### Vấn đề

Trong README §7.4, mode tokens dùng **constrained decoding** (mask logits, argmax over 4 mode token IDs). Nhưng GRPO trong TRL library tính `log_prob` của sampled tokens bằng **full forward pass trên unmasked logits**.

Nếu hai logits distributions khác nhau:
- Sample mode bằng masked logits (chỉ 4 options)
- Compute log_prob bằng unmasked logits (full vocab)

→ Gradient signal **sai**: reward của mode được sample sẽ được credit cho probability của mode khác (nếu mode khác có higher unmasked logit).

### Concrete Example

Giả sử tại mode position:
- Unmasked logits: token "M0"=2.0, token "M1"=1.5, token "M3"=1.0, token "the"=3.0 (highest)
- Masked logits (mode-only): "M0"=2.0, "M1"=1.5, "M3"=1.0
- Sample: argmax masked → chọn "M0"
- TRL compute log_prob: softmax over **full vocab** → P("M0") rất nhỏ vì "the" dominate
- Reward attributed to "M0" → gradient updates cho P("the") thay vì P("M0")

→ **Training broken silently, không có error message.**

### Fix Approaches (chọn 1)

**Approach A — Manual log_prob computation (recommended)**

```python
# Custom log_prob trên masked vocab
def compute_constrained_log_prob(logits, sampled_token_ids, mode_token_ids):
    """Compute log_prob over only the constrained vocabulary."""
    # logits: [batch, vocab_size]
    # mode_token_ids: list of 4 valid mode token IDs
    
    # Mask: chỉ keep mode tokens
    mask = torch.full_like(logits, float('-inf'))
    mask[:, mode_token_ids] = logits[:, mode_token_ids]
    
    # Softmax over constrained vocab only
    log_probs = torch.log_softmax(mask, dim=-1)
    
    # Gather log_prob của sampled tokens
    return log_probs.gather(1, sampled_token_ids.unsqueeze(1)).squeeze(1)
```

**Approach B — Check TRL version hỗ trợ constrained vocab**

```python
# Newer TRL versions (>=0.18) có thể có support
from trl import GRPOConfig

config = GRPOConfig(
    # ... existing params ...
    constrained_vocab_ids=MODE_TOKEN_IDS,  # nếu version hỗ trợ
)

# Check TRL docs:
# pip show trl
# https://github.com/huggingface/trl/releases
```

**Approach C — Subclass GRPOTrainer (most robust)**

```python
from trl import GRPOTrainer

class ConstrainedGRPOTrainer(GRPOTrainer):
    def __init__(self, *args, mode_token_ids=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode_token_ids = mode_token_ids
    
    def _get_per_token_logps(self, model, input_ids, attention_mask, ...):
        """Override to use masked logits for mode positions only."""
        # Standard forward pass
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [B, T, V]
        
        # Detect mode positions (where output should be M0/M1/M2/M3)
        mode_position_mask = self._detect_mode_positions(input_ids)  # [B, T]
        
        # Mask logits only at mode positions
        masked_logits = logits.clone()
        full_mask = torch.full_like(logits, float('-inf'))
        full_mask[..., self.mode_token_ids] = 0.0
        masked_logits = torch.where(
            mode_position_mask.unsqueeze(-1),
            logits + full_mask,  # Apply mask only at mode positions
            logits  # Keep full logits elsewhere
        )
        
        log_probs = torch.log_softmax(masked_logits, dim=-1)
        return log_probs
```

### Verification Code (paste vào W4)

```python
# Sanity check: log_prob của mode tokens phải hợp lý
def verify_log_prob_consistency(model, tokenizer, sample_query):
    """Check log_prob distribution at mode position."""
    prompt = build_mode_prompt(sample_query, history=[])
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]  # Last position
    
    # Full vocab log_probs
    full_log_probs = torch.log_softmax(logits, dim=-1)
    
    # Constrained log_probs
    masked = torch.full_like(logits, float('-inf'))
    masked[:, MODE_TOKEN_IDS] = logits[:, MODE_TOKEN_IDS]
    constrained_log_probs = torch.log_softmax(masked, dim=-1)
    
    # Mode tokens chiếm bao nhiêu probability mass trong full vocab?
    mode_prob_mass = full_log_probs[:, MODE_TOKEN_IDS].exp().sum().item()
    
    print(f"Mode tokens probability mass (full vocab): {mode_prob_mass:.6f}")
    print(f"If this is < 0.01, gradient signal sẽ rất noisy với approach A.")
    print(f"Recommendation: Use Approach C (custom GRPOTrainer subclass).")
    
    # Compare log_probs
    for mode_id, mode_str in zip(MODE_TOKEN_IDS, ["M0", "M1", "M2", "M3"]):
        full_lp = full_log_probs[0, mode_id].item()
        const_lp = constrained_log_probs[0, mode_id].item()
        print(f"{mode_str}: full_log_prob={full_lp:.4f}, constrained_log_prob={const_lp:.4f}")
```

### Severity & Time Cost

- **Severity:** HIGH (training silently broken)
- **Time to implement Approach A:** ~30 minutes
- **Time to implement Approach C:** ~2-3 hours
- **Time to verify:** ~15 minutes

→ **Mandatory** trước first GRPO trial run (W4).

---

## Implementation Sanity Checklist (Pre-W11)

Trước khi commit full GRPO training run trong W11, verify checklist:

```
[ ] Pitfall 1: Tool-use LoRA verified frozen (assert passed)
[ ] Pitfall 2: Constrained log_prob consistency verified
[ ] §17 Rule 1: Reward function reviewed, NO oracle_mode reference
[ ] §17 Rule 2: Only mode_routing adapter active for training
[ ] §17 Rule 3: Mode tokens working with constrained decoding
[ ] β-sensitivity sweep complete (W10 Day 1-2)
[ ] Mode distribution monitoring callback registered
[ ] Pre-computed tool embeddings loaded (verified: ~10ms retrieval per query)
[ ] Cold-start SFT model checkpoint exists
[ ] Validation set held out (10% of ViTool-2.0K)
[ ] Reward function unit tested với synthetic trajectories
```

Nếu tất cả checked → green light cho W11 full training.

---

## Common Mistakes Cần Tránh

### Mistake 1: Reward function reference oracle dù không nhận thức

```python
# ❌ BAD (subtle leak):
def trajectory_reward(traj, ground_truth):
    # ground_truth có thể chứa expected_mode trong some implementations
    correct_mode_used = traj.mode == ground_truth.get('expected_mode', None)
    reward = 1.0 if correct_mode_used else 0.5
    return reward

# ✅ GOOD:
def trajectory_reward(traj, ground_truth):
    # ground_truth chỉ chứa expected_tool, expected_params, expected_outcome
    # KHÔNG bao gồm expected_mode
    success = verify_step_outcome(traj.step, ground_truth)
    cost = mode_cost[traj.mode]
    return success - β * cost
```

### Mistake 2: Forget to set adapter trước khi train

```python
# ❌ BAD:
model.add_adapter("mode_routing", lora_config)
trainer.train()  # Có thể train default adapter (tool-use)!

# ✅ GOOD:
model.add_adapter("mode_routing", lora_config)
model.set_adapter("mode_routing")  # Explicit
print(f"Active adapter: {model.active_adapter}")  # Verify
trainer.train()
```

### Mistake 3: Mode token IDs sai do tokenizer behavior

```python
# ❌ BAD:
mode_ids = [tokenizer.encode(m)[0] for m in ["M0", "M1", "M2", "M3"]]
# Có thể include BOS token at index 0!

# ✅ GOOD:
mode_ids = [tokenizer.encode(m, add_special_tokens=False)[0] 
            for m in ["M0", "M1", "M2", "M3"]]

# Hoặc safer:
def get_mode_token_id(tokenizer, mode_str):
    ids = tokenizer.encode(mode_str, add_special_tokens=False)
    assert len(ids) == 1, f"'{mode_str}' tokenized to {len(ids)} tokens, expected 1"
    return ids[0]

MODE_TOKEN_IDS = [get_mode_token_id(tokenizer, m) for m in ["M0", "M1", "M2", "M3"]]
print(f"Mode token IDs: {MODE_TOKEN_IDS}")
# Verify: should be 4 distinct integers
```

### Mistake 4: GRPO group size collision với T4 memory

```python
# ❌ BAD: G=8 trên T4 sẽ OOM
config = GRPOConfig(num_generations=8, ...)

# ✅ GOOD: Start với G=4, verify memory, scale up nếu OK
config = GRPOConfig(
    num_generations=4,
    per_device_train_batch_size=2,  # Effective batch = 4 × 2 = 8 sequences
    gradient_accumulation_steps=4,  # Total effective = 32
)
```

---

## Reference Implementation Structure

Khuyến nghị folder structure cho `/training/`:

```
/training/
  ├── reward.py              # trajectory_reward() — REVIEW HEAVILY (Rule 1)
  ├── grpo_trainer.py        # ConstrainedGRPOTrainer subclass (Pitfall 2)
  ├── lora_setup.py          # Two-LoRA loading + freezing (Pitfall 1)
  ├── mode_token_utils.py    # Constrained decoding utils (Rule 3)
  ├── beta_sweep.py          # W10 β-sensitivity sweep
  ├── monitoring.py          # Mode distribution callbacks
  ├── verification/
  │   ├── verify_lora_frozen.py     # Pitfall 1 check
  │   ├── verify_log_prob.py        # Pitfall 2 check
  │   └── verify_no_oracle_leak.py  # Rule 1 check (grep-based)
  └── tests/
      ├── test_reward.py             # Unit tests cho reward
      ├── test_constrained_decode.py # Unit tests cho mode generation
      └── test_grpo_step.py          # End-to-end mini test
```

`verify_no_oracle_leak.py` đơn giản:

```python
"""Grep through codebase for oracle_mode references in reward-related files."""
import os
import re

REWARD_FILES = ["training/reward.py", "training/grpo_trainer.py"]
FORBIDDEN_PATTERNS = [
    r"oracle_mode",
    r"expected_mode",
    r"correct_mode",
    r"target_mode",
]

violations = []
for file_path in REWARD_FILES:
    with open(file_path) as f:
        content = f.read()
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, content):
                violations.append(f"{file_path}: matches '{pattern}'")

if violations:
    print("❌ ORACLE LEAK DETECTED:")
    for v in violations:
        print(f"  {v}")
    exit(1)
else:
    print("✅ No oracle_mode references in reward files.")
```

→ Add to pre-commit hook để auto-verify mỗi commit.

---

*v9.2 Companion file — created 25/04/2026.*
*Reference: `README_BudgetSwitch_v9.2_FINAL.md` cho full proposal context.*
*Stop iterating. Start building.*
