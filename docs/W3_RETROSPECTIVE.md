# W3 Retrospective — Cold-Start SFT

**Period:** Week 3 of 24
**Status:** ✅ COMPLETE — SFT successful, 90% easy-step success rate
**Next:** W4 — Re-run pilot + constrained decoding + GRPO setup

---

## 1. Kết quả SFT

| Metric | Target | Actual | Status |
|---|---|---|---|
| Train loss (epoch 3) | < 1.0 | **0.4176** | ✅ Excellent |
| Runtime | ~3-5h | **11.9 min** | ✅ Unsloth 2x speedup |
| Post-SFT M0 easy success | ≥ 40% | **90% (18/20)** | ✅ Far exceeds target |
| Adapter size | ~32 MB | 131.2 MB | ✅ Expected (rank 16 + MLP layers) |
| Val loss direction | Decreasing | Decreasing | ✅ No overfitting |

**Verdict: 🟢 SFT successful — proceed to W4**

---

## 2. Phân tích kết quả

### 2.1 Tại sao 90% > 40% target?

Target 40% được set conservative cho base model với minimal SFT. Thực tế:
- 239 examples với 3 epochs = 717 gradient steps (small but high quality)
- Unsloth padding-free training + double buffering → effective learning
- Gold trajectories từ Groq Llama-3.3-70B chất lượng cao
- Task format đơn giản (JSON tool call) → model học nhanh

**Implication cho paper:** Post-SFT baseline sẽ rất strong. Gap M0 vs M1/M2 trên hard steps sau SFT sẽ là chứng minh rõ ràng nhất cho overthink/underthink thesis.

### 2.2 Adapter size 131.2 MB (vs estimate 32 MB)

W3 notebook thêm `gate_proj`, `up_proj`, `down_proj` (MLP layers) vào target_modules so với estimate ban đầu (chỉ q/k/v/o). MLP layers quan trọng cho tool-calling format learning. Size lớn hơn nhưng vẫn reasonable cho tool-use LoRA.

Mode-routing LoRA (W4) sẽ dùng rank 8, chỉ q/k/v/o → ~16 MB như thiết kế.

### 2.3 Mode distribution random đúng

```
{'M0': 70, 'M2': 47, 'M1': 67, 'M3': 55}  (~25% mỗi mode)
```

**Critical Rule 1 maintained:** SFT học syntax (cú pháp mode token + JSON format), không học policy (khi nào dùng mode nào). GRPO mới dạy policy.

### 2.4 Runtime 11.9 phút

Estimate 3-5h dựa trên vanilla SFT. Unsloth v2026.5.8 với:
- Padding-free auto-enabled
- Double buffering (parallel H2D + compute)
- Smart gradient offload

→ Thực tế gần 20x faster than vanilla. **Implication for GRPO:** GRPO sẽ cũng benefit tương tự. Estimate 10-12h GRPO có thể rút xuống 4-6h.

---

## 3. Critical Rules status

| Rule | Status | Notes |
|---|---|---|
| Rule 1: No oracle_mode in reward | ✅ ENFORCED | SFT data dùng random mode tokens, verified |
| Rule 2: Tool-use LoRA frozen during GRPO | ⏳ W4 | Will freeze before adding mode-routing LoRA |
| Rule 3: Mode tokens = regular text + constrained decoding | ⏳ W4 | Implement constrained decode in W4 |

---

## 4. Environment notes

**Unsloth version:** 2026.5.8 (newest — faster than 2026.5.5 dùng trong W1-W2)
**Training hardware:** 1 GPU T4 (Unsloth uses 1 GPU for SFT despite 2 available — correct behavior)
**bfloat16:** FALSE (T4 không support bf16 — fp16 used instead, confirmed working)

**Warning về max_new_tokens + max_length:** Harmless, max_new_tokens wins. Document để không gây confuse.

---

## 5. Files sau W3

```
budgetswitch/
├── models/
│   └── tool_use_lora/        ← NEW: SFT adapter (131.2 MB, rank 16)
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── tokenizer files
└── notebooks/
    └── W3_cold_start_sft.ipynb  ← NEW: 11.9 min run, loss 0.4176
```

---

## 6. W4 Plan

**3 mục tiêu W4:**

1. **Re-run pilot với SFT model** (50 steps) → verify H1-H2-H3 → GREEN expected với 90% easy baseline
2. **Constrained decoding** cho mode tokens (Critical Rule 3)
3. **Mode-routing LoRA setup** (rank 8) + GRPO preparation (β-sensitivity sweep)

**Expected W4 outcome:** Pilot GREEN → commit to full GRPO training (W5-W8)

---

*W3 Retrospective — hoàn thành trong 11.9 phút training*