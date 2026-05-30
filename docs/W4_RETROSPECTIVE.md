# W4 Retrospective — Pilot Rerun + Constrained Decode + Mode-Routing Setup

**Period:** Week 4 of 24
**Status:** ✅ COMPLETE — infrastructure ready, 3 issues identified and addressed
**Next:** W5 — Generate hard trajectories + Full GRPO training

---

## 1. Kết quả W4

| Component | Result | Status |
|---|---|---|
| Pilot H1 (overthink-easy) | M0=88%, gap M1-M0=+6% | ✅ PASS |
| Pilot H2 (underthink-hard) | Only 1 hard step → inconclusive | ⚠️ Need more hard data |
| Mode token IDs | M0=15, M1=16, M2=17, M3=18 (single tokens) | ✅ |
| Constrained decoding (Rule 3) | All outputs in {M0,M1,M2,M3} | ✅ |
| Two-LoRA (Rule 2) | tool-use FROZEN, mode-routing TRAINABLE (14.9M) | ✅ |
| Gradient isolation (Pitfall 1) | 504 tool-use LoRA tensors unchanged | ✅ CLEAR |
| Pitfall 2 (log_prob) | mass=0.0027 (low) → ModeTokenBooster installed | ✅ FIXED |
| β sweep (50 steps each) | All 4 β values BALANCED | ✅ |

---

## 2. Issues phát hiện — cần fix trước W5 GRPO

### Issue 1 (CRITICAL): M2/M3 = 0% sau tất cả β values

**Observation:** β sweep 50 steps shows M0/M1 only — M2 và M3 không bao giờ được chọn.

```
β=0.01: M0=50%, M1=50%, M2=0%, M3=0%
β=0.05: M0=45%, M1=55%, M2=0%, M3=0%
β=0.10: M0=40%, M1=60%, M2=0%, M3=0%
β=0.20: M0=30%, M1=70%, M2=0%, M3=0%
```

**Root cause:** 2 vấn đề kết hợp:
1. 239 training examples chủ yếu là easy/medium → M0/M1 đủ succeed → M2/M3 không có reward signal
2. Cost penalty (M2=5×, M3=15×) discourage exploration ngay cả β=0.01

**Tại sao critical cho paper:** C1 contribution là "4-mode action space." Nếu full GRPO chỉ dùng M0/M1, paper chỉ có 2-mode system thực tế — reviewer sẽ attack C1.

**Fix cho W5:**
- Thêm entropy bonus vào reward để encourage M2/M3 exploration
- Generate thêm 50+ hard trajectories (3+ step compositions) trước GRPO
- Monitor mode distribution mỗi 100 GRPO steps

### Issue 2: Data imbalance (17 easy + 17 medium + 1 hard trong pilot)

**Root cause:** Groq Llama-3.3-70B tự nhiên generate tasks dễ hơn. Cần prompt engineering rõ ràng hơn để get hard tasks.

**Fix:** Thêm explicit prompt instruction: "Tạo tasks KHÓ: yêu cầu 3+ tool calls, params mơ hồ, cần reasoning để xác định đúng tool sequence."

### Issue 3: Adapter size 179.7 MB (expected ~16 MB)

**Root cause:** `model.save_pretrained()` có thể save cả 2 adapters. Cần `model.save_pretrained(path, adapter_name="mode_routing")` explicit.

**Fix trong W5 notebook:** Specify `adapter_name="mode_routing"` khi save.

---

## 3. β Choice cho full GRPO

**Recommendation: β=0.05** (không phải β=0.01 như auto-recommend)

**Reasoning:**
- β=0.01: M0=M1=50%, quá ít cost pressure. Sau 4500 steps, model sẽ vẫn tend toward M0/M1 vì reward từ correct answer dominates cost signal.
- β=0.05: Đủ cost pressure để khi model thấy M0 fail trên hard step, việc dùng M1/M2 có net positive reward.
- β=0.10 và 0.20: Quá aggressive — có thể push model về M0-only để minimize cost.

**Điều chỉnh thêm:** Thêm entropy bonus `γ_entropy × H(mode_distribution)` để explicitly encourage mode diversity trong early GRPO phase (first 500 steps), sau đó anneal xuống 0.

---

## 4. Pitfall 2 — ModeTokenBooster analysis

**Query 1** (simple): mass=0.0027 — TRIGGERED
**Query 2** (complex): mass=0.1032 — SAFE

Pattern: Simple Vietnamese queries → model không expect mode token → low mass. Complex queries → model đã quen với format → higher mass.

**ModeTokenBooster** (installed trong Cell 15) giải quyết bằng cách boost mode token logits +10 trước sampling. Điều này đảm bảo:
- Sampling dùng masked distribution (chỉ 4 mode tokens)
- Log_prob trong GRPO computation sẽ consistent với sampling distribution
- Gradient signal correct

**Cần verify trong W5:** Confirm ModeTokenBooster được apply nhất quán trong mọi GRPO rollout, không chỉ inference.

---

## 5. Decisions log W4

| Decision | Chosen | Rationale |
|---|---|---|
| β cho full GRPO | 0.05 | More cost pressure than 0.01 to differentiate modes; less aggressive than 0.10 |
| Entropy bonus | γ_entropy=0.1, anneal over 500 steps | Encourage M2/M3 exploration in early training |
| Hard tasks | +50 before GRPO | 3+ step compositions to provide M2/M3 reward signal |
| ModeTokenBooster | Keep installed | Fixes Pitfall 2 low prob mass issue |

---

## 6. W5 Plan

### W5a (trước GRPO): Generate 50 hard trajectories
```bash
python data/generate_drafts.py --domain travel --count 15 \
  --hard-only --min-steps 3 --out data/vitool/drafts_hard_travel.jsonl
# Lặp cho banking (10), ecommerce (10), public_services (10), info_retrieval (5)
```

Hard trajectory requirements:
- Minimum 3 tool calls
- At least 1 tool that requires previous step's output as input
- Ambiguous params requiring reasoning (e.g., relative dates, implicit cities)

### W5b: Full GRPO training
- β=0.05, entropy_bonus=0.1 (anneal to 0 at step 500)
- G=4 group size
- 4500 steps (~3 epochs × 1500 trajectories)
- Monitor mode distribution mỗi 100 steps
- Target mode dist: M0≥25%, M1≥25%, M2≥15%, M3≥5%

---

## 7. Progress overview

```
W1 ✅  GRPO smoke test (VRAM 2.61GB, 5 steps)
W2 ✅  200 trajectories, pilot infrastructure
W3 ✅  Cold-start SFT (loss 0.4176, 90% easy success)
W4 ✅  Two-LoRA + constrained decode + β=0.05 chosen
        Issues: M2/M3=0%, data imbalance, adapter size
W5 ⏳  Hard data generation + Full GRPO training
W6-W8 ○  GRPO monitoring + ablations setup
W9-W12 ○  Full experiments (8 baselines)
W13-W16 ○  Paper writing §1-§11
W17-W20 ○  Revision + advisor review
W21-W24 ○  Submission RIVF/SOICT 2026
```

---

*W4 Retrospective — Key finding: need hard trajectories to activate M2/M3 modes*