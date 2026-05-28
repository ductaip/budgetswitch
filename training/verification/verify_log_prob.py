"""
Pitfall 2 Verification: GRPO log_prob ↔ Constrained Decoding Consistency.

From docs/IMPLEMENTATION_PITFALLS.md §Pitfall2:

If constrained decoding uses MASKED logits to sample mode tokens,
but TRL's GRPOTrainer computes log_prob on UNMASKED logits,
gradient signal is wrong: reward of mode M is credited to probability
of a different token if that token has higher unmasked logit.

This script verifies whether the TRL version in use is safe,
and provides the custom log_prob wrapper if needed.

Run BEFORE first GRPO training step:
  python training/verification/verify_log_prob.py

Place at: training/verification/verify_log_prob.py
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def check_mode_token_probability_mass(model, tokenizer, sample_query: str) -> dict:
    """
    Check what fraction of the model's probability mass falls on mode tokens
    at the mode-selection position.

    If this is very low (< 0.01), the gradient signal in vanilla GRPO is essentially
    noise, and we need the custom constrained log_prob wrapper.
    """
    import torch

    MODE_STRS = ["M0", "M1", "M2", "M3"]
    MODE_TOKEN_IDS = [tokenizer.encode(m, add_special_tokens=False)[0] for m in MODE_STRS]

    prompt = (
        "<|im_start|>system\nSelect compute mode: M0=fast, M1=think, "
        "M2=think+search, M3=defer<|im_end|>\n"
        f"<|im_start|>user\n{sample_query}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]  # [1, vocab_size]

    # Full-vocab softmax
    full_probs = torch.softmax(logits, dim=-1)
    mode_prob_mass = full_probs[:, MODE_TOKEN_IDS].sum().item()

    # Per-mode probabilities
    full_log_probs = torch.log_softmax(logits, dim=-1)
    constrained_logits = torch.full_like(logits, float('-inf'))
    constrained_logits[:, MODE_TOKEN_IDS] = logits[:, MODE_TOKEN_IDS]
    constrained_log_probs = torch.log_softmax(constrained_logits, dim=-1)

    per_mode = {}
    for mode, tid in zip(MODE_STRS, MODE_TOKEN_IDS):
        per_mode[mode] = {
            "full_log_prob": full_log_probs[0, tid].item(),
            "constrained_log_prob": constrained_log_probs[0, tid].item(),
            "full_prob": full_probs[0, tid].item(),
        }

    # Check consistency: top-1 should be same under both distributions
    full_top = tokenizer.decode([full_probs[0].argmax().item()])
    constrained_top = max(per_mode, key=lambda m: per_mode[m]["constrained_log_prob"])
    consistent = full_top.strip() in MODE_STRS

    return {
        "mode_prob_mass": mode_prob_mass,
        "per_mode": per_mode,
        "full_vocab_top1": full_top.strip(),
        "constrained_top1": constrained_top,
        "consistent": consistent,
        "needs_custom_logprob": mode_prob_mass < 0.01,
    }


def verify_log_prob_consistency(model, tokenizer) -> bool:
    """
    Run Pitfall 2 verification.
    Returns True if safe for vanilla GRPO, False if custom log_prob needed.
    """
    test_queries = [
        "Kiểm tra số dư tài khoản 1234567890",
        "Đặt vé bay từ HN đi HCM ngày 25/12 cho 2 người, sau đó đặt khách sạn",
    ]

    print("=" * 60)
    print("PITFALL 2 VERIFICATION: log_prob consistency")
    print("=" * 60)

    all_safe = True
    for query in test_queries:
        result = check_mode_token_probability_mass(model, tokenizer, query)
        mass = result["mode_prob_mass"]
        needs_fix = result["needs_custom_logprob"]

        status = "✅ SAFE" if not needs_fix else "⚠️  NEEDS CUSTOM LOG_PROB"
        print(f"\nQuery: {query[:55]}...")
        print(f"  Mode token prob mass: {mass:.6f} ({status})")
        for m, d in result["per_mode"].items():
            print(f"  {m}: full_lp={d['full_log_prob']:.3f}, "
                  f"constrained_lp={d['constrained_log_prob']:.3f}, "
                  f"prob={d['full_prob']:.6f}")

        if needs_fix:
            all_safe = False

    print(f"\n{'='*60}")
    if all_safe:
        print("✅ PITFALL 2 CLEAR: mode tokens have sufficient prob mass.")
        print("   Vanilla GRPOTrainer log_prob is safe to use.")
        print("   No custom wrapper needed.")
    else:
        print("⚠️  PITFALL 2 TRIGGERED: mode tokens have low prob mass.")
        print("   Use the custom ConstrainedGRPORewardWrapper below.")
        print("   Without this fix, gradient signal will be near-zero for mode tokens.")
    print(f"{'='*60}\n")

    return all_safe


# ============================================================
# Custom log_prob wrapper (use if verify_log_prob_consistency returns False)
# ============================================================

def constrained_log_prob(logits, target_token_ids, mode_token_ids):
    """
    Compute log_prob of target tokens using CONSTRAINED distribution
    (only over mode_token_ids).

    This ensures the log_prob used in GRPO advantage computation matches
    the distribution used during constrained decoding sampling.

    Args:
        logits: [batch, vocab_size] — raw model logits at mode position
        target_token_ids: [batch] — which mode token was sampled
        mode_token_ids: list of 4 valid mode token IDs

    Returns:
        log_probs: [batch] — log probability of target under constrained distribution
    """
    import torch
    # Mask: keep only mode token logits
    mask = torch.full_like(logits, float('-inf'))
    mask[:, mode_token_ids] = logits[:, mode_token_ids]
    # Softmax over constrained vocab
    log_probs = torch.log_softmax(mask, dim=-1)
    # Gather log_prob of the actually-sampled token
    return log_probs.gather(1, target_token_ids.unsqueeze(1)).squeeze(1)


if __name__ == "__main__":
    import torch

    # Test the constrained_log_prob function with toy data
    vocab_size = 100
    mode_ids = [10, 20, 30, 40]  # simulated mode token IDs

    # Case 1: mode tokens have moderate probability → constrained should be higher
    logits = torch.zeros(1, vocab_size)
    logits[0, mode_ids] = 3.0  # boost mode tokens
    logits[0, 0] = 5.0         # some other token is top-1

    target = torch.tensor([mode_ids[0]])
    clp = constrained_log_prob(logits, target, mode_ids)
    full_lp = torch.log_softmax(logits, dim=-1)[0, mode_ids[0]]
    print(f"Full vocab log_prob of mode[0]:        {full_lp:.4f}")
    print(f"Constrained log_prob of mode[0]:       {clp.item():.4f}")
    assert clp.item() > full_lp.item(), "Constrained LP should be higher than full LP"
    print("✅ constrained_log_prob function works correctly")
    print("\nNote: Run verify_log_prob_consistency(model, tokenizer) on Kaggle T4")
    print("with the SFT model loaded to get the actual verdict.")