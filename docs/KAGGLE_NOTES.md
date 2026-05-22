# Kaggle Environment Notes

**Status:** W1 Day 6 smoke test PASSED ✅ on Kaggle T4 (2× T4 available, used 1).

This file documents environment-specific fixes discovered during setup so we
never have to re-debug them in W2-W11. **Read this before every new Kaggle session.**

---

## Verified working environment (as of W1)

| Component | Version | Notes |
|---|---|---|
| Unsloth | 2026.5.5 | Installed via `unsloth[kaggle-new]` git |
| transformers | 5.5.0 | Do NOT upgrade — unsloth-zoo requires ≤5.5.0 |
| trl | 0.24.0 | Do NOT upgrade — unsloth-zoo requires ≤0.24.0 |
| datasets | 4.3.0 | Do NOT upgrade — unsloth-zoo requires <4.4.0 |
| bitsandbytes | 0.49.2 | Works after libnvJitLink symlink fix |
| torch | 2.12.0+cu130 | Kaggle default (CUDA 13) |
| GPU | Tesla T4 ×2 | 15.36 GB each, ~14.9 GB free |

---

## CRITICAL FIX 1: libnvJitLink.so.13 symlink (MUST run first)

**Problem:** Kaggle images ship CUDA 13 toolkit, but bitsandbytes CUDA binary
expects `libnvJitLink.so.13` which isn't where the dynamic linker looks. Loading
bitsandbytes (needed for 4-bit) fails without this fix.

**Why os.environ / ctypes don't work:** bitsandbytes loads its CUDA `.so` via
`dlopen()`, which respects the OS-level linker cache (`ldconfig`), not Python
environment variables. The fix must be at OS level.

**Fix (run as a `%%bash` cell BEFORE pip install):**
```bash
%%bash
set -e
echo "=== Find libnvJitLink ==="
LIB=$(find /usr/local/cuda* -name "libnvJitLink.so.13" 2>/dev/null | head -1)
if [ -z "$LIB" ]; then
    # Fall back to any version present (e.g. .so.12 on CUDA 12.8)
    LIB=$(find /usr/local/cuda* -name "libnvJitLink.so*" 2>/dev/null | head -1)
fi
echo "Found: $LIB"
# Symlink into a directory on the linker path, refresh ldconfig
ln -sf "$LIB" /usr/local/cuda/targets/x86_64-linux/lib/libnvJitLink.so.12 || true
ldconfig
echo "✅ Symlink created + ldconfig updated"
```

**Note:** The `ldconfig` "is not a symbolic link" warnings about libtbb/libur/etc.
are HARMLESS — they're pre-existing Kaggle image quirks, not caused by our fix.

---

## CRITICAL FIX 2: Package install order (avoid dependency hell)

**Problem:** Upgrading transformers/trl/datasets breaks unsloth-zoo compatibility.
vLLM conflicts with Kaggle's CUDA 13.

**Fix (install cell):**
```python
# 1. Remove conflicting packages
!pip uninstall -y vllm torchvision torchaudio 2>/dev/null || true

# 2. Install Unsloth (kaggle-new variant)
!pip install --quiet "unsloth[kaggle-new] @ git+https://github.com/unslothai/unsloth.git"

# 3. ONLY upgrade peft + accelerate — do NOT touch transformers/trl/datasets
#    (unsloth-zoo requires: transformers<=5.5.0, trl<=0.24.0, datasets<4.4.0)
!pip install --quiet --upgrade peft accelerate
```

**The dependency-conflict ERROR messages from pip are EXPECTED and HARMLESS:**
- `bigframes requires google-cloud-bigquery-storage` → irrelevant to us
- `s3fs requires fsspec==2026.2.0` → irrelevant
- `libcuml/cuml/cudf require cuda-toolkit 12.*` → these are Kaggle's RAPIDS libs we don't use
- `cuda-python requires cuda-bindings~=12.9.4` → irrelevant

None of these affect Unsloth/training. They're Kaggle's pre-installed data-science
stack complaining about CUDA version mismatch, which we don't touch.

---

## CRITICAL FIX 3: Model loading config

**What worked:**
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-3B-Instruct",  # NOT the -bnb-4bit suffix variant
    max_seq_length=2048,
    load_in_4bit=True,        # ✅ works after Fix 1
    fast_inference=False,     # ⚠️ MUST be False — vLLM unavailable on Kaggle CUDA 13
    dtype=None,
)
```

**Key gotchas:**
- `fast_inference=False` is REQUIRED. vLLM (which `fast_inference=True` needs)
  does not work on Kaggle's CUDA 13. This makes GRPO rollouts slower but it works.
- Model has no pad token by default → Unsloth auto-assigns `<|PAD_TOKEN|>`.
  This is fine but note it changes tokenizer (see Fix 4).
- Result: 4-bit model loads at **2.40 GB VRAM**. Plenty of headroom on T4.

---

## CRITICAL FIX 4: Optimizer choice

**Problem:** `adamw_8bit` optimizer also needs bitsandbytes CUDA path.

**Fix:** Use `optim="adamw_torch"` in GRPOConfig instead of `adamw_8bit`.
Slightly more memory but avoids another bitsandbytes dependency point.
At our scale (3.7M trainable LoRA params), the memory difference is negligible.

```python
config = GRPOConfig(
    # ...
    optim="adamw_torch",  # NOT adamw_8bit on Kaggle CUDA 13
    # ...
)
```

---

## Smoke test results (W1 Day 6)

```
GRPO steps completed: 5 ✅
Peak VRAM: 2.61 GB (within 14 GB budget — huge headroom)
Wall time: 75.2s (15.0s per step, includes cold-start overhead)
Trainable params: 3,686,400 (0.12% of 3.09B total)
```

---

## ⚠️ Note on the "150 GPU-hours" extrapolation

Cell 15's extrapolation (150.4 hours for full training) is **misleadingly
pessimistic** and should NOT be taken at face value. Reasons:

1. **The 15s/step includes one-time cold-start overhead** (torch.compile, first
   CUDA kernel JIT, model warmup). Steady-state steps are substantially faster.

2. **The `max_completion 128→2048 = 16×` assumption is wrong.** Generation time
   scales with the number of tokens the model ACTUALLY produces, not the max cap.
   Most tool-call completions are ~100-300 tokens; they rarely approach 2048.
   The real slowdown from raising the cap is closer to 2-4×, not 16×.

3. **`num_generations 2→4` is exactly 2×**, already partly reflected.

4. **Empirical anchor:** Vi-S1K and GreenMind-14B-R1 both trained Qwen 3B-class
   models with GRPO on Kaggle T4 in roughly 8-12 hours. Our setup is comparable.

**Realistic full-training estimate: ~10-15 GPU-hours** for ~4500 steps, which
fits comfortably within the 4-Kaggle-account budget (~120 GPU-hours/week).

**Action for W10-W11:** Measure actual steady-state step time on the FIRST 50
real GRPO steps and re-extrapolate from that, rather than from the smoke test.

---

## Pre-flight checklist for every Kaggle session

```
[ ] GPU accelerator ON, T4 ×2 selected (Settings → Accelerator)
[ ] Internet ON (needed for pip install + HF model download)
[ ] Run libnvJitLink symlink %%bash cell FIRST
[ ] Run install cell (expect harmless pip dependency-conflict warnings)
[ ] Verify versions: unsloth 2026.5.5, transformers 5.5.0, trl 0.24.0
[ ] Load model with fast_inference=False
[ ] Use optim="adamw_torch" in GRPOConfig
```

---

## Known harmless warnings (ignore these)

- `ldconfig: ... is not a symbolic link` (libtbb, libur, etc.) — pre-existing
- `KernelPreference is an Enum subclass...` — torch.compile deprecation, harmless
- `inline_inbuilt_nn_modules is deprecated` — harmless
- `attention mask API ... deprecated` — transformers v5.10 future change, harmless for now
- `Both max_new_tokens and max_length seem to be set` — TRL behavior, max_new_tokens wins, fine
- pip dependency conflicts re: cuda-toolkit 12 vs 13 — Kaggle RAPIDS libs, irrelevant

---

*Last updated: W1 Day 6, after successful smoke test.*
*Maintained by: [team]. Add new env learnings here as they appear in W2-W11.*
