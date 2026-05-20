# Week 1 Daily Plan — Foundation

**Goal:** End W1 with (a) repo + CRITICAL_RULES in place, (b) 4 Tier-1 papers internalized, (c) Kaggle GRPO pipeline verified with dummy run, (d) email sent to PhD advisor.

**Time commitment:** ~15-20h per member this week (manageable alongside coursework).

---

## Day 1 (Today) — Repo Setup & Reading Kickoff

### Both members (2-3h):

- [ ] Clone this repo locally; verify folder structure
- [ ] Read `CRITICAL_RULES.md` carefully — internalize 3 rules
- [ ] Read `docs/README_BudgetSwitch_v9.2_FINAL.md` §1-§5 (core thesis + contributions)
- [ ] Setup shared Notion/Obsidian workspace; create note templates:
  - `paper_notes/` — one MD per paper read
  - `weekly_sync/` — meeting minutes
  - `decisions_log/` — record key design choices with rationale

### Member A (3-4h):

- [ ] Read **NVIDIA SLM Position** (arXiv 2506.02153)
  - Focus: §3 (why SLMs sufficient for agentic), §5 (modular composition argument)
  - Note: 3-5 quotable claims for our paper's Intro
- [ ] Read **ATLAS** (arXiv 2603.06713) — closest tool-agent RL competitor
  - Focus: reward formulation, training pipeline
  - Note: what's different from BudgetSwitch (orthogonal axis: tool loading vs mode routing)

### Member B (3-4h):

- [ ] Read **BCAE** (arXiv 2604.19780) — closest budget-aware RL competitor
  - Focus: §3 (curriculum scheduling), §4 (reward design)
  - Note: their reward DOES use oracle budget labels → we differentiate by not doing this
- [ ] Read **Fission-GRPO** (arXiv 2601.15625) — recovery gap data source
  - Focus: §4 Experiments (the 20% recovery number for Qwen3-8B)
  - Note: this is our primary motivation citation

### End-of-day check:

- [ ] Both repos cloned, Notion shared
- [ ] 2 papers read per member with structured notes
- [ ] Met for 30 min to discuss papers + share initial thoughts

---

## Day 2 — Reading + Kaggle Setup

### Member A (4h):

- [ ] Read **ReDAct** (arXiv 2604.07036) — closest deferral baseline
  - Focus: their perplexity threshold mechanism
  - Note: query-level vs our per-step granularity
- [ ] Begin Kaggle account setup:
  - Create 4 Kaggle accounts (or use existing if team has)
  - Verify GPU quota: each account gets ~30h/week T4
  - Setup phone verification for all accounts (required for GPU)

### Member B (4h):

- [ ] Read **Vi-S1K** (Vietnamese SLM Test-Time Scaling, Jan 2026)
  - Focus: Kaggle T4 training procedure for VN SLM
  - This is our resource-feasibility proof: "if Vi-S1K did it, we can"
- [ ] Begin Kaggle setup (parallel to Member A):
  - Test Qwen2.5-3B-Instruct inference on Kaggle T4 (notebook in `notebooks/`)
  - Verify `from unsloth import FastLanguageModel` works
  - Test 4-bit loading: peak VRAM should be ~3GB

### End-of-day check:

- [ ] All 6 Tier-1 papers read across team
- [ ] Both members have working Kaggle accounts with GPU access
- [ ] Qwen2.5-3B-Instruct inference verified on at least one Kaggle session

---

## Day 3 — Code: First Mode (M0 Fast) + Schema Verifier

### Member A (4h): Implement M0 Fast Mode

Create `modes/m0_fast/m0_fast.py`:

```python
"""
M0 Fast Mode: Direct tool call generation, no internal CoT.

Lowest cost (~150 tokens, ~200ms on T4).
"""

from unsloth import FastLanguageModel


def m0_fast_generate(
    model,
    tokenizer,
    query: str,
    history: list,
    tool_schemas: dict,
    max_new_tokens: int = 150,
) -> dict:
    """
    Generate a tool call directly, no <think> tokens, no retrieval.
    
    Args:
        model: Qwen2.5-3B-Instruct loaded via Unsloth
        tokenizer: corresponding tokenizer
        query: user query (any language)
        history: list of prior (step, observation) tuples
        tool_schemas: dict mapping tool_name -> JSON schema
    
    Returns:
        dict with keys: tool_name, params, raw_output
    """
    prompt = _build_m0_prompt(query, history, tool_schemas)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for M0 (no exploration)
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    raw_output = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], 
                                   skip_special_tokens=True)
    
    return _parse_tool_call(raw_output, tool_schemas)


def _build_m0_prompt(query, history, tool_schemas):
    """Minimal prompt: NO <think> tags, direct to tool call."""
    # Implementation: render tools, history, query in compact format
    # TODO Day 3
    raise NotImplementedError("Implement Day 3")


def _parse_tool_call(raw_output, tool_schemas):
    """Parse raw model output into {tool_name, params} dict."""
    # Implementation: regex or JSON parsing
    # TODO Day 3
    raise NotImplementedError("Implement Day 3")
```

- [ ] Implement `_build_m0_prompt` and `_parse_tool_call`
- [ ] Test on 5 hand-crafted Vietnamese queries
- [ ] Commit to repo with descriptive message

### Member B (4h): Implement Schema + Outcome Verifier

Create `verifiers/universal/schema_verifier.py`:

```python
"""
Schema and outcome verifiers.

CRITICAL RULE 1: This module MUST NOT reference oracle_mode in any form.
Verifiers check OUTCOME correctness (did tool match? did params match?),
NOT mode correctness (was M1 the right mode? was M3 cheaper?).
"""

import json
from jsonschema import validate, ValidationError


def verify_tool_match(predicted_tool: str, expected_tool: str) -> bool:
    """Check if predicted tool name matches expected."""
    return predicted_tool.strip().lower() == expected_tool.strip().lower()


def verify_schema(params: dict, schema: dict) -> bool:
    """Check if params satisfy the JSON schema of the chosen tool."""
    try:
        validate(instance=params, schema=schema)
        return True
    except ValidationError:
        return False


def verify_param_semantic(
    predicted_params: dict,
    expected_params: dict,
    tolerance: dict = None,
) -> bool:
    """
    Check if predicted params semantically match expected.
    
    Tolerance dict allows fuzzy matching (e.g., date format variations).
    """
    # TODO Day 3: implement with case sensitivity, format normalization
    raise NotImplementedError("Implement Day 3")


def verify_step_outcome(step: dict, ground_truth: dict) -> bool:
    """
    Composite outcome verifier — used directly by reward function.
    
    Args:
        step: {'tool_name': str, 'params': dict}
        ground_truth: {'expected_tool': str, 'expected_params': dict, 
                       'expected_outcome': any}
    
    Returns:
        True if step is functionally correct, False otherwise.
    
    NOTE: This function MUST NOT receive or reference oracle_mode.
    Reward function only cares about: did this step produce correct output?
    """
    if not verify_tool_match(step['tool_name'], ground_truth['expected_tool']):
        return False
    
    # Load schema for this tool (from tool catalog)
    schema = _load_schema(step['tool_name'])
    if not verify_schema(step['params'], schema):
        return False
    
    if not verify_param_semantic(step['params'], ground_truth['expected_params']):
        return False
    
    return True


def _load_schema(tool_name: str) -> dict:
    """Load JSON schema for a tool from the catalog."""
    # TODO Day 3: implement catalog loading
    raise NotImplementedError("Implement Day 3")
```

- [ ] Implement `verify_param_semantic` with reasonable fuzzy matching
- [ ] Implement `_load_schema` reading from `data/vitool/tool_catalog.json` placeholder
- [ ] Write 10 unit tests covering edge cases (case sensitivity, missing params, type mismatch)
- [ ] **Verify NO `oracle_mode` references** with grep before committing

### End-of-day check:

- [ ] M0 fast mode generates parsable tool calls on 5/5 test queries
- [ ] Schema verifier passes all 10 unit tests
- [ ] Grep check: `grep -rn "oracle_mode" verifiers/` returns nothing
- [ ] Both code committed to repo with PR descriptions

---

## Day 4 — Code: M1 Think + Integration Tests

### Member A (4h): Implement M1 Think Mode

Create `modes/m1_think/m1_think.py`:
- Same structure as M0 but with `<think>...</think>` prefix
- Higher `max_new_tokens` (~500)
- Same greedy decoding for reproducibility

- [ ] Implement M1 with `<think>` token handling
- [ ] Test on same 5 queries; compare token count to M0 (should be ~3-5× more)
- [ ] Document any prompt engineering needed

### Member B (4h): Integration tests + ground_truth schema

Create `tests/test_verifiers.py`:
- Test schema verifier on real BFCL examples (English)
- Translate 5 BFCL queries to VN manually for VN tests
- Verify: M0 + schema_verifier roundtrip works end-to-end

Create `data/vitool/tool_catalog.json` skeleton:
```json
{
  "search_flight": {
    "description": "Search for flights between cities",
    "description_vi": "Tìm chuyến bay giữa các thành phố",
    "parameters": {
      "type": "object",
      "properties": {
        "origin": {"type": "string"},
        "destination": {"type": "string"},
        "date": {"type": "string", "format": "date"}
      },
      "required": ["origin", "destination", "date"]
    }
  }
}
```

- [ ] Add 5 starter tool schemas (banking, travel, ecommerce-VN flavored)
- [ ] All integration tests pass

---

## Day 5 — Email PhD Advisor + Read Remaining Papers

### Both members (2h together):

- [ ] Polish email draft (template below)
- [ ] Identify 2-3 candidate advisors at your university:
  - Search university website for AI/NLP/ML faculty
  - Filter: anyone with publications in agentic AI, LLMs, NLP, or related
  - Prioritize: PhD with recent (2024-2026) AI/ML publications, ideally with RL/agent expertise
- [ ] Send email to ALL 2-3 advisors in parallel (not sequentially)

**Email template (Vietnamese):**

```
Subject: [NCKH] Đề xuất nghiên cứu - BudgetSwitch: Adaptive Compute 
         Allocation cho SLM Agents - Mong thầy hỗ trợ supervisorship

Kính thưa thầy [Tên thầy],

Em là [Tên A], sinh viên năm 3 ngành AI Engineering tại [Trường]. 
Cùng với bạn [Tên B], em đang chuẩn bị một đề tài NCKH về tối ưu 
compute allocation cho small language model agents — một hướng 
nghiên cứu agentic AI đang rất nóng năm 2026 (NVIDIA SLM-first 
thesis, DeepSeek-V4 thinking modes, etc.).

Sau khi nghiên cứu kỹ literature (66+ papers reviewed), nhóm em 
đã xây dựng được:
- Methodology defensible: outcome-verifiable GRPO không cần oracle 
  supervision hay LLM teacher
- Roadmap 24 tuần thực tế trên Kaggle T4 hardware
- Risk mitigation plan với 5 fallback layers
- Target submission: RIVF 2026 hoặc SOICT 2026 (Springer CCIS, Scopus)

Em xin phép đính kèm full proposal (12 trang PDF) cho thầy xem xét.

Em rất mong được thầy cân nhắc supervisorship cho project này, với 
vai trò Corresponding Author. Nhóm em commit chịu trách nhiệm toàn 
bộ execution (coding, experiments, initial paper drafting); phần 
thầy hỗ trợ sẽ là methodology guidance, paper writing feedback, 
và direction-setting trong các phases quan trọng.

Nếu thầy quan tâm và có thời gian, em xin được hẹn 1 buổi ngắn 
(15-30 phút online hoặc trực tiếp) để trình bày chi tiết hơn 
và lắng nghe feedback từ thầy về scope, focus, hoặc bất kỳ điều 
chỉnh nào thầy thấy phù hợp.

Em chân thành cảm ơn thầy đã dành thời gian xem xét!

Trân trọng,
[Tên A] - [MSSV] - [Email] - [SĐT]
[Tên B] - [MSSV] - [Email] - [SĐT]

[Đính kèm: BudgetSwitch_Proposal.pdf]
```

### Member A (2h): Read remaining Tier-1 papers

- [ ] Skim **AnytimeReasoner BRPO** (arXiv 2505.13438)
- [ ] Skim **BATS** (arXiv 2511.17006) — prompt-only baseline

### Member B (2h): Read remaining Tier-1 papers

- [ ] Skim **Thinker** (arXiv 2505.21097)
- [ ] Skim **MASSIVE-Agents** (EMNLP 2025) — get the 23% VN vs 57% EN numbers

---

## Day 6 — Kaggle GRPO Pipeline Trial (Dummy Run)

This is the most important day of W1. Goal: verify GRPO works end-to-end on Kaggle T4 with our setup, before any real implementation.

### Both members (5-6h, work in parallel):

#### Member A: Setup Unsloth GRPO on Kaggle

Create `notebooks/W1_grpo_smoke_test.ipynb`:

```python
# Cell 1: Install
!pip install unsloth vllm trl peft

# Cell 2: Load Qwen2.5-3B
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
    fast_inference=True,  # vLLM for fast generation during GRPO
)

# Cell 3: Add LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=8,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

# Cell 4: DUMMY REWARD (no real verifier yet — just to test pipeline)
def dummy_reward(prompts, completions, **kwargs):
    """Stub reward for pipeline test. Returns random rewards."""
    import random
    return [random.uniform(0, 1) for _ in completions]

# Cell 5: Tiny GRPO config
from trl import GRPOConfig, GRPOTrainer

config = GRPOConfig(
    learning_rate=5e-6,
    num_generations=2,  # Small for smoke test
    max_prompt_length=512,
    max_completion_length=256,
    max_steps=5,  # JUST 5 STEPS for pipeline test
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
)

# Cell 6: Run 5 GRPO steps
# Should complete in ~5 min on T4 if pipeline works
trainer = GRPOTrainer(
    model=model,
    args=config,
    reward_funcs=[dummy_reward],
    train_dataset=_make_dummy_dataset(),  # 10 toy prompts
)
trainer.train()

# Success criterion: completes 5 steps without OOM or exception
print("✅ GRPO smoke test passed.")
```

- [ ] Get this notebook running on Kaggle T4
- [ ] Document any issues encountered (Unsloth version, vLLM conflicts, etc.)
- [ ] Save working notebook to repo

#### Member B: Verify peak VRAM + identify Kaggle pitfalls

- [ ] Monitor VRAM during dummy GRPO: should be ≤ 14GB
- [ ] If VRAM peaks higher, try:
  - `per_device_train_batch_size=1`
  - `gradient_accumulation_steps=4`
  - Lower `max_completion_length`
- [ ] Document findings in `docs/KAGGLE_NOTES.md`
- [ ] Verify Kaggle session restart doesn't break checkpoint loading

### End-of-day check:

- [ ] Dummy GRPO completes 5 steps on Kaggle T4 without errors
- [ ] Peak VRAM documented; configuration finalized
- [ ] Notebook committed to repo

---

## Day 7 — Sync, Documentation, W2 Planning

### Both members together (2-3h):

- [ ] **Sunday sync meeting** (60 min):
  - Each member presents key papers read (3 min summary each)
  - Discuss any methodological surprises or concerns
  - Review code committed during W1
  - Verify CRITICAL_RULES.md still aligned with implementation

- [ ] Update `docs/W1_RETROSPECTIVE.md`:
  - What was completed
  - What was harder than expected
  - Open questions for advisor (if no reply yet) or to research independently
  - Calibrate W2 plan based on actual W1 velocity

- [ ] If advisor replied:
  - Schedule first meeting within 1 week
  - Prepare 1-page summary slide for that meeting
  - Note specific feedback advisor wants to give

- [ ] If no advisor reply yet:
  - Send follow-up to top-priority advisor
  - Consider 1-2 additional advisors as backup
  - Continue W2 plan regardless (we proceed with our roadmap)

### End-of-week 1 deliverables:

```
✅ Repo: budgetswitch/ with full folder structure + CRITICAL_RULES.md
✅ Reading: 6 Tier-1 papers read, notes shared
✅ Code: M0 fast mode + schema/outcome verifiers + 10 unit tests
✅ Pipeline: Kaggle GRPO smoke test passed
✅ Email: 2-3 advisors contacted
✅ Decision log: All design choices documented
```

---

## What Goes Wrong (And How to Handle)

### "Unsloth GRPO fails on Kaggle"
→ Try Colab Pro $10/month as fallback (mentioned in Vi-S1K paper)
→ Check Unsloth GitHub issue #1676 for known Kaggle GRPO issue, find workaround
→ If still broken: use PEFT + TRL native GRPO (slower but more stable)

### "No advisor replies in 7 days"
→ Send follow-up to all 3 advisors
→ Ask current professors for introductions
→ Continue project execution — advisor can join later in W4-W6

### "Member can't keep up with reading load"
→ It's fine. Focus reading on top 4 (NVIDIA SLM, BCAE, ATLAS, Fission-GRPO)
→ Other papers can be read alongside W2 work

### "Schema verifier edge cases keep failing"
→ Start simple: exact match only
→ Add fuzzy matching iteratively as data appears
→ Don't over-engineer in W1; iterate in W5-W6 when real data exists

---

## Key Reminders

🚨 **Never write `oracle_mode` in reward code.** Run grep before every commit.
🚨 **Only mode-routing LoRA trains during GRPO.** Verify in W9.
🚨 **Mode tokens are regular text + constrained decoding.** Don't modify tokenizer.

📌 **Stop reading more papers after Day 5.** You have enough to start building.
📌 **W2 is data construction (the hard week).** Don't let W1 slip and compress W2.
📌 **Document everything.** Future-you and advisor need decision logs.

---

*Last updated: W1 Day 1.*
*Next file to consult after W1: `docs/W2_DAILY_PLAN.md` (to be written end of W1).*
