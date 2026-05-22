# Week 2 Daily Plan — Data Pipeline + Pilot Study Foundation

**Goal:** End W2 with (a) tool schema expanded to ~30-40 tools, (b) first ~400-500
trajectories built with gold annotations, (c) error injection / mode simulation
pipeline scaffolded, (d) pilot study runnable on first 100 trajectories.

**Why W2 is the hard week:** Data construction is the highest-effort, lowest-glamour
phase. If W2 slips, everything downstream slips. Protect this week.

**Time commitment:** ~18-22h per member.

> **Prerequisite:** W1 smoke test passed ✅. Pipeline confirmed working on Kaggle T4.
> If you haven't yet sent advisor emails or finished paper reading, do those in
> parallel (evenings) but do NOT let them block data construction.

---

## Reframing: What "data" means for BudgetSwitch

We are NOT building a linguistic-error dataset (that was the abandoned v1-v8 direction).

We ARE building **tool-use trajectories** where each trajectory is:
```
{
  "query": "Đặt vé từ HN đi HCM ngày 25/12/2026 cho 2 người",
  "language": "vi",
  "steps": [
    {
      "expected_tool": "search_flight",
      "expected_params": {"origin": "HN", "destination": "HCM", "date": "25/12/2026", "passengers": 2},
      "fuzzy_keys": ["date"],
      "difficulty": "easy"   // for pilot analysis ONLY, never for training reward
    },
    ...
  ],
  "expected_final_state": {...}
}
```

The `difficulty` field is used ONLY in the pilot study (§6) for analysis. It is
NEVER passed to the reward function (Rule 1).

---

## Day 1 (W2) — Verify real model inference + expand tool schema

### Both members (3-4h):

- [ ] **Run M0 fast mode with REAL model on Kaggle** (not just prompt test):
  - Load Qwen2.5-3B (use `docs/KAGGLE_NOTES.md` setup)
  - Run `m0_fast_generate` on 5 Vietnamese queries from `tool_catalog.json`
  - Verify: model produces parsable tool calls (even if not always correct)
  - This validates the M0 path works with a real model, not just prompt logic

### Member A (4h): Expand tool schema — domains 1-2

- [ ] Banking domain: 8 tools (check_balance, transfer, transaction_history,
  open_account, card_block, loan_inquiry, interest_rates, branch_locator)
- [ ] Public services domain: 8 tools (lookup_bhxh, tax_inquiry, civil_status,
  vehicle_registration, utility_bill, public_appointment, document_status, fee_lookup)
- [ ] Each tool: name, description (EN + VI), domain, JSON schema with realistic
  Vietnamese param formats (dates dd/mm/yyyy, currency VNĐ, phone +84)

### Member B (4h): Expand tool schema — domains 3-5

- [ ] E-commerce domain: 8 tools (search_product, add_to_cart, checkout,
  track_order, return_request, compare_prices, apply_voucher, seller_info)
- [ ] Travel domain: 8 tools (search_flight, book_flight, search_hotel,
  book_hotel, search_train, cancel_booking, check_in, baggage_info)
- [ ] Info retrieval domain: 6 tools (get_weather, news_search, translate,
  currency_convert, calculate, wiki_lookup)

**Milestone Day 1:** ~38 tools across 5 domains in `data/vitool/tool_catalog.json`.
Real M0 inference verified on Kaggle.

---

## Day 2 — Trajectory schema + first BFCL translations

### Both members (1h): Agree on trajectory schema

- [ ] Finalize the trajectory JSON format (see "Reframing" above)
- [ ] Create `data/vitool/schema.py` with a `Trajectory` dataclass + validator
- [ ] Decide difficulty heuristic rules (write down in `docs/decisions_log.md`):
  - easy: tool explicit in query, params verbatim, single step
  - medium: tool inferable, params from context, 1-2 steps
  - hard: tool composition, ambiguous params, 3+ steps OR recovery needed

### Member A (4h): BFCL → VN translation (batch 1)

- [ ] Pull 50 single-turn tasks from BFCL-v3 (simple/multiple categories)
- [ ] Translate queries to natural Vietnamese (use GPT-4o assist, then manual review)
- [ ] Map BFCL tools to our tool_catalog (or add new tools as needed)
- [ ] Annotate expected_tool + expected_params per step
- [ ] Target: 50 trajectories this session

### Member B (4h): Native VN tasks (batch 1)

- [ ] Write 50 native Vietnamese tasks across banking + public services
- [ ] These should reflect realistic VN scenarios (BHXH lookup, bank transfer, etc.)
- [ ] Annotate gold trajectories
- [ ] Target: 50 trajectories this session

**Milestone Day 2:** Trajectory schema finalized, ~100 trajectories drafted.

---

## Day 3 — Gold trajectory generation + quality control round 1

### Both members (4h):

- [ ] **Set up gold generation pipeline** (`data/generate_gold.py`):
  - For each task, call GPT-4o (or Claude) to produce a reference trajectory
  - Prompt: "Given query + available tools, produce the correct sequence of tool calls"
  - Parse + validate against schema
  - Budget: ~$10-15 this session (track spending in `docs/api_budget.md`)

- [ ] **Quality control round 1:**
  - Cross-review: A reviews B's 50, B reviews A's 50
  - Flag any incorrect expected_tool or expected_params
  - Compute rough inter-annotator agreement on difficulty labels

### Member A (2h): Continue BFCL translation (batch 2)
- [ ] 50 more BFCL → VN (multi-turn / parallel categories)

### Member B (2h): Continue native VN (batch 2)
- [ ] 50 more native VN (e-commerce + travel)

**Milestone Day 3:** ~200 trajectories with gold annotations, QC round 1 done.

---

## Day 4 — Mode simulation harness (the pilot study engine)

This is the technically important day. Build the offline mode simulation that
powers both the pilot study (§6) AND later reward computation.

### Member A (5h): Build M1 + M2 modes

- [ ] Implement `modes/m1_think/m1_think.py` (CoT prefix, ~500 tokens)
  - Same structure as M0 but with `<think>...</think>` before tool call
- [ ] Implement `modes/m2_think_search/m2_think_search.py`:
  - First retrieve top-k tools via embedding similarity (precompute embeddings!)
  - Then think + generate with retrieved tools in context
  - **Pre-compute tool embeddings now** (see IMPLEMENTATION_PITFALLS.md §8.4):
    ```python
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer("intfloat/multilingual-e5-base")
    tool_texts = [f"{name}: {schema['description']}" for name, schema in catalog.items()]
    tool_embeddings = embedder.encode(tool_texts, normalize_embeddings=True)
    np.save("embeddings/tool_embeddings.npy", tool_embeddings)
    ```

### Member B (5h): Build mode simulation harness

- [ ] Implement `pilot/simulate_modes.py`:
  ```python
  def simulate_all_modes(trajectory, model, tokenizer, tool_catalog, llm_api):
      """For each step, run all 4 modes offline, record (success, cost, latency)."""
      results = []
      for step_idx, gt_step in enumerate(trajectory.steps):
          history = trajectory.steps[:step_idx]
          step_result = {}
          for mode in ["M0", "M1", "M2", "M3"]:
              output = run_mode(mode, trajectory.query, history, tool_catalog, ...)
              success = verify_step_outcome(output, gt_step, tool_catalog)
              cost = MODE_COST[mode]
              step_result[mode] = {"success": success, "cost": cost}
          results.append(step_result)
      return results
  ```
- [ ] M3 defer: call GPT-4o-mini, parse its tool call, verify (budget ~$5)

**Milestone Day 4:** All 4 modes implemented; mode simulation harness works on
1 sample trajectory end-to-end.

---

## Day 5 — Run pilot study on first 100 trajectories

### Both members (5h):

- [ ] Run `pilot/simulate_modes.py` on first 100 trajectories
  - This is ~100 trajectories × ~3-5 steps × 4 modes = ~1500 mode executions
  - M0/M1/M2 on Kaggle T4; M3 via API (~$10)
  - Estimated runtime: 1-2 hours on Kaggle

- [ ] Generate pilot artifacts (`pilot/analyze_pilot.py`):
  - **Figure 1:** Mode-success heatmap (4 modes × 3 difficulty classes)
  - **Figure 2:** Cost-success Pareto (fixed-mode vs oracle)
  - **Table 1:** Per-mode token/cost/success summary
  - Save to `pilot/results/`

- [ ] **Verify the core hypotheses:**
  - H1 (overthink-easy): does M0 succeed ≥85% on easy, M1 wasteful?
  - H2 (underthink-hard): does success increase with mode cost on hard steps?
  - H3 (allocation gap): does oracle beat uniform-M1 at lower cost?

**Milestone Day 5:** Pilot study complete. We now have EMPIRICAL data showing the
overthink/underthink gap — the motivation for §3 of the paper. If hypotheses
don't hold, this is a critical early signal (see "Decision Gate" below).

---

## Day 6 — Analyze pilot, decision gate, plan W3

### Both members (4h):

- [ ] **DECISION GATE: Do the pilot results support the thesis?**

  **If YES (H1, H2, H3 mostly hold):**
  - Document results in `pilot/results/pilot_report.md`
  - Proceed confidently to W3 (constrained decoding + cold-start SFT)
  - Use these numbers to motivate §3 of paper

  **If PARTIAL (gap exists but smaller than hoped):**
  - Still proceed, but adjust framing
  - Maybe fewer modes suffice (e.g., M0/M1/M3 without M2)
  - Note for ablation #1 (number of modes)

  **If NO (no clear allocation gap):**
  - STOP and reconsider. This is exactly why we run pilot before GRPO.
  - Options: (a) harder task distribution (more multi-step), (b) pivot to
    Fallback 2 (empirical study of compute modes), (c) discuss with advisor
  - Better to discover this in W2 than after W11 training.

- [ ] Continue building trajectories toward the 2,000 target
  (W2 builds ~400-500; W3-W5 continue to 2,000)

**Milestone Day 6:** Decision gate passed (or pivot decided). Pilot report written.

---

## Day 7 — Sync, documentation, W3 planning

### Both members (3h):

- [ ] **Sunday sync:**
  - Review pilot results together
  - Confirm decision gate outcome
  - Update advisor (if engaged) with pilot findings — this is a great concrete
    update to send: "We empirically validated the core hypothesis"

- [ ] Update docs:
  - `docs/W2_RETROSPECTIVE.md`
  - `docs/decisions_log.md` (difficulty heuristics, mode cost values chosen)
  - `docs/api_budget.md` (track spending — should be ~$30-40 so far)

- [ ] Draft `docs/W3_DAILY_PLAN.md`:
  - W3 focus: constrained decoding for mode tokens (Rule 3 + Pitfall 2),
    cold-start SFT data prep, continue trajectory construction

**End-of-W2 deliverables:**
```
✅ Tool catalog: ~38 tools across 5 domains
✅ Trajectories: ~400-500 built with gold annotations (toward 2000)
✅ All 4 modes implemented (M0, M1, M2, M3)
✅ Pre-computed tool embeddings (Pitfall 8.4 addressed)
✅ Mode simulation harness working
✅ Pilot study complete with empirical motivation for §3
✅ Decision gate passed
```

---

## Critical reminders for W2

🚨 **Difficulty labels are for pilot analysis ONLY.** Never pass `difficulty`
   or any oracle mode info to the reward function. (Rule 1)

🚨 **Pre-compute tool embeddings once** (Day 4). Don't re-encode tools every
   retrieval — that's the M2 bottleneck from Pitfall 8.4.

🚨 **The pilot study is a decision gate, not a formality.** If the overthink/
   underthink gap doesn't appear empirically, better to know now than after
   12 GPU-hours of GRPO. Be honest with the data.

📌 **Track API spending** in `docs/api_budget.md`. Target ≤$40 through W2.

📌 **Quality > quantity for trajectories.** 400 clean trajectories beat 800 noisy
   ones. The reward function depends on correct gold annotations.

---

## What goes wrong in W2 (and how to handle)

### "Translating BFCL to natural Vietnamese is slow"
→ Use GPT-4o for first-pass translation, manual review only. Don't translate
  word-by-word; aim for natural VN that a real user would type.

### "Gold trajectory from GPT-4o is sometimes wrong"
→ That's why we manually review. GPT-4o is a draft generator, not ground truth.
  The human review IS the ground truth step.

### "Pilot shows no clear mode gap"
→ Check task difficulty distribution first — maybe too many easy tasks.
  Add more multi-step / composition tasks. If gap still absent, invoke decision gate.

### "Mode simulation is slow on Kaggle"
→ M0/M1 are fast. M2 with pre-computed embeddings is fast. M3 (API) is the
  bottleneck — batch API calls, use GPT-4o-mini (cheap), cache results.

### "API budget creeping up"
→ M3 simulation is the main cost. For pilot, 100 trajectories is enough — don't
  simulate all 2000 in pilot. Full mode data only needed for GRPO reward, computed
  on-the-fly during training (not pre-simulated).

---

*Created: W1 Day 6, after smoke test success.*
*Next: execute W2, then draft W3 plan based on W2 velocity.*
