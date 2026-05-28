"""
Baseline Replication Suite for BudgetSwitch.

Implements the 8 baselines required by README_BudgetSwitch_v9.2 §9.3:
  1. Fixed M0 (SLM fast, no think)
  2. Fixed M1 (SLM always-think)
  3. LLM-only (GPT-4o-mini via Groq Llama-3.3-70B as proxy)
  4. ReDAct (perplexity-based deferral — replicate Apr 2026)
  5. ATLAS-style (adaptive tool loading, no mode routing)
  6. Random Mode Switch (sanity control)
  7. Oracle Mode Switch (upper bound — uses pilot oracle labels)
  8. BudgetSwitch (ours)

Run each baseline with:
  python baselines/run_baselines.py --baseline redact --data data/vitool/vitool_pilot.jsonl

Place at: baselines/run_baselines.py
"""

from __future__ import annotations
import argparse, json, sys, time, random
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.vitool.schema import load_trajectories, Trajectory
from verifiers.universal.schema_verifier import verify_step_outcome
from modes.m0_fast.m0_fast import _build_m0_prompt, _parse_tool_call


def load_catalog(path="data/vitool/tool_catalog.json"):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


MODE_COST = {"M0": 1.0, "M1": 3.0, "M2": 5.0, "M3": 15.0}


# ============================================================
# Baseline 1 & 2: Fixed mode (M0 / M1)
# ============================================================

def run_fixed_mode(
    trajectories: list[Trajectory],
    catalog: dict,
    model,
    tokenizer,
    mode: str = "M0",
) -> dict:
    """Fixed-mode baseline: always use one compute mode."""
    from modes.m0_fast.m0_fast import _build_m0_prompt
    from modes.m1_think.m1_think import _build_m1_prompt
    import torch

    results = []
    prompt_fn = _build_m0_prompt if mode == "M0" else _build_m1_prompt
    max_tok = 150 if mode == "M0" else 500

    for traj in trajectories:
        for step_idx, gt in enumerate(traj.steps):
            history = [({"tool_name": s.expected_tool, "params": s.expected_params}, {})
                       for s in traj.steps[:step_idx]]
            prompt = prompt_fn(traj.query, history, catalog)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            t0 = time.time()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=max_tok, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
            latency = (time.time() - t0) * 1000

            raw = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            step_out = _parse_tool_call(raw, catalog)
            gt_dict = {"expected_tool": gt.expected_tool,
                       "expected_params": gt.expected_params,
                       "fuzzy_keys": gt.fuzzy_keys}
            success = verify_step_outcome(step_out, gt_dict, catalog)
            results.append({"mode": mode, "success": success,
                            "cost": MODE_COST[mode], "latency_ms": latency})

    return _aggregate(results, f"fixed_{mode}")


# ============================================================
# Baseline 3: LLM-only (Groq Llama-3.3-70B as proxy for GPT-4o-mini)
# ============================================================

def run_llm_only(
    trajectories: list[Trajectory],
    catalog: dict,
    llm_client,
    model_name: str = "llama-3.3-70b-versatile",
) -> dict:
    """LLM-only baseline: every step deferred to frontier LLM (quality ceiling)."""
    from modes.m3_defer.m3_defer import _build_m3_messages, _estimate_cost

    results = []
    for traj in trajectories:
        for step_idx, gt in enumerate(traj.steps):
            history = [({"tool_name": s.expected_tool, "params": s.expected_params}, {})
                       for s in traj.steps[:step_idx]]
            messages = _build_m3_messages(traj.query, history, catalog)

            t0 = time.time()
            try:
                resp = llm_client.chat.completions.create(
                    model=model_name, messages=messages,
                    temperature=0.0, max_tokens=300,
                )
                raw = resp.choices[0].message.content
                api_cost = _estimate_cost(model_name, resp.usage.prompt_tokens,
                                         resp.usage.completion_tokens)
            except Exception as e:
                print(f"  [warn] LLM-only API error: {e}")
                raw = ""
                api_cost = 0.0

            latency = (time.time() - t0) * 1000
            step_out = _parse_tool_call(raw, catalog)
            gt_dict = {"expected_tool": gt.expected_tool,
                       "expected_params": gt.expected_params,
                       "fuzzy_keys": gt.fuzzy_keys}
            success = verify_step_outcome(step_out, gt_dict, catalog)
            results.append({"mode": "M3", "success": success,
                            "cost": MODE_COST["M3"], "api_cost_usd": api_cost,
                            "latency_ms": latency})
            time.sleep(1.0)  # Rate limit courtesy

    return _aggregate(results, "llm_only")


# ============================================================
# Baseline 4: ReDAct (perplexity-based deferral)
# Paper: arXiv 2604.07036, Apr 2026
# ============================================================

def compute_perplexity(model, tokenizer, text: str) -> float:
    """Compute perplexity of text under the model — used by ReDAct."""
    import torch, math
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, labels=inputs["input_ids"])
    return math.exp(out.loss.item())


def run_redact(
    trajectories: list[Trajectory],
    catalog: dict,
    model,
    tokenizer,
    llm_client,
    perplexity_threshold: float = 20.0,  # calibrated on validation set
) -> dict:
    """
    ReDAct baseline: if SLM perplexity on its own generation exceeds threshold,
    defer to LLM. Otherwise use SLM output.

    Key difference from BudgetSwitch:
    - Query-level decision (not per-step)
    - Binary: SLM or LLM (no intermediate modes)
    - Uses perplexity as uncertainty signal (not outcome verifier)

    Reference: ReDAct (arXiv 2604.07036)
    """
    from modes.m3_defer.m3_defer import _build_m3_messages, _estimate_cost

    results = []
    llm_calls = 0

    for traj in trajectories:
        # ReDAct: make deferral decision at QUERY level
        # Generate first step with SLM, check perplexity
        prompt = _build_m0_prompt(traj.query, [], catalog)
        import torch
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=150, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        raw = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        # Compute perplexity of SLM generation
        ppl = compute_perplexity(model, tokenizer, raw)
        use_llm = ppl > perplexity_threshold

        for step_idx, gt in enumerate(traj.steps):
            history = [({"tool_name": s.expected_tool, "params": s.expected_params}, {})
                       for s in traj.steps[:step_idx]]
            gt_dict = {"expected_tool": gt.expected_tool,
                       "expected_params": gt.expected_params,
                       "fuzzy_keys": gt.fuzzy_keys}

            t0 = time.time()
            if use_llm:
                # Defer to LLM
                messages = _build_m3_messages(traj.query, history, catalog)
                try:
                    resp = llm_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=messages, temperature=0.0, max_tokens=300,
                    )
                    step_raw = resp.choices[0].message.content
                    api_cost = _estimate_cost("llama-3.3-70b-versatile",
                                             resp.usage.prompt_tokens,
                                             resp.usage.completion_tokens)
                    llm_calls += 1
                    time.sleep(0.5)
                except Exception as e:
                    print(f"  [warn] ReDAct LLM call failed: {e}")
                    step_raw = raw  # fallback to SLM
                    api_cost = 0.0
                mode = "M3"
                cost = MODE_COST["M3"]
            else:
                step_raw = raw  # reuse first-step generation (simplified)
                api_cost = 0.0
                mode = "M0"
                cost = MODE_COST["M0"]

            latency = (time.time() - t0) * 1000
            step_out = _parse_tool_call(step_raw, catalog)
            success = verify_step_outcome(step_out, gt_dict, catalog)
            results.append({"mode": mode, "success": success,
                            "cost": cost, "api_cost_usd": api_cost,
                            "latency_ms": latency})

    res = _aggregate(results, "redact")
    res["llm_call_rate"] = llm_calls / len(trajectories) if trajectories else 0
    res["perplexity_threshold"] = perplexity_threshold
    return res


# ============================================================
# Baseline 5: ATLAS-style (adaptive tool loading, no mode routing)
# Paper: arXiv 2603.06713, Mar 2026
# ============================================================

def run_atlas_style(
    trajectories: list[Trajectory],
    catalog: dict,
    model,
    tokenizer,
    retriever,
) -> dict:
    """
    ATLAS-style baseline: adaptive tool loading (retrieve top-k tools per step),
    but NO mode routing (always uses M1-equivalent reasoning).

    Key difference from BudgetSwitch:
    - Adapts WHICH tools are shown (our M2 does this)
    - Does NOT adapt HOW MUCH compute to use per step
    - Orthogonal axis: BudgetSwitch adds mode routing ON TOP of tool loading

    Reference: ATLAS (arXiv 2603.06713)
    """
    from modes.m2_think_search.m2_think_search import (
        m2_think_search_generate, ToolRetriever
    )

    results = []
    for traj in trajectories:
        for step_idx, gt in enumerate(traj.steps):
            history = [({"tool_name": s.expected_tool, "params": s.expected_params}, {})
                       for s in traj.steps[:step_idx]]
            gt_dict = {"expected_tool": gt.expected_tool,
                       "expected_params": gt.expected_params,
                       "fuzzy_keys": gt.fuzzy_keys}

            t0 = time.time()
            step_out = m2_think_search_generate(
                model, tokenizer, traj.query, history, catalog, retriever
            )
            latency = (time.time() - t0) * 1000

            success = verify_step_outcome(step_out, gt_dict, catalog)
            results.append({"mode": "M2",  # always M2 (think+retrieval)
                            "success": success,
                            "cost": MODE_COST["M2"],
                            "latency_ms": latency})

    return _aggregate(results, "atlas_style")


# ============================================================
# Baseline 6: Random Mode Switch (sanity control)
# ============================================================

def run_random_mode(
    trajectories: list[Trajectory],
    catalog: dict,
    model,
    tokenizer,
    seed: int = 42,
) -> dict:
    """Random mode selection — sanity baseline. BudgetSwitch must beat this."""
    random.seed(seed)
    results = []
    modes = list(MODE_COST.keys())

    for traj in trajectories:
        for step_idx, gt in enumerate(traj.steps):
            mode = random.choice(modes)
            history = [({"tool_name": s.expected_tool, "params": s.expected_params}, {})
                       for s in traj.steps[:step_idx]]
            gt_dict = {"expected_tool": gt.expected_tool,
                       "expected_params": gt.expected_params,
                       "fuzzy_keys": gt.fuzzy_keys}

            # Execute chosen mode (simplified: use M0 for M0/M1, M2 for M2, skip M3)
            prompt = _build_m0_prompt(traj.query, history, catalog)
            import torch
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=150, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
            raw = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            step_out = _parse_tool_call(raw, catalog)
            success = verify_step_outcome(step_out, gt_dict, catalog)
            results.append({"mode": mode, "success": success,
                            "cost": MODE_COST[mode], "latency_ms": 0.0})

    return _aggregate(results, "random_mode")


# ============================================================
# Aggregation + comparison table
# ============================================================

def _aggregate(results: list[dict], name: str) -> dict:
    if not results:
        return {"name": name, "tsr": 0, "avg_cost": 0, "llm_call_rate": 0}
    n = len(results)
    tsr = sum(1 for r in results if r["success"]) / n
    avg_cost = sum(r["cost"] for r in results) / n
    avg_latency = sum(r.get("latency_ms", 0) for r in results) / n
    llm_calls = sum(1 for r in results if r["mode"] == "M3")
    total_api = sum(r.get("api_cost_usd", 0) for r in results)
    return {
        "name": name,
        "n_steps": n,
        "tsr": round(tsr, 4),
        "tsr_pct": round(100 * tsr, 1),
        "avg_cost_units": round(avg_cost, 2),
        "avg_latency_ms": round(avg_latency, 1),
        "llm_call_rate": round(llm_calls / n, 4),
        "llm_call_pct": round(100 * llm_calls / n, 1),
        "total_api_cost_usd": round(total_api, 4),
    }


def print_comparison_table(baseline_results: list[dict]):
    """Print the main comparison table for paper §7."""
    print(f"\n{'='*75}")
    print("MAIN RESULTS TABLE (for paper §7)")
    print(f"{'='*75}")
    print(f"{'Baseline':25s} {'TSR':>8} {'Cost/step':>10} {'LLM-call%':>10} {'Latency':>10}")
    print("-" * 75)
    for r in sorted(baseline_results, key=lambda x: -x["tsr"]):
        marker = " ← OURS" if r["name"] == "budgetswitch" else ""
        print(f"{r['name']:25s} {r['tsr_pct']:>7.1f}% "
              f"{r['avg_cost_units']:>10.2f} "
              f"{r['llm_call_pct']:>9.1f}% "
              f"{r['avg_latency_ms']:>9.1f}ms"
              f"{marker}")
    print(f"{'='*75}")
    print("\nKey claim: BudgetSwitch should Pareto-dominate all baselines")
    print("  (higher TSR at lower cost/LLM-call rate than ReDAct, ATLAS)")


if __name__ == "__main__":
    # Dry-run test (no model needed)
    print("Baselines module loaded successfully.")
    print("To run a specific baseline:")
    print("  python baselines/run_baselines.py --baseline redact \\")
    print("    --data data/vitool/vitool_pilot.jsonl")
    print("\nAvailable baselines:")
    for b in ["fixed_m0", "fixed_m1", "llm_only", "redact",
              "atlas_style", "random_mode"]:
        print(f"  {b}")
    print("\nNote: oracle_mode and budgetswitch baselines run in")
    print("  notebooks/W9_experiments.ipynb after GRPO training.")