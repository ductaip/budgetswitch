"""
Mode Simulation Harness — the engine for the pilot study (paper §3).

For each step in a set of trajectories, run all 4 compute modes offline and
record (success, cost, latency). This produces the data that:
  1. Motivates §3 of the paper (the overthink/underthink gap)
  2. Serves as the "Oracle Mode" upper-bound baseline in experiments
  3. Acts as the W2 DECISION GATE (does the gap actually exist?)

============================================================
CRITICAL RULE 1 reminder:
The oracle mode labels this harness produces are used ONLY for analysis
and the upper-bound baseline. They are NEVER fed to the GRPO reward function.
============================================================

Place file at: pilot/simulate_modes.py
"""

from __future__ import annotations

import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.vitool.schema import Trajectory, Step, load_trajectories
from verifiers.universal.schema_verifier import verify_step_outcome


# Relative compute cost per mode (used in pilot Pareto + later reward)
MODE_COST = {"M0": 1.0, "M1": 3.0, "M2": 5.0, "M3": 15.0}
MODES = ["M0", "M1", "M2", "M3"]


@dataclass
class ModeResult:
    """Result of running one mode on one step."""
    mode: str
    success: bool
    cost: float
    latency_ms: float
    completion_tokens: int = 0
    api_cost_usd: float = 0.0


@dataclass
class StepSimulation:
    """All-mode results for a single step."""
    traj_id: str
    step_idx: int
    difficulty: str  # for analysis only
    results: dict[str, ModeResult]  # mode -> ModeResult

    def oracle_mode(self) -> str:
        """
        Cheapest mode that succeeds (the analysis-only oracle).
        If no mode succeeds, returns the highest mode (M3).
        NEVER used as a training signal — analysis + upper bound only.
        """
        successful = [(m, MODE_COST[m]) for m in MODES if self.results[m].success]
        if not successful:
            return "M3"
        return min(successful, key=lambda x: x[1])[0]

    def to_dict(self) -> dict:
        return {
            "traj_id": self.traj_id,
            "step_idx": self.step_idx,
            "difficulty": self.difficulty,
            "oracle_mode": self.oracle_mode(),
            "results": {m: asdict(r) for m, r in self.results.items()},
        }


# ============================================================
# Mode dispatch
# ============================================================

def run_single_mode(
    mode: str,
    query: str,
    history: list,
    tool_catalog: dict,
    model=None,
    tokenizer=None,
    retriever=None,
    llm_client=None,
) -> dict[str, Any]:
    """
    Dispatch one mode and return its tool-call dict.
    
    Lazily imports mode modules so the harness can be unit-tested with mocks.
    """
    if mode == "M0":
        from modes.m0_fast.m0_fast import m0_fast_generate
        return m0_fast_generate(model, tokenizer, query, history, tool_catalog)
    elif mode == "M1":
        from modes.m1_think.m1_think import m1_think_generate
        return m1_think_generate(model, tokenizer, query, history, tool_catalog)
    elif mode == "M2":
        from modes.m2_think_search.m2_think_search import m2_think_search_generate
        return m2_think_search_generate(
            model, tokenizer, query, history, tool_catalog, retriever
        )
    elif mode == "M3":
        from modes.m3_defer.m3_defer import m3_defer_generate
        return m3_defer_generate(query, history, tool_catalog, llm_client)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def simulate_step(
    traj: Trajectory,
    step_idx: int,
    tool_catalog: dict,
    model=None,
    tokenizer=None,
    retriever=None,
    llm_client=None,
    modes_to_run: Optional[list[str]] = None,
    skip_m3: bool = False,
) -> StepSimulation:
    """
    Run all (or selected) modes on a single step and record results.
    
    Args:
        skip_m3: if True, skip the API-cost M3 mode (for budget control in pilot).
                 M3 result is recorded as failed/zero so Pareto still computes.
    """
    modes_to_run = modes_to_run or MODES
    gt_step = traj.steps[step_idx]
    history = [
        ({"tool_name": s.expected_tool, "params": s.expected_params},
         s.expected_observation or {})
        for s in traj.steps[:step_idx]
    ]

    gt_dict = {
        "expected_tool": gt_step.expected_tool,
        "expected_params": gt_step.expected_params,
        "fuzzy_keys": gt_step.fuzzy_keys,
    }

    results = {}
    for mode in modes_to_run:
        if mode == "M3" and skip_m3:
            results[mode] = ModeResult(mode="M3", success=False, cost=MODE_COST["M3"],
                                       latency_ms=0.0)
            continue

        start = time.time()
        try:
            output = run_single_mode(
                mode, traj.query, history, tool_catalog,
                model=model, tokenizer=tokenizer,
                retriever=retriever, llm_client=llm_client,
            )
            success = verify_step_outcome(output, gt_dict, tool_catalog)
            comp_tokens = output.get("completion_tokens", 0)
            api_cost = output.get("api_cost_usd", 0.0)
        except Exception as e:
            print(f"  [warn] {mode} failed on {traj.traj_id} step {step_idx}: {e}")
            success = False
            comp_tokens = 0
            api_cost = 0.0

        latency = (time.time() - start) * 1000
        results[mode] = ModeResult(
            mode=mode, success=success, cost=MODE_COST[mode],
            latency_ms=latency, completion_tokens=comp_tokens, api_cost_usd=api_cost,
        )

    return StepSimulation(
        traj_id=traj.traj_id,
        step_idx=step_idx,
        difficulty=gt_step.difficulty.value,
        results=results,
    )


def simulate_trajectories(
    trajectories: list[Trajectory],
    tool_catalog: dict,
    model=None,
    tokenizer=None,
    retriever=None,
    llm_client=None,
    skip_m3: bool = False,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> list[StepSimulation]:
    """
    Run mode simulation over a list of trajectories.
    
    Returns a flat list of StepSimulation (one per step across all trajectories).
    Optionally streams results to output_path (JSONL) for crash recovery.
    """
    all_sims = []
    out_f = open(output_path, "w", encoding="utf-8") if output_path else None

    total_steps = sum(t.num_steps for t in trajectories)
    done = 0

    for traj in trajectories:
        for step_idx in range(traj.num_steps):
            sim = simulate_step(
                traj, step_idx, tool_catalog,
                model=model, tokenizer=tokenizer,
                retriever=retriever, llm_client=llm_client,
                skip_m3=skip_m3,
            )
            all_sims.append(sim)
            if out_f:
                out_f.write(json.dumps(sim.to_dict(), ensure_ascii=False) + "\n")
                out_f.flush()
            done += 1
            if verbose and done % 25 == 0:
                print(f"  Simulated {done}/{total_steps} steps...")

    if out_f:
        out_f.close()
    if verbose:
        print(f"✅ Simulated {done} steps across {len(trajectories)} trajectories.")
    return all_sims


# ============================================================
# Unit test with mock model (no GPU needed)
# ============================================================
if __name__ == "__main__":
    from data.vitool.schema import Language, Difficulty

    # Mock catalog
    catalog = {"search_flight": {"parameters": {"type": "object", "required": ["origin"]}}}

    # Mock trajectory
    traj = Trajectory(
        traj_id="test_001",
        query="Tìm chuyến bay HN-HCM",
        language=Language.VI,
        source="native_vi",
        domain="travel",
        steps=[Step(expected_tool="search_flight",
                    expected_params={"origin": "HN", "destination": "HCM"},
                    difficulty=Difficulty.EASY)],
    )

    # Monkeypatch run_single_mode to simulate different mode behaviors.
    # NOTE: simulate_step calls run_single_mode by its module-global name,
    # so we patch it in this module's namespace (globals()).
    def mock_run(mode, query, history, catalog, **kwargs):
        # Simulate: M0 fails (too fast), M1/M2/M3 succeed
        if mode == "M0":
            return {"tool_name": "wrong_tool", "params": {}}
        return {"tool_name": "search_flight", "params": {"origin": "HN", "destination": "HCM"}}

    globals()["run_single_mode"] = mock_run

    sim = simulate_step(traj, 0, catalog)
    print("Mode results:")
    for mode, r in sim.results.items():
        print(f"  {mode}: success={r.success}, cost={r.cost}")
    print(f"Oracle mode (cheapest success): {sim.oracle_mode()}")
    assert sim.oracle_mode() == "M1", "M1 should be cheapest success"
    print("✅ Mode simulation harness works (oracle = M1 as expected)")