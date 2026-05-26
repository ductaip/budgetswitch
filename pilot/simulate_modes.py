"""
Mode Simulation Harness — engine for the pilot study (paper §3).

v1.1 fixes:
- Bug 1: filter _metadata keys from tool_catalog before passing to mode functions
- Bug 2: documented that libnvJitLink symlink MUST run before any bitsandbytes import

CRITICAL RULE 1: oracle_mode labels from this harness are for ANALYSIS ONLY.
They are NEVER used in the GRPO reward function.

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

MODE_COST = {"M0": 1.0, "M1": 3.0, "M2": 5.0, "M3": 15.0}
MODES = ["M0", "M1", "M2", "M3"]


# ============================================================
# FIX Bug 1: always strip _metadata keys from catalog
# ============================================================

def clean_catalog(catalog: dict) -> dict:
    """
    Remove metadata keys (starting with '_') from tool catalog.

    The catalog JSON includes _description, _version, _domains etc.
    These are strings, not tool schemas — passing them to mode functions
    causes 'str object has no attribute get' errors.

    Always call this before passing catalog to any mode function or verifier.
    """
    return {k: v for k, v in catalog.items() if not k.startswith("_")}


# ============================================================
# Data classes
# ============================================================

@dataclass
class ModeResult:
    mode: str
    success: bool
    cost: float
    latency_ms: float
    completion_tokens: int = 0
    api_cost_usd: float = 0.0


@dataclass
class StepSimulation:
    traj_id: str
    step_idx: int
    difficulty: str
    results: dict[str, ModeResult]

    def oracle_mode(self) -> str:
        """
        Cheapest mode that succeeds — ANALYSIS + UPPER BOUND ONLY.
        NEVER used as a training signal.
        """
        successful = [(m, MODE_COST[m]) for m in MODES if self.results[m].success]
        return min(successful, key=lambda x: x[1])[0] if successful else "M3"

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
    tool_catalog: dict,   # must already be clean (no _ keys)
    model=None,
    tokenizer=None,
    retriever=None,
    llm_client=None,
) -> dict[str, Any]:
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
    # Fix Bug 1: strip metadata before anything uses the catalog
    catalog = clean_catalog(tool_catalog)

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
            results[mode] = ModeResult(mode="M3", success=False,
                                       cost=MODE_COST["M3"], latency_ms=0.0)
            continue

        start = time.time()
        try:
            output = run_single_mode(
                mode, traj.query, history, catalog,
                model=model, tokenizer=tokenizer,
                retriever=retriever, llm_client=llm_client,
            )
            success = verify_step_outcome(output, gt_dict, catalog)
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
            latency_ms=latency, completion_tokens=comp_tokens,
            api_cost_usd=api_cost,
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
    # Fix Bug 1: strip metadata once at entry point
    catalog = clean_catalog(tool_catalog)

    all_sims = []
    out_f = open(output_path, "w", encoding="utf-8") if output_path else None
    total_steps = sum(t.num_steps for t in trajectories)
    done = 0

    for traj in trajectories:
        for step_idx in range(traj.num_steps):
            sim = simulate_step(
                traj, step_idx, catalog,
                model=model, tokenizer=tokenizer,
                retriever=retriever, llm_client=llm_client,
                skip_m3=skip_m3,
            )
            all_sims.append(sim)
            if out_f:
                out_f.write(json.dumps(sim.to_dict(), ensure_ascii=False) + "\n")
                out_f.flush()
            done += 1
            if verbose and done % 10 == 0:
                print(f"  Simulated {done}/{total_steps} steps...")

    if out_f:
        out_f.close()
    if verbose:
        print(f"✅ Simulated {done} steps across {len(trajectories)} trajectories.")
    return all_sims


# ============================================================
# Smoke test
# ============================================================
if __name__ == "__main__":
    from data.vitool.schema import Language, Difficulty

    # Catalog WITH metadata keys (simulating the real file)
    catalog_with_meta = {
        "_description": "some string",   # ← this caused the bug
        "_version": "0.2",
        "search_flight": {
            "description": "Search flights",
            "domain": "travel",
            "parameters": {"type": "object", "required": ["origin"]}
        },
    }

    traj = Trajectory(
        traj_id="test_001", query="Tìm chuyến bay HN-HCM", language=Language.VI,
        source="native_vi", domain="travel",
        steps=[Step(expected_tool="search_flight",
                    expected_params={"origin": "HN", "destination": "HCM"},
                    difficulty=Difficulty.EASY)],
    )

    # Patch run_single_mode for offline test
    def mock_run(mode, query, history, catalog, **kwargs):
        # Verify catalog is clean (no _ keys)
        assert not any(k.startswith("_") for k in catalog), \
            f"Bug 1 still present: {[k for k in catalog if k.startswith('_')]}"
        if mode == "M0":
            return {"tool_name": "wrong_tool", "params": {}}
        return {"tool_name": "search_flight",
                "params": {"origin": "HN", "destination": "HCM"}}

    globals()["run_single_mode"] = mock_run

    sim = simulate_step(traj, 0, catalog_with_meta)  # pass raw catalog with meta
    print("Mode results:")
    for mode, r in sim.results.items():
        print(f"  {mode}: success={r.success}, cost={r.cost}")

    assert sim.results["M0"].success == False
    assert sim.results["M1"].success == True
    assert sim.oracle_mode() == "M1"
    print(f"Oracle mode: {sim.oracle_mode()}")
    print("✅ Bug 1 fixed: metadata keys correctly filtered")
    print("✅ Mode simulation harness works correctly")