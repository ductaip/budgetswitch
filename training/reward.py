"""
BudgetSwitch GRPO Reward Function.

============================================================
CRITICAL RULE 1 — ENFORCED HERE
============================================================
This function MUST NOT reference oracle_mode, expected_mode,
correct_mode, target_mode, or ANY per-step mode label.

The reward uses ONLY:
  - Deterministic step outcome (tool name, params, schema)
  - Mode cost (a fixed lookup, not a label comparison)
  - Final trajectory success

This is what makes BudgetSwitch genuine RL, not SFT-disguised-as-RL.
The model discovers mode policy through GRPO group exploration.

Verification: run `bash scripts/verify_no_oracle_leak.sh` before every commit.
============================================================

Place file at: training/reward.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verifiers.universal.schema_verifier import verify_step_outcome

# Relative compute cost per mode.
# Used in reward penalty — NOT as mode labels.
MODE_COST: dict[str, float] = {
    "M0": 1.0,
    "M1": 3.0,
    "M2": 5.0,
    "M3": 15.0,
}

# Reward hyperparameters (tuned via β-sensitivity sweep in W10)
DEFAULT_BETA = 0.05   # cost-quality trade-off; sweep range: [0.01, 0.05, 0.10, 0.20]
DEFAULT_GAMMA = 2.0   # final-success bonus weight
STEP_FAIL_PENALTY = -0.3  # penalty for a failed step


def step_reward(
    step_output: dict[str, Any],
    ground_truth: dict[str, Any],
    mode: str,
    tool_catalog: dict[str, Any],
    beta: float = DEFAULT_BETA,
) -> float:
    """
    Reward for a single step.

    Args:
        step_output: {'tool_name': str, 'params': dict} — what the agent produced.
        ground_truth: {'expected_tool': str, 'expected_params': dict,
                       'fuzzy_keys': list} — gold annotation.
                      NOTE: ground_truth MUST NOT contain 'expected_mode'.
        mode: one of M0/M1/M2/M3 — used for cost lookup only.
        tool_catalog: tool schema catalog.
        beta: cost-quality trade-off coefficient.

    Returns:
        float reward for this step.
    """
    success = verify_step_outcome(step_output, ground_truth, tool_catalog)
    r_outcome = 1.0 if success else STEP_FAIL_PENALTY
    r_cost = -beta * MODE_COST[mode]
    return r_outcome + r_cost


def trajectory_reward(
    steps: list[dict[str, Any]],
    modes: list[str],
    ground_truth_steps: list[dict[str, Any]],
    tool_catalog: dict[str, Any],
    final_success: bool,
    beta: float = DEFAULT_BETA,
    gamma: float = DEFAULT_GAMMA,
) -> float:
    """
    Reward for a complete trajectory. Used directly by GRPOTrainer.

    R(τ) = Σ_t [r_outcome(t) − β·cost(mode_t)] + γ·r_final

    Args:
        steps: list of step outputs from the agent.
        modes: list of modes chosen by the agent (same length as steps).
        ground_truth_steps: list of gold annotations (same length as steps).
               Each element MUST NOT contain 'expected_mode'.
        tool_catalog: tool schema catalog.
        final_success: whether the trajectory completed the user's task.
        beta: cost-quality trade-off.
        gamma: final-success bonus weight.

    Returns:
        scalar total reward.
    """
    assert len(steps) == len(modes) == len(ground_truth_steps), (
        f"Mismatch: steps={len(steps)}, modes={len(modes)}, "
        f"gt={len(ground_truth_steps)}"
    )

    total = 0.0
    for step_out, mode, gt in zip(steps, modes, ground_truth_steps):
        total += step_reward(step_out, gt, mode, tool_catalog, beta)

    total += gamma * (1.0 if final_success else 0.0)
    return total


def make_grpo_reward_fn(tool_catalog: dict, beta: float = DEFAULT_BETA, gamma: float = DEFAULT_GAMMA):
    """
    Factory: returns a reward function compatible with TRL GRPOTrainer.

    The returned fn signature is: fn(prompts, completions, **kwargs) -> list[float]
    (TRL passes completions as generated token strings.)

    In BudgetSwitch we parse each completion to extract:
      - the mode token (M0/M1/M2/M3)
      - the tool call JSON

    Then score against the ground truth stored in the dataset.
    """
    from verifiers.universal.schema_verifier import verify_final_outcome
    from modes.m0_fast.m0_fast import _parse_tool_call

    def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        """
        TRL-compatible reward function.

        kwargs will contain any extra columns from the dataset.
        We store ground_truth_steps and expected_final_state in the dataset.
        """
        rewards = []
        gt_steps_batch = kwargs.get("ground_truth_steps", [None] * len(completions))
        final_states_batch = kwargs.get("expected_final_states", [None] * len(completions))

        for completion, gt_steps, final_state in zip(
            completions, gt_steps_batch, final_states_batch
        ):
            if gt_steps is None:
                # Fallback for testing without real dataset
                rewards.append(0.0)
                continue

            # Parse mode token + tool call from completion
            mode, step_output = _parse_mode_and_step(completion, tool_catalog)

            # Single-step trajectory (pilot/test context)
            if isinstance(gt_steps, dict):
                gt_steps = [gt_steps]

            r = trajectory_reward(
                steps=[step_output],
                modes=[mode],
                ground_truth_steps=gt_steps,
                tool_catalog=tool_catalog,
                final_success=_check_final(step_output, final_state),
                beta=beta,
                gamma=gamma,
            )
            rewards.append(r)

        return rewards

    return reward_fn


def _parse_mode_and_step(completion: str, tool_catalog: dict) -> tuple[str, dict]:
    """
    Extract (mode, step_output) from a raw completion string.

    Expected format produced by the agent:
      M1\n{"tool_name": "...", "params": {...}}

    Gracefully handles missing mode token (defaults to M0).
    """
    from modes.m0_fast.m0_fast import _parse_tool_call

    lines = completion.strip().splitlines()
    mode = "M0"
    for line in lines[:3]:
        candidate = line.strip().upper()
        if candidate in MODE_COST:
            mode = candidate
            break

    step_output = _parse_tool_call(completion, tool_catalog)
    return mode, step_output


def _check_final(step_output: dict, expected_final_state: dict | None) -> bool:
    """Check if a step output satisfies the expected final state."""
    if expected_final_state is None:
        return step_output.get("tool_name") is not None
    last_call = expected_final_state.get("last_call", {})
    if not last_call:
        return True
    return (
        step_output.get("tool_name", "").lower()
        == last_call.get("tool_name", "").lower()
    )


# ============================================================
# Unit tests — run as script
# ============================================================
if __name__ == "__main__":
    import json

    catalog = json.load(open("data/vitool/tool_catalog.json"))

    # Test 1: correct M0 step gets positive reward
    step = {"tool_name": "check_balance", "params": {"account_number": "12345678"}}
    gt = {"expected_tool": "check_balance", "expected_params": {"account_number": "12345678"}, "fuzzy_keys": []}
    r = step_reward(step, gt, "M0", catalog)
    assert r > 0, f"Correct M0 step should have positive reward, got {r}"
    print(f"✅ Test 1: correct M0 step reward = {r:.3f}")

    # Test 2: wrong tool gets penalty
    bad_step = {"tool_name": "wrong_tool", "params": {}}
    r_bad = step_reward(bad_step, gt, "M0", catalog)
    assert r_bad < 0, f"Wrong tool should have negative reward, got {r_bad}"
    print(f"✅ Test 2: wrong tool step reward = {r_bad:.3f} (negative, correct)")

    # Test 3: M3 correct step gets positive reward but lower than M0 (cost penalised)
    r_m3 = step_reward(step, gt, "M3", catalog)
    assert r_m3 < r, f"M3 should be penalized more than M0: {r_m3} vs {r}"
    print(f"✅ Test 3: M3 reward ({r_m3:.3f}) < M0 reward ({r:.3f}) — cost penalty works")

    # Test 4: full trajectory reward
    gt_steps = [gt, gt]
    steps_out = [step, step]
    modes = ["M0", "M1"]
    r_traj = trajectory_reward(steps_out, modes, gt_steps, catalog, final_success=True)
    assert r_traj > 0
    print(f"✅ Test 4: trajectory reward (2 steps, final_success=True) = {r_traj:.3f}")

    # Test 5: CRITICAL — no oracle_mode in code (use the repo grep script)
    import subprocess
    from pathlib import Path
    result = subprocess.run(
        ["bash", "scripts/verify_no_oracle_leak.sh"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).resolve().parent.parent)
    )
    assert result.returncode == 0, f"❌ CRITICAL RULE 1 VIOLATION:\n{result.stdout}"
    print("✅ Test 5: CRITICAL RULE 1 — no oracle_mode in reward code")

    # Test 6: reward is negative for a fail-all trajectory
    r_fail = trajectory_reward([bad_step], ["M3"], [gt], catalog, final_success=False)
    assert r_fail < 0
    print(f"✅ Test 6: all-fail trajectory reward = {r_fail:.3f} (negative)")

    print("\n✅ All reward function tests passed.")