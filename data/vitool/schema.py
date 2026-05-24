"""
Trajectory schema for BudgetSwitch.

A trajectory is a tool-use task: a query plus the gold sequence of tool calls
that correctly accomplishes it. This is the core data structure for both
pilot study (W2) and GRPO training (W11).

============================================================
CRITICAL RULE 1 reminder:
The `difficulty` field on each step is for PILOT ANALYSIS ONLY.
It is NEVER passed to the reward function during training.
There is no `mode` or `oracle_mode` field anywhere in this schema —
the model discovers mode policy via GRPO, not from labels.
============================================================

Place file at: data/vitool/schema.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


class Difficulty(str, Enum):
    """Step difficulty — for PILOT ANALYSIS ONLY, never a training signal."""
    EASY = "easy"       # Tool explicit in query, params verbatim, single step
    MEDIUM = "medium"   # Tool inferable, params from context, 1-2 steps
    HARD = "hard"       # Tool composition, ambiguous params, 3+ steps or recovery


class Language(str, Enum):
    VI = "vi"
    EN = "en"
    TH = "th"  # Thai (cross-lingual transfer)
    ID = "id"  # Indonesian (cross-lingual transfer)


@dataclass
class Step:
    """One gold tool-call step in a trajectory."""
    expected_tool: str
    expected_params: dict[str, Any]
    # fuzzy_keys: param keys allowed to match with format tolerance (dates, currency)
    fuzzy_keys: list[str] = field(default_factory=list)
    # difficulty: PILOT ANALYSIS ONLY — see CRITICAL RULE 1
    difficulty: Difficulty = Difficulty.MEDIUM
    # optional: expected observation after this tool executes (for multi-step)
    expected_observation: Optional[dict] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["difficulty"] = self.difficulty.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Step":
        return cls(
            expected_tool=d["expected_tool"],
            expected_params=d["expected_params"],
            fuzzy_keys=d.get("fuzzy_keys", []),
            difficulty=Difficulty(d.get("difficulty", "medium")),
            expected_observation=d.get("expected_observation"),
        )


@dataclass
class Trajectory:
    """A complete tool-use task with gold step sequence."""
    traj_id: str
    query: str
    language: Language
    steps: list[Step]
    source: str  # "bfcl_vi", "native_vi", "massive_vn", "cross_lingual", "tau_bench"
    domain: str  # "banking", "public_services", "ecommerce", "travel", "info_retrieval"
    expected_final_state: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "traj_id": self.traj_id,
            "query": self.query,
            "language": self.language.value,
            "steps": [s.to_dict() for s in self.steps],
            "source": self.source,
            "domain": self.domain,
            "expected_final_state": self.expected_final_state,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        return cls(
            traj_id=d["traj_id"],
            query=d["query"],
            language=Language(d["language"]),
            steps=[Step.from_dict(s) for s in d["steps"]],
            source=d["source"],
            domain=d["domain"],
            expected_final_state=d.get("expected_final_state", {}),
        )

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def is_multi_step(self) -> bool:
        return len(self.steps) > 1


# ============================================================
# Validation
# ============================================================

class TrajectoryValidationError(Exception):
    pass


def validate_trajectory(traj: Trajectory, tool_catalog: dict) -> list[str]:
    """
    Validate a trajectory against the tool catalog.
    
    Returns a list of error strings (empty list = valid).
    Does NOT raise; collects all issues for batch review.
    """
    errors = []

    if not traj.traj_id:
        errors.append("Missing traj_id")
    if not traj.query or not traj.query.strip():
        errors.append(f"[{traj.traj_id}] Empty query")
    if not traj.steps:
        errors.append(f"[{traj.traj_id}] No steps")

    for i, step in enumerate(traj.steps):
        # Tool must exist in catalog
        if step.expected_tool not in tool_catalog:
            errors.append(
                f"[{traj.traj_id}] Step {i}: tool '{step.expected_tool}' "
                f"not in catalog"
            )
            continue

        # Required params must be present in expected_params
        schema = tool_catalog[step.expected_tool].get("parameters", {})
        required = schema.get("required", [])
        for req_key in required:
            if req_key not in step.expected_params:
                errors.append(
                    f"[{traj.traj_id}] Step {i}: missing required param "
                    f"'{req_key}' for tool '{step.expected_tool}'"
                )

        # fuzzy_keys must reference actual params
        for fk in step.fuzzy_keys:
            if fk not in step.expected_params:
                errors.append(
                    f"[{traj.traj_id}] Step {i}: fuzzy_key '{fk}' "
                    f"not in expected_params"
                )

    return errors


# ============================================================
# JSONL I/O (one trajectory per line)
# ============================================================

def save_trajectories(trajectories: list[Trajectory], path: str) -> None:
    """Save trajectories to JSONL (one JSON object per line)."""
    with open(path, "w", encoding="utf-8") as f:
        for traj in trajectories:
            f.write(json.dumps(traj.to_dict(), ensure_ascii=False) + "\n")


def load_trajectories(path: str) -> list[Trajectory]:
    """Load trajectories from JSONL."""
    trajectories = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                trajectories.append(Trajectory.from_dict(json.loads(line)))
    return trajectories


# ============================================================
# Smoke test
# ============================================================
if __name__ == "__main__":
    # Build a sample trajectory
    traj = Trajectory(
        traj_id="vi_travel_001",
        query="Tìm chuyến bay từ Hà Nội đi TP.HCM ngày 25/12/2026 cho 2 người",
        language=Language.VI,
        source="native_vi",
        domain="travel",
        steps=[
            Step(
                expected_tool="search_flight",
                expected_params={
                    "origin": "Hà Nội",
                    "destination": "TP.HCM",
                    "date": "25/12/2026",
                    "passengers": 2,
                },
                fuzzy_keys=["date"],
                difficulty=Difficulty.EASY,
            )
        ],
        expected_final_state={"last_call": {"tool_name": "search_flight"}},
    )

    # Round-trip test
    d = traj.to_dict()
    traj2 = Trajectory.from_dict(d)
    assert traj2.traj_id == traj.traj_id
    assert traj2.steps[0].difficulty == Difficulty.EASY
    print("✅ Round-trip serialization works")

    # Validation test with mock catalog
    mock_catalog = {
        "search_flight": {
            "parameters": {
                "required": ["origin", "destination", "date"],
            }
        }
    }
    errors = validate_trajectory(traj, mock_catalog)
    assert errors == [], f"Unexpected errors: {errors}"
    print("✅ Valid trajectory passes validation")

    # Validation catches missing tool
    bad_traj = Trajectory(
        traj_id="bad_001", query="test", language=Language.VI,
        source="native_vi", domain="travel",
        steps=[Step(expected_tool="nonexistent_tool", expected_params={})],
    )
    errors = validate_trajectory(bad_traj, mock_catalog)
    assert len(errors) > 0, "Should have caught nonexistent tool"
    print(f"✅ Validation catches bad tool: {errors[0]}")

    # JSONL round-trip
    save_trajectories([traj], "/tmp/test_traj.jsonl")
    loaded = load_trajectories("/tmp/test_traj.jsonl")
    assert len(loaded) == 1
    assert loaded[0].query == traj.query
    print("✅ JSONL save/load works")

    print("\nAll schema tests passed.")