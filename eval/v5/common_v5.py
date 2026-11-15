"""
eval/v5/common_v5.py — shared, GPU-free helpers for the v5 rerun.

Used by the Kaggle runner (M0/M1/M2, teacher-forced + rollout) and the local
M3 runner. Reuses the v4 prompt builders, parser and verifier unchanged so the
v5 numbers measure the same thing as v4; only logging is added.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data.vitool.schema import load_trajectories  # noqa: E402
from env.mock_tools import WorldState, run_step  # noqa: E402
from modes.m0_fast.m0_fast import _parse_tool_call  # noqa: E402
from verifiers.universal.schema_verifier import verify_step_outcome  # noqa: E402

SEED = 42
MODES = ["M0", "M1", "M2", "M3"]


def load_catalog(path) -> dict:
    return {k: v for k, v in json.load(open(path, encoding="utf-8")).items()
            if not k.startswith("_")}


def load_eval(path):
    return load_trajectories(str(path))


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def template_hash(fn) -> str:
    """Hash of the prompt-builder source: changes iff the template changes."""
    return sha256_text(inspect.getsource(fn))[:16]


def pkg_versions(names) -> dict:
    out = {"python": platform.python_version()}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


# ------------------------------------------------------------
# Environment helpers
# ------------------------------------------------------------
def gold_prefix(traj, step_idx):
    """Same as v4 pilot.simulate_modes._replay_history, but also returns the
    world state after the gold prefix (needed to execute the model's call)."""
    world = WorldState()
    hist = []
    for s in traj.steps[:step_idx]:
        call = {"tool_name": s.expected_tool, "params": s.expected_params}
        obs = run_step(s.expected_tool, dict(s.expected_params), world)
        hist.append((call, obs))
    return hist, world


def execute(call: dict, world: WorldState):
    """Run a parsed call on a COPY of `world`.

    Returns (exec_error: bool, error_msg|None, observation|None, new_world).
    exec_error is True when the call is missing/unparsed, the tool raises
    (ToolError, KeyError for a missing param, TypeError/ValueError for bad
    types), or the observation itself reports an error.
    """
    w = copy.deepcopy(world)
    name = call.get("tool_name")
    params = call.get("params", {})
    if not name:
        return True, "no tool call parsed", None, world
    try:
        obs = run_step(name, dict(params) if isinstance(params, dict) else params, w)
    except Exception as e:  # noqa: BLE001 — any raise counts as exec error
        return True, f"{type(e).__name__}: {e}"[:300], None, world
    if isinstance(obs, dict) and "error" in obs:
        return True, str(obs["error"])[:300], obs, w
    return False, None, obs, w


def state_delta(before: WorldState, after: WorldState) -> dict:
    """Mutation-store diff (used later for execution-equivalence)."""
    b, a = before.snapshot(), after.snapshot()
    return {k: a[k] for k in a if a[k] != b[k]}


def gt_dict(step) -> dict:
    return {"expected_tool": step.expected_tool,
            "expected_params": step.expected_params,
            "fuzzy_keys": step.fuzzy_keys}


def score(raw_output: str, step, catalog, world):
    """Parse + strict-verify + execute one model output."""
    parsed = _parse_tool_call(raw_output, catalog)
    parse_ok = parsed.get("tool_name") is not None and "parse_error" not in parsed
    call = {"tool_name": parsed.get("tool_name"), "params": parsed.get("params", {})}
    try:
        success = bool(verify_step_outcome(call, gt_dict(step), catalog))
        verify_error = None
    except Exception as e:  # noqa: BLE001
        success, verify_error = False, f"{type(e).__name__}: {e}"[:300]
    exec_err, exec_msg, obs, new_world = execute(call, world)
    return {
        "parsed_call": call,
        "parse_ok": parse_ok,
        "parse_error": parsed.get("parse_error"),
        "success": success,
        "verify_error": verify_error,
        "exec_error": exec_err,
        "exec_error_msg": exec_msg,
        "state_delta": state_delta(world, new_world) if not exec_err else None,
    }, obs, new_world


def jdump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)


# ------------------------------------------------------------
# Trajectory-level helpers
# ------------------------------------------------------------
def state_match(calls: list, expected_final_state: dict) -> bool:
    """Execute the model's calls in order on a fresh world; compare the final
    mutation snapshot with the gold expected_final_state."""
    from env.mock_tools import verify_final_outcome
    try:
        return bool(verify_final_outcome(
            [{"tool_name": c.get("tool_name"), "params": c.get("params") or {}} for c in calls],
            expected_final_state))
    except Exception:  # noqa: BLE001 — malformed params etc.
        return False


_CLARIFY_WORDS = ("?", "please provide", "could you", "can you provide", "need your",
                  "cung cấp", "vui lòng", "cho biết", "bạn có thể", "xin cho", "cần thêm")


def is_clarification(raw_output: str, parse_ok: bool) -> bool:
    """No tool call was emitted and the text asks the user for information."""
    if parse_ok or not raw_output:
        return False
    t = raw_output.lower()
    return any(w in t for w in _CLARIFY_WORDS)
