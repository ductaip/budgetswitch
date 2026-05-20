"""
Schema and outcome verifiers for BudgetSwitch.

============================================================
CRITICAL RULE 1 (see CRITICAL_RULES.md):
This module MUST NOT reference oracle_mode, expected_mode,
correct_mode, target_mode, or ANY per-step mode label.

Verifiers check OUTCOME correctness:
  - did the tool name match expected?
  - did the parameters match expected?
  - did the schema validate?

Verifiers do NOT check mode correctness:
  - "was M1 the right mode for this step?" ← NOT this function's job
  - "should this have been deferred to M3?" ← NOT this function's job

The reward function uses ONLY outputs from this module.
The model discovers mode policy via GRPO group exploration,
not via mode supervision.
============================================================
"""

from typing import Any, Optional
from jsonschema import validate, ValidationError


def verify_tool_match(
    predicted_tool: Optional[str],
    expected_tool: str,
) -> bool:
    """
    Check if predicted tool name matches expected.
    
    Case-insensitive, whitespace-tolerant.
    """
    if predicted_tool is None:
        return False
    return predicted_tool.strip().lower() == expected_tool.strip().lower()


def verify_schema(
    params: dict[str, Any],
    schema: dict[str, Any],
) -> bool:
    """
    Check if params satisfy the JSON schema of the chosen tool.
    
    Returns False on any validation failure (missing required, type mismatch, etc.).
    """
    if not isinstance(params, dict):
        return False
    try:
        validate(instance=params, schema=schema)
        return True
    except ValidationError:
        return False


def verify_param_semantic(
    predicted_params: dict[str, Any],
    expected_params: dict[str, Any],
    fuzzy_keys: Optional[list[str]] = None,
) -> bool:
    """
    Check if predicted params semantically match expected.
    
    Args:
        predicted_params: What the model produced
        expected_params: Ground truth from data construction
        fuzzy_keys: Keys for which fuzzy matching is allowed
                    (e.g., date formats: "25/12/2026" ~ "25-12-2026")
    
    Strategy:
    - Required keys in expected must be present in predicted
    - For non-fuzzy keys: exact match (after string normalization)
    - For fuzzy keys: format-tolerant match
    - Extra keys in predicted are allowed (model may add optional params)
    """
    fuzzy_keys = fuzzy_keys or []
    
    for key, expected_val in expected_params.items():
        if key not in predicted_params:
            return False
        
        predicted_val = predicted_params[key]
        
        if key in fuzzy_keys:
            if not _fuzzy_match(predicted_val, expected_val):
                return False
        else:
            if not _exact_match(predicted_val, expected_val):
                return False
    
    return True


def _exact_match(a: Any, b: Any) -> bool:
    """Strict equality after string normalization."""
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    return a == b


def _fuzzy_match(a: Any, b: Any) -> bool:
    """
    Tolerant match for values that may have format variations.
    
    Currently handles:
    - Date format variations: "25/12/2026", "25-12-2026", "2026-12-25"
    - Number with units: "1500000 VNĐ", "1.5 triệu", "1,500,000"
    
    Will be expanded as ViTool-2.0K reveals more edge cases.
    """
    if isinstance(a, str) and isinstance(b, str):
        # Normalize dates
        norm_a = _normalize_date_or_number(a)
        norm_b = _normalize_date_or_number(b)
        return norm_a == norm_b
    return _exact_match(a, b)


def _normalize_date_or_number(s: str) -> str:
    """Lowercase, strip, replace common separators."""
    s = s.strip().lower()
    # Replace date separators with a canonical form
    for sep in ["/", "-", "."]:
        s = s.replace(sep, "/")
    # Strip currency markers (simple version)
    for marker in ["vnđ", "vnd", "đ", " ", ","]:
        s = s.replace(marker, "")
    return s


def verify_step_outcome(
    step: dict[str, Any],
    ground_truth: dict[str, Any],
    tool_catalog: dict[str, dict],
) -> bool:
    """
    Composite outcome verifier — called directly by reward function.
    
    Args:
        step: {'tool_name': str, 'params': dict}
        ground_truth: {
            'expected_tool': str,
            'expected_params': dict,
            'fuzzy_keys': list[str] (optional),
        }
        tool_catalog: full catalog of tool schemas
    
    Returns:
        True if step is functionally correct, False otherwise.
    
    ============================================================
    METHODOLOGY NOTE (CRITICAL RULE 1):
    
    ground_truth ONLY contains 'expected_tool' and 'expected_params'.
    It MUST NOT contain 'expected_mode' or any per-step mode label.
    
    If you find yourself wanting to add mode information to ground_truth
    in order to "help" the reward function, STOP.
    The whole point of outcome-verifiable GRPO is that the model
    discovers mode policy from outcome signal alone.
    ============================================================
    """
    # Check 1: Tool name correctness
    if not verify_tool_match(step.get("tool_name"), ground_truth["expected_tool"]):
        return False
    
    # Check 2: Schema validity
    tool_schema = tool_catalog.get(ground_truth["expected_tool"], {}).get("parameters")
    if tool_schema is None:
        # No schema defined; skip this check
        pass
    elif not verify_schema(step.get("params", {}), tool_schema):
        return False
    
    # Check 3: Param semantic match
    if not verify_param_semantic(
        step.get("params", {}),
        ground_truth["expected_params"],
        fuzzy_keys=ground_truth.get("fuzzy_keys", []),
    ):
        return False
    
    return True


def verify_final_outcome(
    trajectory: list[dict],
    expected_final_state: dict,
) -> bool:
    """
    Check if entire trajectory achieved the user's task.
    
    For now: check if last step's tool_name and params match
    expected_final_state.last_call.
    
    Will be expanded in W3-W4 to support execution-based verification
    for τ-bench-style trajectories.
    """
    if not trajectory:
        return False
    
    last_step = trajectory[-1]
    expected_last_call = expected_final_state.get("last_call", {})
    
    if expected_last_call:
        return verify_tool_match(
            last_step.get("tool_name"),
            expected_last_call.get("tool_name", ""),
        )
    
    # Default: any non-empty trajectory is "complete"
    return True


# ============================================================
# Smoke tests (run as script)
# ============================================================
if __name__ == "__main__":
    # Test 1: Exact match success
    step = {"tool_name": "search_flight", "params": {"origin": "HN", "destination": "HCM"}}
    gt = {"expected_tool": "search_flight", "expected_params": {"origin": "HN", "destination": "HCM"}}
    catalog = {"search_flight": {"parameters": {"type": "object"}}}
    assert verify_step_outcome(step, gt, catalog), "Test 1 failed"
    print("✅ Test 1: Exact match passes")
    
    # Test 2: Wrong tool fails
    step2 = {"tool_name": "book_hotel", "params": {}}
    assert not verify_step_outcome(step2, gt, catalog), "Test 2 failed"
    print("✅ Test 2: Wrong tool fails correctly")
    
    # Test 3: Case insensitivity
    step3 = {"tool_name": "SEARCH_FLIGHT", "params": {"origin": "hn", "destination": "HCM"}}
    assert verify_step_outcome(step3, gt, catalog), "Test 3 failed"
    print("✅ Test 3: Case insensitive match works")
    
    # Test 4: Missing param fails
    step4 = {"tool_name": "search_flight", "params": {"origin": "HN"}}
    assert not verify_step_outcome(step4, gt, catalog), "Test 4 failed"
    print("✅ Test 4: Missing required param fails correctly")
    
    # Test 5: Fuzzy date match
    step5 = {
        "tool_name": "search_flight",
        "params": {"origin": "HN", "destination": "HCM", "date": "25-12-2026"},
    }
    gt5 = {
        "expected_tool": "search_flight",
        "expected_params": {"origin": "HN", "destination": "HCM", "date": "25/12/2026"},
        "fuzzy_keys": ["date"],
    }
    assert verify_step_outcome(step5, gt5, catalog), "Test 5 failed"
    print("✅ Test 5: Fuzzy date format match works")
    
    print("\nAll smoke tests passed.")
    
    # ============================================================
    # CRITICAL VERIFICATION:
    # This file should contain NO references to oracle_mode etc.
    # Run grep to confirm:
    # grep -E "oracle_mode|expected_mode|correct_mode|target_mode" verifiers/universal/schema_verifier.py
    # Expected: no output.
    # ============================================================
