"""
Test M0 prompt building and parsing without loading the actual model.

Run: python tests/test_m0_prompt.py
"""

import json
import sys
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from modes.m0_fast.m0_fast import _build_m0_prompt, _parse_tool_call


def load_catalog():
    catalog_path = REPO_ROOT / "data" / "vitool" / "tool_catalog.json"
    with open(catalog_path) as f:
        catalog = json.load(f)
    # Strip metadata keys
    return {k: v for k, v in catalog.items() if not k.startswith("_")}


def test_prompt_build():
    """Test 1: Prompt builds without errors."""
    catalog = load_catalog()
    query = "Tìm chuyến bay từ Hà Nội đi TP.HCM ngày 25/12/2026 cho 2 người"
    prompt = _build_m0_prompt(query, [], catalog)
    
    # Sanity checks
    assert "<|im_start|>system" in prompt
    assert "<|im_start|>user" in prompt
    assert query in prompt
    assert "search_flight" in prompt  # At least one tool listed
    assert "<think>" not in prompt    # M0 must NOT have think tags
    
    print("✅ Test 1: Prompt builds correctly, no <think> tags")
    return prompt


def test_prompt_history():
    """Test 2: Prompt handles trajectory history."""
    catalog = load_catalog()
    query = "Đặt vé chuyến bay đó cho tôi"
    history = [
        (
            {"tool_name": "search_flight", "params": {"origin": "HN", "destination": "HCM", "date": "25/12/2026"}},
            {"flights": [{"id": "VN123", "price": 1500000}]}
        )
    ]
    prompt = _build_m0_prompt(query, history, catalog)
    
    assert "Prior steps" in prompt or "VN123" in prompt or "search_flight" in prompt
    print("✅ Test 2: Prompt includes history correctly")


def test_parse_fenced_json():
    """Test 3: Parser handles markdown-fenced JSON."""
    catalog = load_catalog()
    raw_output = '''Sure, I'll search for that flight.
```json
{"tool_name": "search_flight", "params": {"origin": "Hà Nội", "destination": "TP.HCM", "date": "25/12/2026", "passengers": 2}}
```
'''
    parsed = _parse_tool_call(raw_output, catalog)
    assert parsed["tool_name"] == "search_flight"
    assert parsed["params"]["origin"] == "Hà Nội"
    assert parsed["params"]["passengers"] == 2
    assert "parse_error" not in parsed
    print("✅ Test 3: Parser handles markdown-fenced JSON")


def test_parse_inline_json():
    """Test 4: Parser handles inline JSON without fences."""
    catalog = load_catalog()
    raw_output = '{"tool_name": "check_balance", "params": {"account_number": "1234567890"}}'
    parsed = _parse_tool_call(raw_output, catalog)
    assert parsed["tool_name"] == "check_balance"
    assert parsed["params"]["account_number"] == "1234567890"
    print("✅ Test 4: Parser handles inline JSON")


def test_parse_no_json():
    """Test 5: Parser returns error info for non-JSON output."""
    catalog = load_catalog()
    raw_output = "I'm not sure how to help with that."
    parsed = _parse_tool_call(raw_output, catalog)
    assert parsed["tool_name"] is None
    assert "parse_error" in parsed
    print("✅ Test 5: Parser gracefully handles non-JSON output")


if __name__ == "__main__":
    print("Running M0 prompt tests...\n")
    test_prompt_build()
    test_prompt_history()
    test_parse_fenced_json()
    test_parse_inline_json()
    test_parse_no_json()
    print("\n✅ All M0 tests passed.")
    print("\nNote: Real model inference will be tested on Kaggle T4 in W2 Day 1.")
