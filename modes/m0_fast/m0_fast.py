"""
M0 Fast Mode: Direct tool call generation, no internal CoT.

Lowest cost mode (~150 tokens, ~200ms on T4).
Used for simple parameter-binding steps where the tool and parameters
are clearly specified in the query.

CRITICAL RULES (see CRITICAL_RULES.md at repo root):
- This module does NOT reference oracle_mode in any way
- Mode token "M0" is a regular text token; constrained decoding handled
  in `modes/mode_router.py` (W3)
- Tool-use LoRA must be active when this module is called
"""

from typing import Any, Optional

# torch is imported lazily inside m0_fast_generate so prompt-only tests
# can run without the GPU stack installed (Kaggle has torch pre-installed;
# local laptop dev environments often don't).


def m0_fast_generate(
    model,
    tokenizer,
    query: str,
    history: list[tuple[dict, dict]],
    tool_schemas: dict[str, dict],
    max_new_tokens: int = 150,
) -> dict[str, Any]:
    """
    Generate a tool call directly, no <think> prefix, no retrieval.
    
    Args:
        model: Qwen2.5-3B-Instruct (loaded via Unsloth, 4-bit)
        tokenizer: Corresponding tokenizer
        query: User query (any language)
        history: List of (prior_step, observation) tuples
        tool_schemas: Dict mapping tool_name -> JSON schema
        max_new_tokens: Generation budget (default 150 for M0 minimal)
    
    Returns:
        Dict with keys: tool_name (str), params (dict), raw_output (str)
    """
    prompt = _build_m0_prompt(query, history, tool_schemas)
    
    # Lazy import: only needed when actually running inference
    import torch
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy: M0 is deterministic, no exploration here
            pad_token_id=tokenizer.eos_token_id,
        )
    
    raw_output = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True,
    )
    
    return _parse_tool_call(raw_output, tool_schemas)


def _build_m0_prompt(
    query: str,
    history: list[tuple[dict, dict]],
    tool_schemas: dict[str, dict],
) -> str:
    """
    Build minimal M0 prompt: no <think> tags, direct to tool call.
    
    Format used (Qwen2.5 ChatML style):
    
    <|im_start|>system
    You are a tool-using agent. Output exactly ONE tool call in JSON format:
    {"tool_name": "...", "params": {...}}
    
    Available tools:
    - tool_name_1: description...
    - tool_name_2: description...
    <|im_end|>
    <|im_start|>user
    {query}
    <|im_end|>
    <|im_start|>assistant
    """
    # Render tool catalog compactly
    tools_text = "\n".join(
        f"- {name}: {schema.get('description', '(no description)')}"
        for name, schema in tool_schemas.items()
    )
    
    # Render history compactly (if any)
    history_text = ""
    if history:
        history_text = "\n\nPrior steps:\n"
        for step, obs in history:
            history_text += f"- Called {step['tool_name']}({step['params']}) -> {obs}\n"
    
    system_msg = (
        "You are a tool-using agent. Output exactly ONE tool call in JSON "
        "format: {\"tool_name\": \"...\", \"params\": {...}}\n\n"
        f"Available tools:\n{tools_text}"
        f"{history_text}"
    )
    
    prompt = (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    
    return prompt


def _parse_tool_call(
    raw_output: str,
    tool_schemas: dict[str, dict],
) -> dict[str, Any]:
    """
    Parse raw model output into {tool_name, params, raw_output} dict.
    
    Tolerant parser: handles JSON with extra text, markdown fences, etc.
    """
    import json
    import re
    
    # Strategy 1: Look for ```json ... ``` fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_output, re.DOTALL)
    if fence_match:
        json_str = fence_match.group(1)
    else:
        # Strategy 2: Find a balanced {...} block (handles nested objects)
        json_str = _extract_first_json_object(raw_output)
        if json_str is None:
            return {
                "tool_name": None,
                "params": {},
                "raw_output": raw_output,
                "parse_error": "No JSON found in output",
            }
    
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError as e:
        return {
            "tool_name": None,
            "params": {},
            "raw_output": raw_output,
            "parse_error": f"JSON decode error: {e}",
        }
    
    return {
        "tool_name": parsed.get("tool_name"),
        "params": parsed.get("params", {}),
        "raw_output": raw_output,
    }


def _extract_first_json_object(text: str) -> Optional[str]:
    """
    Find first balanced {...} JSON object in text.
    
    Handles nested braces correctly (regex \\{.*?\\} fails on nested objects).
    Returns None if no balanced object found.
    """
    start = text.find("{")
    if start == -1:
        return None
    
    depth = 0
    in_string = False
    escape = False
    
    for i in range(start, len(text)):
        ch = text[i]
        
        if escape:
            escape = False
            continue
        
        if ch == "\\":
            escape = True
            continue
        
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        
        if in_string:
            continue
        
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    
    return None


# ============================================================
# Quick smoke test (run as script)
# ============================================================
if __name__ == "__main__":
    # Mock test without actual model loading
    sample_schemas = {
        "search_flight": {
            "description": "Search for flights between two cities",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "date": {"type": "string"},
                },
            },
        },
    }
    
    sample_query = "Tìm chuyến bay từ Hà Nội đi TP.HCM ngày 25/12/2026"
    
    prompt = _build_m0_prompt(sample_query, [], sample_schemas)
    print("=== M0 Prompt ===")
    print(prompt)
    
    # Mock raw output for parser test
    mock_output = '''```json
{"tool_name": "search_flight", "params": {"origin": "Hà Nội", "destination": "TP.HCM", "date": "25/12/2026"}}
```'''
    
    parsed = _parse_tool_call(mock_output, sample_schemas)
    print("\n=== Parsed ===")
    print(parsed)
