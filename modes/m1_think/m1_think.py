"""
M1 Think Mode: Internal chain-of-thought before tool call.

Medium cost (~500 tokens, ~600ms on T4).
Used for steps where the tool or parameters require some reasoning to determine.

CRITICAL RULES (see CRITICAL_RULES.md):
- This module does NOT reference oracle_mode
- Mode token "M1" handled by mode_router (W3); this module just executes the M1 path
"""

from typing import Any

# Reuse the parser from M0 (single source of truth for tool-call parsing)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from modes.m0_fast.m0_fast import _parse_tool_call


def m1_think_generate(
    model,
    tokenizer,
    query: str,
    history: list,
    tool_schemas: dict,
    max_new_tokens: int = 500,
) -> dict[str, Any]:
    """
    Generate a tool call with an internal chain-of-thought first.
    
    The model is prompted to reason inside <think>...</think> before
    emitting the JSON tool call. We parse only the tool call (think content
    is discarded for execution, kept in raw_output for analysis).
    """
    prompt = _build_m1_prompt(query, history, tool_schemas)

    import torch
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs.input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    raw_output = tokenizer.decode(
        outputs[0][input_length:],
        skip_special_tokens=True,
    )

    result = _parse_tool_call(raw_output, tool_schemas)
    # Track approximate token cost for pilot/reward analysis
    result["completion_tokens"] = outputs[0].shape[0] - input_length
    return result


def _build_m1_prompt(query, history, tool_schemas):
    """M1 prompt: instructs the model to think before the tool call."""
    tools_text = "\n".join(
        f"- {name}: {schema.get('description', '')}"
        for name, schema in tool_schemas.items()
    )

    history_text = ""
    if history:
        history_text = "\n\nPrior steps:\n"
        for step, obs in history:
            history_text += f"- Called {step.get('tool_name')}({step.get('params')}) -> {obs}\n"

    system_msg = (
        "You are a tool-using agent. First reason step-by-step inside "
        "<think>...</think> tags about which tool and parameters to use. "
        "Then output exactly ONE tool call in JSON format: "
        "{\"tool_name\": \"...\", \"params\": {...}}\n\n"
        f"Available tools:\n{tools_text}"
        f"{history_text}"
    )

    return (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>"
    )


if __name__ == "__main__":
    catalog = {
        "search_flight": {"description": "Search flights"},
        "book_flight": {"description": "Book a flight"},
    }
    prompt = _build_m1_prompt("Đặt vé Hà Nội đi HCM", [], catalog)
    assert "<think>" in prompt
    print("✅ M1 prompt includes <think> tag")
    print(prompt[-200:])