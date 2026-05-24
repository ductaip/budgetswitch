"""
M3 Defer Mode: Delegate the current step to a frontier LLM.

Highest cost (~$0.005/call, ~1500ms). Used only when the SLM is uncertain
and a stronger model is warranted.

CRITICAL RULES (see CRITICAL_RULES.md):
- This module does NOT reference oracle_mode

NOTE: M3 is the only mode with real $ cost. During pilot study (W2), we cap
M3 calls to control budget. During GRPO training (W11), M3 cost is part of the
reward's cost penalty, which naturally discourages over-deferral.
"""

from typing import Any, Optional
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from modes.m0_fast.m0_fast import _parse_tool_call


def m3_defer_generate(
    query: str,
    history: list,
    tool_schemas: dict,
    llm_client=None,
    model_name: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """
    Defer the step to a frontier LLM (default GPT-4o-mini).
    
    Returns the same dict shape as other modes, plus 'api_cost_usd'.
    
    Args:
        llm_client: An OpenAI-compatible client. If None, attempts to
                    construct one from OPENAI_API_KEY env var.
        model_name: Frontier model to defer to. GPT-4o-mini for cost;
                    DeepSeek-V3 as cheaper backup.
    """
    if llm_client is None:
        llm_client = _default_client()

    prompt_messages = _build_m3_messages(query, history, tool_schemas)

    response = llm_client.chat.completions.create(
        model=model_name,
        messages=prompt_messages,
        temperature=0.0,
        max_tokens=300,
    )

    raw_output = response.choices[0].message.content
    result = _parse_tool_call(raw_output, tool_schemas)

    # Track cost (approximate; update rates as pricing changes)
    usage = response.usage
    result["api_cost_usd"] = _estimate_cost(
        model_name, usage.prompt_tokens, usage.completion_tokens
    )
    result["completion_tokens"] = usage.completion_tokens
    return result


def _default_client():
    """Construct an OpenAI client from env. Override for DeepSeek/other."""
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. For M3 defer, set the env var or pass "
            "llm_client explicitly. On Kaggle, use Secrets to store the key."
        )
    return OpenAI(api_key=api_key)


def _build_m3_messages(query, history, tool_schemas):
    """Build chat messages for the frontier LLM."""
    tools_desc = "\n".join(
        f"- {name}: {schema.get('description', '')} "
        f"(params: {list(schema.get('parameters', {}).get('properties', {}).keys())})"
        for name, schema in tool_schemas.items()
    )

    history_text = ""
    if history:
        history_text = "\n\nPrior steps in this trajectory:\n"
        for step, obs in history:
            history_text += f"- {step.get('tool_name')}({step.get('params')}) -> {obs}\n"

    system = (
        "You are an expert tool-using agent. Given the user request and the "
        "available tools, output exactly ONE tool call as JSON: "
        "{\"tool_name\": \"...\", \"params\": {...}}. Output ONLY the JSON, "
        "no explanation.\n\n"
        f"Available tools:\n{tools_desc}{history_text}"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]


# Pricing per 1M tokens (USD). Update as needed.
_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "deepseek-chat": {"input": 0.14, "output": 0.28},
}


def _estimate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = _PRICING.get(model_name, _PRICING["gpt-4o-mini"])
    return (
        prompt_tokens / 1e6 * rates["input"]
        + completion_tokens / 1e6 * rates["output"]
    )


if __name__ == "__main__":
    # Test message building (no API call)
    catalog = {
        "search_flight": {
            "description": "Search flights",
            "parameters": {"properties": {"origin": {}, "destination": {}, "date": {}}},
        },
    }
    msgs = _build_m3_messages("Tìm chuyến bay HN-HCM", [], catalog)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert "search_flight" in msgs[0]["content"]
    print("✅ M3 message building works")

    # Test cost estimation
    cost = _estimate_cost("gpt-4o-mini", 500, 100)
    print(f"✅ Cost estimate for 500 in + 100 out (gpt-4o-mini): ${cost:.6f}")
    assert cost > 0

    print("\nNote: Actual M3 defer requires OPENAI_API_KEY and runs in W2 Day 4-5.")