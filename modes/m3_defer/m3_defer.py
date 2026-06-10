"""
M3 Defer Mode: Delegate the current step to a frontier LLM.

Highest cost (~$0.0005/call GPT-4o-mini, ~1500ms). Used only when the SLM is
uncertain and a stronger model is warranted.

CRITICAL RULES (see CRITICAL_RULES.md):
- This module does NOT reference oracle_mode (R4).
- R1 (CONTAMINATION): M3 model PHẢI KHÁC model dùng để GEN gold data.
  Gold draft = Llama-3.3-70B (Groq) → M3 default = GPT-4o-mini (OpenAI).
  Nếu M3 trùng model gen → "quality ceiling" = model tự chấm chính nó ≈ giả.
  Hàm sẽ CẢNH BÁO nếu phát hiện trùng (xem GENERATOR_MODEL env).

NOTE: M3 is the only mode with real $ cost. During pilot, cap M3 calls to
control budget.
"""

from typing import Any, Optional
import json
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
# Lưu ý: Đảm bảo m0_fast_generate sử dụng chung logic parse chuẩn OpenAI format
from modes.m0_fast.m0_fast import _parse_tool_call


def m3_defer_generate(
    query: str,
    history: list,
    tool_schemas: dict,
    llm_client=None,
    model_name: str | None = None,
) -> dict[str, Any]:
    """
    Defer the step to a frontier LLM.
    model_name: None → tự chọn theo M3_BACKEND (openai→gpt-4o-mini, groq→deepseek-r1-distill-llama-70b).
    R1: model PHẢI khác model gen gold (Llama-70B). Tự cảnh báo nếu trùng.
    """
    if model_name is None:
        backend = os.environ.get("M3_BACKEND", "openai").lower()
        default_model = "deepseek-r1-distill-llama-70b" if backend == "groq" else "gpt-4o-mini"
        model_name = os.environ.get("M3_MODEL", default_model)
    _warn_if_contaminated(model_name)
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

    # Đảm bảo kết quả trả về luôn bọc đúng cấu trúc dict kể cả khi parse fail
    if result is None or not isinstance(result, dict):
        result = {"name": None, "parameters": {}}

    # Track cost (approximate; update rates as pricing changes)
    usage = response.usage
    result["api_cost_usd"] = _estimate_cost(
        model_name, usage.prompt_tokens, usage.completion_tokens
    )
    result["completion_tokens"] = usage.completion_tokens
    return result


def _warn_if_contaminated(model_name: str) -> None:
    """R1: M3 không được trùng model gen gold (mặc định gen = llama-3.3-70b)."""
    gen_model = os.environ.get("GENERATOR_MODEL", "llama-3.3-70b-versatile")
    if model_name == gen_model:
        warnings.warn(
            f"⚠️ R1 CONTAMINATION: M3 dùng '{model_name}' TRÙNG model gen gold "
            f"('{gen_model}'). 'Quality ceiling' sẽ bị thổi phồng (model tự chấm "
            f"chính nó). Đổi M3 sang model KHÁC (vd gpt-4o-mini).",
            stacklevel=2,
        )


def _default_client():
    """
    Client cho M3 defer. Chọn backend qua env M3_BACKEND:
      - "openai" (default): GPT-4o-mini, độc lập hệ sinh thái, sạch contamination nhất.
      - "groq": FREE, nhưng PHẢI dùng model KHÁC generator (vd deepseek-r1-distill-llama-70b).
    Trả (client, default_model).
    """
    from openai import OpenAI
    backend = os.environ.get("M3_BACKEND", "openai").lower()
    if backend == "groq":
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("M3_BACKEND=groq nhưng thiếu GROQ_API_KEY.")
        return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    # openai (default)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. M3 mặc định GPT-4o-mini (OpenAI).\n"
            "  → Muốn FREE: đặt M3_BACKEND=groq + M3_MODEL=deepseek-r1-distill-llama-70b "
            "(model KHÁC generator để giữ R1). Trên Kaggle dùng Secrets."
        )
    return OpenAI(api_key=api_key)


def _groq_client():
    """(Tùy chọn) client Groq — CHỈ dùng nếu gen gold bằng model KHÁC Groq."""
    from openai import OpenAI
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set.")
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")


def _build_m3_messages(query, history, tool_schemas):
    """Build chat messages for the frontier LLM aligned with project schema."""
    tools_desc = "\n".join(
        f"- {name}: {schema.get('description', '')} "
        f"(params: {list(schema.get('parameters', {}).get('properties', {}).keys())})"
        for name, schema in tool_schemas.items()
    )

    history_text = ""
    if history:
        history_text = "\n\nPrior steps in this trajectory:\n"
        for step, obs in history:
            # Phòng hờ trường hợp step là object hoặc dict có key khác nhau
            t_name = step.get('name') or step.get('tool_name') or getattr(step, 'name', '')
            t_params = step.get('parameters') or step.get('params') or getattr(step, 'parameters', {})
            
            # Chuyển params sang chuỗi JSON gọn để LLM dễ đọc hơn là chuỗi thô của Python
            params_str = json.dumps(t_params, ensure_ascii=False)
            history_text += f"- {t_name}({params_str}) -> {obs}\n"

    system = (
        "You are an expert tool-using agent. Given the user request and the "
        "available tools, output exactly ONE tool call as JSON: "
        "{\"name\": \"...\", \"parameters\": {...}}. Output ONLY the JSON, "
        "no explanation.\n\n"
        f"Available tools:\n{tools_desc}{history_text}"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]


# Pricing per 1M tokens (USD). Update as needed.
_PRICING = {
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "deepseek-r1-distill-llama-70b": {"input": 0.59, "output": 0.79},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
}


def _estimate_cost(model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = _PRICING.get(model_name, _PRICING["llama-3.3-70b-versatile"])
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
    assert '"name"' in msgs[0]["content"]
    print("✅ M3 message building works")

    # Test cost estimation — default giờ là gpt-4o-mini (rẻ hơn ~10x Llama)
    cost = _estimate_cost("gpt-4o-mini", 500, 100)
    print(f"✅ Cost GPT-4o-mini (500 in + 100 out): ${cost:.6f}")
    assert cost > 0

    # Test R1 contamination guard: M3 trùng generator → cảnh báo
    import warnings as _w
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        _warn_if_contaminated("llama-3.3-70b-versatile")  # = GENERATOR_MODEL mặc định
        assert any("CONTAMINATION" in str(x.message) for x in caught), "phải cảnh báo khi trùng"
    print("✅ R1 contamination guard: cảnh báo khi M3 trùng model gen")
    # M3 = gpt-4o-mini (khác generator) → KHÔNG cảnh báo
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        _warn_if_contaminated("gpt-4o-mini")
        assert not any("CONTAMINATION" in str(x.message) for x in caught)
    print("✅ R1 guard: GPT-4o-mini (khác gen) → không cảnh báo")

    print("\n✅ M3 defer (GPT-4o-mini default, R1-safe). Cần OPENAI_API_KEY khi chạy thật.")