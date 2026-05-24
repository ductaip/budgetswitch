"""
M2 Think+Search Mode: CoT + dynamic retrieval over the tool catalog.

Higher cost (~800 tokens + retrieval, ~800ms on T4).
Used for steps where the right tool is not obvious and must be retrieved
from a large catalog before reasoning.

CRITICAL RULES (see CRITICAL_RULES.md):
- This module does NOT reference oracle_mode

IMPLEMENTATION PITFALL 8.4 (see docs/IMPLEMENTATION_PITFALLS.md):
- Tool embeddings are PRE-COMPUTED ONCE (offline) and loaded here.
- We only encode the QUERY at runtime (~10ms), never re-encode tools.
- This avoids the M2 retrieval bottleneck during GRPO rollouts.
"""

from typing import Any, Optional
from pathlib import Path
import json
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from modes.m0_fast.m0_fast import _parse_tool_call


# ============================================================
# Tool retriever (pre-computed embeddings)
# ============================================================

class ToolRetriever:
    """
    Retrieves top-k relevant tools by query-embedding similarity.
    
    Tool embeddings are precomputed once via build_tool_embeddings() and
    saved to disk. At runtime, only the query is encoded.
    """

    def __init__(self, embeddings_path: str, catalog_path: str, embedder=None):
        self.embeddings = np.load(embeddings_path)  # [num_tools, dim]
        with open(catalog_path, encoding="utf-8") as f:
            catalog = json.load(f)
        self.tool_names = [k for k in catalog if not k.startswith("_")]
        assert len(self.tool_names) == self.embeddings.shape[0], (
            f"Mismatch: {len(self.tool_names)} tools vs "
            f"{self.embeddings.shape[0]} embeddings. Rebuild embeddings."
        )
        self.catalog = catalog
        self._embedder = embedder  # lazy; only needed to encode queries

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("intfloat/multilingual-e5-base")
        return self._embedder

    def retrieve(self, query: str, history: list = None, k: int = 10) -> list[str]:
        """Return top-k tool names most relevant to the query."""
        embedder = self._get_embedder()
        # multilingual-e5 convention: prefix queries with "query: "
        q_emb = embedder.encode([f"query: {query}"], normalize_embeddings=True)
        scores = (q_emb @ self.embeddings.T)[0]  # [num_tools]
        top_k_idx = np.argsort(scores)[-k:][::-1]
        return [self.tool_names[i] for i in top_k_idx]


def build_tool_embeddings(
    catalog_path: str,
    output_path: str,
    embedder=None,
) -> None:
    """
    Pre-compute tool embeddings ONCE. Run this in W2 Day 4 setup.
    
    Saves embeddings to output_path as .npy. Order matches catalog key order
    (excluding _metadata keys).
    """
    from sentence_transformers import SentenceTransformer
    if embedder is None:
        embedder = SentenceTransformer("intfloat/multilingual-e5-base")

    with open(catalog_path, encoding="utf-8") as f:
        catalog = json.load(f)

    tool_names = [k for k in catalog if not k.startswith("_")]
    # multilingual-e5 convention: prefix passages with "passage: "
    tool_texts = [
        f"passage: {name}: {catalog[name].get('description', '')} "
        f"{catalog[name].get('description_vi', '')}"
        for name in tool_names
    ]
    embeddings = embedder.encode(tool_texts, normalize_embeddings=True)
    np.save(output_path, embeddings)
    print(f"✅ Saved {embeddings.shape[0]} tool embeddings to {output_path}")
    print(f"   Embedding dim: {embeddings.shape[1]}")


# ============================================================
# M2 generation
# ============================================================

def m2_think_search_generate(
    model,
    tokenizer,
    query: str,
    history: list,
    tool_schemas: dict,
    retriever: ToolRetriever,
    k: int = 10,
    max_new_tokens: int = 800,
) -> dict[str, Any]:
    """
    Retrieve top-k tools, then think + generate a tool call using only
    the retrieved subset (simulates working with a focused tool set).
    """
    # Step 1: retrieve relevant tools (fast — only query is encoded)
    relevant_tool_names = retriever.retrieve(query, history, k=k)
    relevant_schemas = {
        name: tool_schemas[name]
        for name in relevant_tool_names
        if name in tool_schemas
    }

    # Step 2: think + generate with retrieved subset
    prompt = _build_m2_prompt(query, history, relevant_schemas)

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
    result["completion_tokens"] = outputs[0].shape[0] - input_length
    result["retrieved_tools"] = relevant_tool_names
    return result


def _build_m2_prompt(query, history, relevant_schemas):
    """M2 prompt: retrieved tool subset + think instruction."""
    tools_text = "\n".join(
        f"- {name}: {schema.get('description', '')}"
        for name, schema in relevant_schemas.items()
    )

    history_text = ""
    if history:
        history_text = "\n\nPrior steps:\n"
        for step, obs in history:
            history_text += f"- Called {step.get('tool_name')}({step.get('params')}) -> {obs}\n"

    system_msg = (
        "You are a tool-using agent. The following tools were retrieved as "
        "most relevant to the request. Reason step-by-step inside "
        "<think>...</think>, then output exactly ONE tool call in JSON: "
        "{\"tool_name\": \"...\", \"params\": {...}}\n\n"
        f"Retrieved tools:\n{tools_text}"
        f"{history_text}"
    )

    return (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{query}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>"
    )


if __name__ == "__main__":
    # Test prompt building (no model/embeddings needed)
    catalog = {
        "search_flight": {"description": "Search flights"},
        "book_hotel": {"description": "Book hotel"},
    }
    prompt = _build_m2_prompt("Tìm khách sạn Đà Nẵng", [], catalog)
    assert "<think>" in prompt
    assert "Retrieved tools" in prompt
    print("✅ M2 prompt builds with retrieved tools + <think>")
    print("\nNote: ToolRetriever + build_tool_embeddings require")
    print("      sentence-transformers and run on Kaggle in W2 Day 4.")