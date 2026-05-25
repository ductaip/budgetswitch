"""
Data Generation Helper — draft trajectory generation via Groq API.

Generates DRAFT tool-use trajectories that the team then REVIEWS and corrects.
The LLM is a draft generator, NOT ground truth. Human review is the ground-truth step.

============================================================
DESIGN NOTES
- Uses Groq free tier (OpenAI-SDK compatible). 1 account is sufficient:
  we need ~200-400 trajectories total = ~600 API calls, well under the
  1000 req/day free cap if spread over a few sessions.
- DO NOT use multiple accounts to bypass rate limits — Groq ToS forbids it
  and risks suspension. Instead: checkpoint + resume across days.
- Rate-limit aware: handles 429 with exponential backoff.
- Checkpointing: every trajectory saved immediately, so you can stop/resume.

CRITICAL RULE 1 reminder:
Generated trajectories contain expected_tool + expected_params only.
No mode labels. The `difficulty` field is assigned heuristically for pilot
analysis only, never used as a training signal.
============================================================

Place file at: data/generate_drafts.py

Usage:
    export GROQ_API_KEY=gsk_...
    python data/generate_drafts.py --domain banking --count 40 --out data/vitool/drafts_banking.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.vitool.schema import (
    Trajectory, Step, Language, Difficulty,
    validate_trajectory, save_trajectories, load_trajectories,
)

# Default Groq model — Llama 3.3 70B gives GPT-4o-level quality on free tier.
DEFAULT_MODEL = "llama-3.3-70b-versatile"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


# ============================================================
# Groq client (OpenAI-SDK compatible)
# ============================================================

def get_groq_client(api_key: str = None):
    """Return an OpenAI client pointed at Groq's endpoint."""
    from openai import OpenAI
    api_key = api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Get a free key at console.groq.com "
            "(no credit card needed), then: export GROQ_API_KEY=gsk_..."
        )
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


def call_with_backoff(client, model, messages, max_retries=5, **kwargs):
    """Call Groq with exponential backoff on 429 / transient errors."""
    delay = 2.0
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
        except Exception as e:
            msg = str(e).lower()
            is_rate = "429" in msg or "rate" in msg or "too many" in msg
            if attempt == max_retries - 1:
                raise
            wait = delay * (2 ** attempt) + random.uniform(0, 1)
            if is_rate:
                # Rate limit: wait longer (Groq TPM resets each minute)
                wait = max(wait, 60.0)
                print(f"  [rate limit] waiting {wait:.0f}s (attempt {attempt+1})...")
            else:
                print(f"  [error] {e}; retrying in {wait:.0f}s...")
            time.sleep(wait)


# ============================================================
# Prompt construction for trajectory generation
# ============================================================

def build_generation_prompt(tool_catalog: dict, domain: str, n_tasks: int) -> list:
    """
    Ask the LLM to produce n_tasks draft trajectories for a domain.
    
    We request a STRICT JSON array so it parses cleanly. The team will
    review each one afterward.
    """
    domain_tools = {
        name: schema for name, schema in tool_catalog.items()
        if not name.startswith("_") and schema.get("domain") == domain
    }

    tools_desc = "\n".join(
        f"- {name}: {s['description_vi']} | params: "
        f"{json.dumps(s['parameters']['properties'], ensure_ascii=False)} | "
        f"required: {s['parameters'].get('required', [])}"
        for name, s in domain_tools.items()
    )

    system = (
        "Bạn là chuyên gia tạo dữ liệu cho hệ thống tool-using agent tiếng Việt. "
        "Nhiệm vụ: tạo các tác vụ (task) thực tế bằng tiếng Việt mà người dùng "
        "Việt Nam có thể yêu cầu, kèm chuỗi tool call đúng để giải quyết.\n\n"
        "QUY TẮC:\n"
        "1. Query phải tự nhiên, đúng văn phong người Việt thật sự gõ.\n"
        "2. Dùng format Việt Nam: ngày dd/mm/yyyy, tiền VNĐ, SĐT +84/0..., địa danh VN.\n"
        "3. Mỗi task có 1-3 steps. Đa dạng độ khó.\n"
        "4. expected_params phải KHỚP CHÍNH XÁC schema của tool.\n"
        "5. Chỉ dùng tool trong danh sách được cung cấp.\n"
        "6. Phân loại difficulty: easy (tool+param rõ trong query), "
        "medium (cần suy luận nhẹ), hard (nhiều bước hoặc tham số mơ hồ).\n\n"
        f"DANH SÁCH TOOL (domain: {domain}):\n{tools_desc}"
    )

    user = (
        f"Tạo {n_tasks} task đa dạng cho domain '{domain}'. "
        "Trả về DUY NHẤT một JSON array, không giải thích, không markdown fence. "
        "Mỗi phần tử có cấu trúc:\n"
        "{\n"
        '  "query": "<câu yêu cầu tiếng Việt tự nhiên>",\n'
        '  "steps": [\n'
        '    {"expected_tool": "<tên tool>", '
        '"expected_params": {<param>: <value>}, '
        '"fuzzy_keys": [<param cần so khớp linh hoạt như ngày tháng>], '
        '"difficulty": "easy|medium|hard"}\n'
        "  ]\n"
        "}"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def parse_generated_tasks(raw: str) -> list[dict]:
    """Parse the LLM's JSON array, tolerant of markdown fences / preamble."""
    import re
    # Strip markdown fences
    fence = re.search(r"```(?:json)?\s*(\[.*\])\s*```", raw, re.DOTALL)
    if fence:
        raw = fence.group(1)
    else:
        # Find first [ ... ] block
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1:
            raw = raw[start : end + 1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  [warn] JSON parse failed: {e}")
        return []


# ============================================================
# Main generation loop
# ============================================================

def generate_domain_drafts(
    domain: str,
    count: int,
    tool_catalog: dict,
    client,
    model: str = DEFAULT_MODEL,
    batch_size: int = 5,
    out_path: str = None,
    start_id: int = 0,
) -> list[Trajectory]:
    """
    Generate `count` draft trajectories for a domain, in batches.
    
    Streams to out_path (JSONL) for crash recovery — safe to stop/resume.
    Validates each against the catalog; invalid ones are flagged but kept
    (team reviews/fixes them).
    """
    generated = []
    seen_queries = set()

    # Resume support: load existing
    if out_path and Path(out_path).exists():
        existing = load_trajectories(out_path)
        generated.extend(existing)
        seen_queries = {t.query for t in existing}
        start_id = len(existing)
        print(f"  Resuming: {len(existing)} already generated for {domain}")

    out_f = open(out_path, "a", encoding="utf-8") if out_path else None

    while len(generated) < count:
        n_this_batch = min(batch_size, count - len(generated))
        messages = build_generation_prompt(tool_catalog, domain, n_this_batch)

        resp = call_with_backoff(
            client, model, messages,
            temperature=0.8,  # Some diversity in tasks
            max_tokens=2000,
        )
        raw = resp.choices[0].message.content
        tasks = parse_generated_tasks(raw)

        for task in tasks:
            if len(generated) >= count:
                break
            query = task.get("query", "").strip()
            if not query or query in seen_queries:
                continue
            seen_queries.add(query)

            try:
                steps = [
                    Step(
                        expected_tool=s["expected_tool"],
                        expected_params=s["expected_params"],
                        fuzzy_keys=s.get("fuzzy_keys", []),
                        difficulty=Difficulty(s.get("difficulty", "medium")),
                    )
                    for s in task["steps"]
                ]
            except (KeyError, ValueError) as e:
                print(f"  [skip] malformed task: {e}")
                continue

            traj = Trajectory(
                traj_id=f"{domain}_{start_id + len(generated):04d}",
                query=query,
                language=Language.VI,
                steps=steps,
                source="groq_draft",
                domain=domain,
            )

            # Validate (flag but keep — team will fix)
            errors = validate_trajectory(traj, tool_catalog)
            if errors:
                print(f"  [flag] {traj.traj_id}: {errors[0]}")

            generated.append(traj)
            if out_f:
                out_f.write(json.dumps(traj.to_dict(), ensure_ascii=False) + "\n")
                out_f.flush()

        print(f"  {domain}: {len(generated)}/{count} generated")
        # Gentle pacing to respect free-tier TPM (6K TPM on 70B)
        time.sleep(3)

    if out_f:
        out_f.close()
    return generated[:count]


def main():
    parser = argparse.ArgumentParser(description="Generate draft trajectories via Groq")
    parser.add_argument("--domain", required=True,
                        choices=["banking", "public_services", "ecommerce", "travel", "info_retrieval"])
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--out", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--catalog", default="data/vitool/tool_catalog.json")
    parser.add_argument("--batch-size", type=int, default=5)
    args = parser.parse_args()

    out_path = args.out or f"data/vitool/drafts_{args.domain}.jsonl"

    with open(args.catalog, encoding="utf-8") as f:
        catalog = json.load(f)

    client = get_groq_client()

    print(f"Generating {args.count} draft trajectories for '{args.domain}'...")
    print(f"Model: {args.model} | Output: {out_path}")
    print(f"⚠️  These are DRAFTS. Team MUST review each before use as gold data.\n")

    trajectories = generate_domain_drafts(
        args.domain, args.count, catalog, client,
        model=args.model, batch_size=args.batch_size, out_path=out_path,
    )

    # Summary
    from collections import Counter
    diffs = Counter(s.difficulty.value for t in trajectories for s in t.steps)
    n_steps = sum(t.num_steps for t in trajectories)
    n_flagged = sum(1 for t in trajectories if validate_trajectory(t, catalog))

    print(f"\n{'='*50}")
    print(f"Generated {len(trajectories)} trajectories ({n_steps} steps)")
    print(f"Difficulty: {dict(diffs)}")
    print(f"Flagged for review (validation errors): {n_flagged}")
    print(f"Saved to: {out_path}")
    print(f"{'='*50}")
    print("\nNEXT STEP: Team reviews each trajectory in the JSONL.")
    print("Fix any flagged ones. Verify expected_params match real-world correctness.")
    print("The LLM drafts the structure; YOUR review makes it ground truth.")


if __name__ == "__main__":
    # If run without args, do a dry-run prompt preview (no API call)
    if len(sys.argv) == 1:
        with open("data/vitool/tool_catalog.json", encoding="utf-8") as f:
            catalog = json.load(f)
        msgs = build_generation_prompt(catalog, "banking", 3)
        print("=== DRY RUN: Generation prompt preview (banking) ===\n")
        print(msgs[0]["content"][:800])
        print("\n--- USER ---")
        print(msgs[1]["content"])
        print("\n\n(Run with --domain banking --count 40 to actually generate)")
    else:
        main()