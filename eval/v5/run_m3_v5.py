"""
eval/v5/run_m3_v5.py — v5 rerun of M3 (defer to a Groq-hosted LLM), local, no GPU.

Same prompt / decoding as v4 (modes/m3_defer.py: _build_m3_messages,
temperature 0, max_tokens 1500, reasoning_effort=low for gpt-oss), gold history.
Adds full logging.

Uses ONE key: GROQ_API_KEY, from the environment or .env (never printed or
written). Rate limits (429) are handled by retry with backoff: the wait the
server asks for ("try again in ...") is honoured, otherwise exponential
backoff with jitter. No key rotation.
(The first three M3 runs and the gpt-oss 3-shot run were made with an earlier
version that rotated over several keys; same model and settings.)

Usage:
  python eval/v5/run_m3_v5.py --run 1 --out eval/v5/m3/m3_run1_v5.jsonl [--resume]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.v5.common_v5 import (  # noqa: E402
    ROOT, SEED, gold_prefix, jdump, load_catalog, load_eval, pkg_versions,
    score, sha256_file, template_hash,
)
from modes.m3_defer.m3_defer import _PRICING, _build_m3_messages, _estimate_cost  # noqa: E402

GROQ_URL = "https://api.groq.com/openai/v1"


def pick_demos(train_path, k=3, grounded_only=False):
    """k fixed demonstrations from the TRAIN split (seed 42): one easy first
    step, one medium step, one hard step with history (step_idx >= 1).
    grounded_only: the demo step AND every earlier step of its trajectory must be
    grounded under the strict audit (eval/v5/grounding_v5.py), so no demo shows
    an invented value."""
    import random
    trajs = load_eval(train_path)
    rng = random.Random(SEED)
    ok_step = None
    if grounded_only:
        g = {}
        for line in open(ROOT / "eval/v5/cache/grounding_steps_v5.jsonl"):
            r = json.loads(line)
            g[(r["traj_id"], r["step_idx"])] = r["grounded"]
        ok_step = lambda t, i: all(g[(t.traj_id, j)] for j in range(i + 1))  # noqa: E731
    want = [("easy", lambda i: i == 0), ("medium", lambda i: True), ("hard", lambda i: i >= 1)]
    demos = []
    for tier, ok in want[:k]:
        cands = [(t, i) for t in trajs for i, s in enumerate(t.steps)
                 if s.difficulty.value == tier and ok(i) and (ok_step is None or ok_step(t, i))]
        demos.append(rng.choice(cands))
    return demos


def build_messages_fewshot(query, history, catalog, demos):
    """v4 M3 messages + a fixed block of worked examples in the system prompt."""
    msgs = _build_m3_messages(query, history, catalog)
    blocks = []
    for n, (t, i) in enumerate(demos, 1):
        hist, _ = gold_prefix(t, i)
        h = "".join(f"\n  - {c['tool_name']}({json.dumps(c['params'], ensure_ascii=False)}) -> "
                    f"{json.dumps(o, ensure_ascii=False)}" for c, o in hist) or " (none)"
        s = t.steps[i]
        gold = json.dumps({"tool_name": s.expected_tool, "params": s.expected_params},
                          ensure_ascii=False)
        blocks.append(f"Example {n}:\nUser request: {t.query}\nPrior steps:{h}\nOutput: {gold}")
    ex = "\n\nWorked examples (from other tasks):\n" + "\n\n".join(blocks)
    sys_txt = msgs[0]["content"]
    cut = sys_txt.index("\n\nAvailable tools:")
    msgs[0]["content"] = sys_txt[:cut] + ex + sys_txt[cut:]
    return msgs


def load_key() -> str:
    key = os.environ.get("GROQ_API_KEY")
    p = ROOT / ".env"
    if not key and p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line.startswith("GROQ_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        raise SystemExit("GROQ_API_KEY not found in env or .env")
    return key


class GroqClient:
    """Single-key client with retry/backoff on 429 and transient errors."""

    def __init__(self, key, max_retries=12, max_wait=3600.0):
        from openai import OpenAI
        self.client = OpenAI(api_key=key, base_url=GROQ_URL, max_retries=0)
        self.max_retries = max_retries
        self.max_wait = max_wait

    def call(self, **kw):
        """Returns (response, key_idx=1, n_retries, latency_ms of the successful call)."""
        import random
        retries = 0
        while True:
            t0 = time.perf_counter()
            try:
                r = self.client.chat.completions.create(**kw)
                return r, 0, retries, (time.perf_counter() - t0) * 1000
            except Exception as e:  # noqa: BLE001
                msg = str(e)
                retries += 1
                rate = "429" in msg or "rate" in msg.lower()
                transient = any(s in msg for s in ("500", "502", "503", "timeout", "Connection"))
                if retries > self.max_retries or not (rate or transient):
                    raise
                m = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?([\d.]+)s", msg)
                if m:
                    wait = int(m.group(1) or 0) * 3600 + int(m.group(2) or 0) * 60 + float(m.group(3))
                else:
                    wait = min(300.0, 2.0 * 2 ** (retries - 1))
                wait = min(self.max_wait, wait) + random.uniform(0, 1.0)
                if wait > 60:
                    print(f"  rate-limited; waiting {wait/60:.1f} min (retry {retries})", flush=True)
                time.sleep(wait)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, required=True, help="repeat index (1..3)")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--data", default=str(ROOT / "data/vitool/pilot_eval.jsonl"))
    ap.add_argument("--catalog", default=str(ROOT / "data/vitool/tool_catalog.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--fewshot", type=int, default=0, help="0 = v4 zero-shot; 3 = fixed TRAIN demos")
    ap.add_argument("--grounded-demos", action="store_true", help="only demos without ungrounded values")
    ap.add_argument("--train", default=str(ROOT / "data/vitool/sft_train.jsonl"))
    args = ap.parse_args()

    catalog = load_catalog(args.catalog)
    trajs = load_eval(args.data)
    if args.limit:
        trajs = trajs[:args.limit]
    pool = GroqClient(load_key())
    demos = pick_demos(args.train, args.fewshot, args.grounded_demos) if args.fewshot else []
    eval_ids = {t.traj_id for t in load_eval(args.data)}
    assert not ({t.traj_id for t, _ in demos} & eval_ids), "demo leaked from eval split"

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.resume and out.exists():
        for line in out.open():
            r = json.loads(line)
            done.add((r["traj_id"], r["step_idx"]))
    meta = {
        "run": args.run, "model": args.model, "endpoint": GROQ_URL,
        "temperature": 0.0, "max_tokens": 1500,
        "reasoning_effort": "low" if "gpt-oss" in args.model else None,
        "seed_param": SEED, "protocol": "teacher_forced",
        "prompt_template_hash": template_hash(_build_m3_messages)
        + (("+fs" + template_hash(build_messages_fewshot)) if demos else ""),
        "fewshot_demos": [[t.traj_id, i] for t, i in demos],
        "grounded_demos": args.grounded_demos,
        "price_known": args.model in _PRICING,
        "data_sha256": sha256_file(args.data),
        "hardware": "Groq cloud API (hardware not disclosed); client-side wall clock",
        "packages": pkg_versions(["openai", "jsonschema"]),
        "key_policy": "single key (GROQ_API_KEY), retry+backoff",
    }
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

    total = sum(t.num_steps for t in trajs)
    f = out.open("a" if args.resume else "w", encoding="utf-8")
    n = len(done)
    for traj in trajs:
        for i, step in enumerate(traj.steps):
            if (traj.traj_id, i) in done:
                continue
            hist, world = gold_prefix(traj, i)
            msgs = (build_messages_fewshot(traj.query, hist, catalog, demos) if demos
                    else _build_m3_messages(traj.query, hist, catalog))
            kw = dict(model=args.model, messages=msgs, temperature=0.0,
                      max_tokens=1500, seed=SEED)
            if "gpt-oss" in args.model:
                kw["reasoning_effort"] = "low"
            api_error = None
            try:
                resp, key_idx, retries, lat = pool.call(**kw)
                msg = resp.choices[0].message
                raw = msg.content or ""
                used_reasoning_field = False
                if not raw.strip():  # same fallback as v4
                    raw = getattr(msg, "reasoning", None) or ""
                    used_reasoning_field = True
                u = resp.usage
                details = getattr(u, "completion_tokens_details", None)
                extra = {
                    "prompt_tokens": u.prompt_tokens,
                    "completion_tokens": u.completion_tokens,
                    "reasoning_tokens": getattr(details, "reasoning_tokens", None) if details else None,
                    "latency_ms": lat,
                    # unknown model → None (v4 helper would silently use Llama prices)
                    "api_cost_usd": (_estimate_cost(args.model, u.prompt_tokens, u.completion_tokens)
                                     if args.model in _PRICING else None),
                    "finish_reason": resp.choices[0].finish_reason,
                    "response_model": resp.model,
                    "system_fingerprint": getattr(resp, "system_fingerprint", None),
                    "reasoning_text": getattr(msg, "reasoning", None),
                    "used_reasoning_field": used_reasoning_field,
                    "key_idx": key_idx + 1, "retries": retries,
                }
            except Exception as e:  # noqa: BLE001 — logged, counted as fail
                raw, api_error = "", f"{type(e).__name__}: {e}"[:300]
                extra = {"prompt_tokens": None, "completion_tokens": None,
                         "latency_ms": None, "api_cost_usd": 0.0}
            sc, _, _ = score(raw, step, catalog, world)
            row = {"traj_id": traj.traj_id, "step_idx": i, "domain": traj.domain,
                   "difficulty": step.difficulty.value, "protocol": "teacher_forced",
                   "m3_run": args.run, "fewshot": args.fewshot, "grounded_demos": args.grounded_demos,
                   "result": {"mode": "M3", "m3_model": args.model, "raw_output": raw,
                              **sc, **extra, "api_error": api_error}}
            f.write(jdump(row) + "\n"); f.flush()
            n += 1
            if n % 20 == 0:
                print(f"[M3 run{args.run}] {n}/{total}", flush=True)
            time.sleep(args.sleep)
    f.close()
    print(f"DONE M3 run{args.run} → {out}")


if __name__ == "__main__":
    main()
