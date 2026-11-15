"""
eval/v5/run_local_modes_v5.py — v5 rerun of the local modes (M0/M1/M2) on GPU.

Two tasks:
  --task tf       teacher-forced (gold history), same protocol as v4,
                  M0/M1/M2 on all 284 held-out steps.
  --task rollout  end-to-end: each mode sees its OWN previous calls and the
                  env observations they produced (errors included). M0 and M1.

Model loading and prompts are identical to v4 (pilot/run_pilot.py +
modes/*). Changes vs v4 are logging only:
  - raw output, parsed call, parse_ok, exec_error, state_delta
  - prompt_tokens / completion_tokens for every mode (v4 M0 was always 0)
  - latency = time.perf_counter() around model.generate only (cuda-synced);
    M2 retrieval latency logged separately
  - hardware, model id, adapter sha256, prompt template hash, package versions

Usage (Kaggle):
  python eval/v5/run_local_modes_v5.py --task tf --lora /path/tool_use_lora_v2 \
      --out /kaggle/working/local_tf_v5.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from eval.v5.common_v5 import (  # noqa: E402
    ROOT, SEED, execute, gold_prefix, jdump, load_catalog, load_eval,
    pkg_versions, score, sha256_file, template_hash,
)
from env.mock_tools import WorldState  # noqa: E402
from modes.m0_fast.m0_fast import _build_m0_prompt  # noqa: E402
from modes.m1_think.m1_think import _build_m1_prompt  # noqa: E402
from modes.m2_think_search.m2_think_search import _build_m2_prompt  # noqa: E402

MAX_NEW = {"M0": 150, "M1": 500, "M2": 800}  # same budgets as v4
BUILDERS = {"M0": _build_m0_prompt, "M1": _build_m1_prompt, "M2": _build_m2_prompt}


def load_model(base_model, lora_path):
    """Identical to pilot/run_pilot.py:load_sft_model (4-bit nf4, fp16)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, device_map="auto",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True),
    )
    model = PeftModel.from_pretrained(model, lora_path)
    model.eval()
    return model, tok


def generate(model, tok, prompt, max_new):
    import torch
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    n_in = int(inputs.input_ids.shape[1])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    torch.cuda.synchronize()
    lat = (time.perf_counter() - t0) * 1000
    gen = out[0][n_in:]
    raw = tok.decode(gen, skip_special_tokens=True)
    return raw, n_in, int(gen.shape[0]), lat


def run_mode(mode, model, tok, retriever, catalog, query, hist):
    """One generation. Returns (raw, extra-log dict)."""
    extra = {}
    schemas = catalog
    if mode == "M2":
        t0 = time.perf_counter()
        names = retriever.retrieve(query, hist, k=10)  # v4: query only
        extra["retrieval_ms"] = (time.perf_counter() - t0) * 1000
        extra["retrieved_tools"] = names
        schemas = {n: catalog[n] for n in names if n in catalog}
    prompt = BUILDERS[mode](query, hist, schemas)
    raw, n_in, n_out, lat = generate(model, tok, prompt, MAX_NEW[mode])
    extra.update(prompt_tokens=n_in, completion_tokens=n_out, latency_ms=lat,
                 hit_max_new_tokens=n_out >= MAX_NEW[mode])
    return raw, extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["tf", "rollout"], required=True)
    ap.add_argument("--modes", default=None, help="comma list; default tf=M0,M1,M2 rollout=M0,M1")
    ap.add_argument("--data", default=str(ROOT / "data/vitool/pilot_eval.jsonl"))
    ap.add_argument("--catalog", default=str(ROOT / "data/vitool/tool_catalog.json"))
    ap.add_argument("--emb", default=str(ROOT / "data/vitool/tool_emb.npy"))
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--lora", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    modes = (args.modes or ("M0,M1,M2" if args.task == "tf" else "M0,M1")).split(",")

    import numpy as np
    import torch
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    catalog = load_catalog(args.catalog)
    trajs = load_eval(args.data)
    if args.limit:
        trajs = trajs[:args.limit]

    model, tok = load_model(args.base_model, args.lora)
    retriever = None
    if "M2" in modes:
        from modes.m2_think_search.m2_think_search import ToolRetriever
        retriever = ToolRetriever(args.emb, args.catalog)
        retriever.retrieve("warmup")  # load e5 before timing anything

    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    run_meta = {
        "task": args.task,
        "base_model": args.base_model,
        "adapter_path": args.lora,
        "adapter_sha256": sha256_file(Path(args.lora) / "adapter_model.safetensors"),
        "data_sha256": sha256_file(args.data),
        "hardware": f"{len(gpus)}x {gpus[0] if gpus else 'cpu'} (device_map=auto, 4-bit nf4)",
        "decoding": "greedy (do_sample=False)", "seed": SEED,
        "max_new_tokens": MAX_NEW,
        "prompt_template_hash": {m: template_hash(BUILDERS[m]) for m in modes},
        "retriever": "intfloat/multilingual-e5-base, k=10" if retriever else None,
        "packages": pkg_versions(["torch", "transformers", "peft", "bitsandbytes",
                                  "accelerate", "sentence-transformers", "jsonschema"]),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    meta_path = Path(args.out).with_suffix(".meta.json")
    meta_path.write_text(json.dumps(run_meta, indent=2, ensure_ascii=False))
    print(json.dumps(run_meta, indent=2, ensure_ascii=False), flush=True)

    total = sum(t.num_steps for t in trajs)
    f = open(args.out, "w", encoding="utf-8")
    done = 0
    t_start = time.time()

    if args.task == "tf":
        for traj in trajs:
            for i, step in enumerate(traj.steps):
                hist, world = gold_prefix(traj, i)
                row = {"traj_id": traj.traj_id, "step_idx": i, "domain": traj.domain,
                       "difficulty": step.difficulty.value, "protocol": "teacher_forced",
                       "results": {}}
                for m in modes:
                    raw, extra = run_mode(m, model, tok, retriever, catalog, traj.query, hist)
                    sc, _, _ = score(raw, step, catalog, world)
                    row["results"][m] = {"mode": m, "raw_output": raw, **sc, **extra,
                                         "api_cost_usd": 0.0}
                f.write(jdump(row) + "\n"); f.flush()
                done += 1
                if done % 10 == 0:
                    el = time.time() - t_start
                    print(f"[tf] {done}/{total} steps, {el/60:.1f} min, "
                          f"eta {el/done*(total-done)/60:.0f} min", flush=True)
    else:
        for m in modes:
            for traj in trajs:
                world, hist = WorldState(), []
                for i, step in enumerate(traj.steps):
                    raw, extra = run_mode(m, model, tok, retriever, catalog, traj.query, hist)
                    sc, obs, new_world = score(raw, step, catalog, world)
                    if sc["exec_error"]:
                        obs = {"error": sc["exec_error_msg"]}
                    else:
                        world = new_world
                    hist.append((sc["parsed_call"], obs))
                    row = {"traj_id": traj.traj_id, "step_idx": i, "domain": traj.domain,
                           "difficulty": step.difficulty.value, "protocol": "rollout",
                           "mode": m, "raw_output": raw, **sc, **extra,
                           "observation": obs}
                    f.write(jdump(row) + "\n"); f.flush()
                final = world.snapshot()
                f.write(jdump({"traj_id": traj.traj_id, "protocol": "rollout_final",
                               "mode": m, "final_state": final,
                               "final_state_match": final == traj.expected_final_state}) + "\n")
                f.flush()
                done += traj.num_steps
                print(f"[rollout {m}] {done}/{total*len(modes)} steps, "
                      f"{(time.time()-t_start)/60:.1f} min", flush=True)
    f.close()
    print(f"DONE {args.task} → {args.out}", flush=True)


if __name__ == "__main__":
    main()
