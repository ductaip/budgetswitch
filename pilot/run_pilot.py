"""
pilot/run_pilot.py — Chạy pilot end-to-end trên ViTool (v10.0). Kaggle T4-ready.

Pipeline: load Qwen-3B+tool_use_lora (4-bit) + e5 retriever (M2) + M3 client
(GPT-4o-mini/DeepSeek) → simulate_trajectories chạy 4 mode/step → xuất JSONL.
Output này là INPUT cho eval/oracle.py → eval/figures.py.

CHẠY (Kaggle, GPU T4):
  # 1) precompute tool embeddings cho M2 (1 lần)
  python -c "from modes.m2_think_search.m2_think_search import build_tool_embeddings; \
             build_tool_embeddings('data/vitool/tool_catalog.json','data/vitool/tool_emb.npy')"
  # 2) chạy pilot
  export OPENAI_API_KEY=...        # cho M3 (hoặc M3_BACKEND=groq + GROQ_API_KEY)
  python pilot/run_pilot.py \
      --data data/vitool/vitool_final.jsonl \
      --lora models/tool_use_lora \
      --emb  data/vitool/tool_emb.npy \
      --out  pilot/pilot_results.jsonl \
      --max-m3 200          # cap số M3 call để kiểm soát $ (M3 quá cap → skip, đánh fail)

  # 3) phân tích
  python eval/oracle.py  --in pilot/pilot_results.jsonl --out eval/oracle_report.json
  python eval/figures.py --in eval/oracle_report.json   --outdir eval/figs

Lưu ý:
- M0/M1/M2 = Qwen-3B (LoRA). M3 = model KHÁC (GPT-4o-mini/DeepSeek) — R1.
- --max-m3 để không đốt $ M3: pilot ~600 step × M3 mỗi step = tốn. Cap lại,
  các step vượt cap M3 = fail (cost vẫn tính) → vẫn fair cho oracle.
- --limit để chạy thử subset trước khi full.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.vitool.schema import load_trajectories  # noqa: E402


# ============================================================
# Load SFT model (Qwen-3B + tool_use_lora), 4-bit cho T4
# ============================================================
def load_sft_model(base_model: str, lora_path: str, load_4bit: bool = True):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {"torch_dtype": torch.float16, "device_map": "auto"}
    if load_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(base_model, **kwargs)

    if lora_path and Path(lora_path).exists():
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path)
        print(f"✅ Loaded LoRA: {lora_path}")
    else:
        print(f"⚠️ Không thấy LoRA tại '{lora_path}' — dùng base model (M0/M1/M2 sẽ yếu).")
    model.eval()
    return model, tokenizer


# ============================================================
# M3 client (R1: KHÁC generator)
# ============================================================
def make_m3_client():
    from openai import OpenAI
    backend = os.environ.get("M3_BACKEND", "openai").lower()
    if backend == "groq":
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise SystemExit("M3_BACKEND=groq cần GROQ_API_KEY.")
        print("M3 = DeepSeek-70B trên Groq (free, khác generator Llama → R1 OK)")
        return OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise SystemExit(
            "M3 cần OPENAI_API_KEY (GPT-4o-mini). Muốn free: M3_BACKEND=groq + GROQ_API_KEY."
        )
    print("M3 = GPT-4o-mini (OpenAI)")
    return OpenAI(api_key=key)


# ============================================================
# M3 cap wrapper: giới hạn số M3 call để kiểm soát $
# ============================================================
class M3Budget:
    def __init__(self, max_calls: int):
        self.max = max_calls
        self.used = 0

    def allow(self) -> bool:
        if self.used < self.max:
            self.used += 1
            return True
        return False


def main():
    ap = argparse.ArgumentParser(description="ViTool pilot runner")
    ap.add_argument("--data", default="data/vitool/vitool_final.jsonl")
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--lora", default="models/tool_use_lora")
    ap.add_argument("--emb", default="data/vitool/tool_emb.npy", help="tool embeddings cho M2")
    ap.add_argument("--catalog", default="data/vitool/tool_catalog.json")
    ap.add_argument("--out", default="pilot/pilot_results.jsonl")
    ap.add_argument("--max-m3", type=int, default=200, help="cap số M3 call (kiểm soát $)")
    ap.add_argument("--limit", type=int, default=0, help=">0 để chạy thử subset")
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="kiểm config/data, KHÔNG load model")
    args = ap.parse_args()

    import json
    catalog = {k: v for k, v in json.load(open(args.catalog)).items()}
    trajs = load_trajectories(args.data)
    if args.limit > 0:
        trajs = trajs[:args.limit]
    print(f"Pilot trên {len(trajs)} trajectory ({sum(t.num_steps for t in trajs)} steps). "
          f"Out → {args.out}")

    if args.dry_run:
        print("✅ dry-run: data load OK, config OK. Bỏ load model.")
        return

    # M3 client + budget
    m3_client = make_m3_client()
    m3_budget = M3Budget(args.max_m3)

    # bọc client để áp cap M3 (vượt cap → raise để simulate đánh fail)
    class CappedClient:
        def __init__(self, inner, budget):
            self._inner = inner
            self._budget = budget
            self.chat = self  # cho .chat.completions.create

        @property
        def completions(self):
            return self

        def create(self, **kw):
            if not self._budget.allow():
                raise RuntimeError("M3 budget exhausted (skip → fail)")
            return self._inner.chat.completions.create(**kw)

    capped_m3 = CappedClient(m3_client, m3_budget)

    # load model + retriever
    model, tokenizer = load_sft_model(args.base_model, args.lora, load_4bit=not args.no_4bit)
    from modes.m2_think_search.m2_think_search import ToolRetriever
    retriever = ToolRetriever(args.emb, args.catalog)

    from pilot.simulate_modes import simulate_trajectories
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    simulate_trajectories(
        trajs, catalog, model=model, tokenizer=tokenizer,
        retriever=retriever, llm_client=capped_m3,
        skip_m3=False, output_path=args.out, verbose=True,
    )
    print(f"\n✅ Pilot xong. M3 dùng {m3_budget.used}/{args.max_m3} call.")
    print(f"   → python eval/oracle.py --in {args.out} --out eval/oracle_report.json")


if __name__ == "__main__":
    main()