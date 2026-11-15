"""
P1 item 9 — second/third backbone (journal only; not part of the main REPORT).

Same LoRA recipe + train split as the v4 adapter (Kaggle kernel
ductaiphan/vitool-pilot, cells 10-12): Unsloth 4-bit base, LoRA r=16/alpha=32
on all 7 projections, SFT 3 epochs, bs 4 x grad-acc 4, lr 2e-4 cosine,
seed 42, examples = M0-style (random mode-token prefix) + M1-style think,
built from data/vitool/sft_train.jsonl with gold-history replay.
Then the v5 teacher-forced runner (M0/M1/M2, full logging) on the 284 held-out
steps, with the same base model in nf4 + the new adapter (as for the 3B run).

Kaggle kernel script; SIZE is substituted per kernel ("1.5B" / "7B").
"""
import glob
import json
import os
import shutil
import subprocess
import sys

SIZE = "1.5B"
OUT = "/kaggle/working"

print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout, flush=True)

# --- deps: pin Unsloth to the version used for the v4 adapter; fall back if it fails
pin = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "unsloth==2026.6.2"],
                     capture_output=True, text=True)
unsloth_note = "pinned 2026.6.2"
if pin.returncode != 0:
    print("pin failed:", pin.stderr[-1500:], flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "unsloth"], check=True)
    unsloth_note = "latest (pin to 2026.6.2 failed)"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft", "bitsandbytes",
                "sentence-transformers", "jsonschema", "trl", "datasets"], check=True)

hit = glob.glob("/kaggle/input/**/eval/v5/common_v5.py", recursive=True)
assert hit, "bundle not mounted"
repo_src = os.path.dirname(os.path.dirname(os.path.dirname(hit[0])))
REPO = f"{OUT}/bs"
shutil.copytree(repo_src, REPO, dirs_exist_ok=True)
os.chdir(REPO)

TRAIN = r'''
import json, os, random, sys, time
sys.path.insert(0, "%(repo)s")
from unsloth import FastLanguageModel
import torch
SIZE = "%(size)s"
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=f"unsloth/Qwen2.5-{SIZE}-Instruct",
    max_seq_length=2048, load_in_4bit=True, fast_inference=False, dtype=None)
model = FastLanguageModel.get_peft_model(model, r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=32, lora_dropout=0, bias="none",
    use_gradient_checkpointing="unsloth", random_state=42)

random.seed(42)
from data.vitool.schema import load_trajectories
from modes.m0_fast.m0_fast import _build_m0_prompt
from modes.m1_think.m1_think import _build_m1_prompt
from pilot.simulate_modes import _replay_history
CATALOG = {k:v for k,v in json.load(open("data/vitool/tool_catalog.json")).items() if not k.startswith("_")}
trajs = load_trajectories("data/vitool/sft_train.jsonl")
MODES = ["M0","M1","M2","M3"]

def make_think(gt, hist):
    parts = [f"Cần dùng tool '{gt.expected_tool}'."]
    blob = json.dumps([o for _,o in hist], ensure_ascii=False, default=str)
    deps = [k for k,v in gt.expected_params.items() if isinstance(v,str) and len(v)>=6 and v in blob]
    if deps:
        parts.append("Giá trị "+", ".join(f"{k}={gt.expected_params[k]}" for k in deps)+" lấy từ kết quả bước trước.")
    else:
        parts.append("Tham số lấy trực tiếp từ yêu cầu người dùng.")
    return " ".join(parts)

def build_pair(traj, i):
    gt = traj.steps[i]; hist = _replay_history(traj, i)
    gold = json.dumps({"tool_name":gt.expected_tool,"params":gt.expected_params}, ensure_ascii=False)
    return [
      {"prompt":_build_m0_prompt(traj.query,hist,CATALOG), "completion":f"{random.choice(MODES)}\n{gold}"},
      {"prompt":_build_m1_prompt(traj.query,hist,CATALOG), "completion":f"{make_think(gt,hist)}</think>\n{gold}"},
    ]

sft_examples = [e for t in trajs for i in range(t.num_steps) for e in build_pair(t,i)]
print(f"{len(sft_examples)} SFT examples", flush=True)
from datasets import Dataset
hf = Dataset.from_list(sft_examples).map(lambda ex: {"text": ex["prompt"]+ex["completion"]+tokenizer.eos_token})
split = hf.train_test_split(test_size=0.1, seed=42)
train_ds, val_ds = split["train"], split["test"]

import inspect
from trl import SFTTrainer, SFTConfig
kw = dict(output_dir="%(out)s/sft_ckpt", num_train_epochs=3,
    per_device_train_batch_size=4, gradient_accumulation_steps=4, warmup_steps=10,
    learning_rate=2e-4, fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(), logging_steps=10,
    eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
    optim="adamw_torch", weight_decay=0.01, lr_scheduler_type="cosine",
    report_to="none", dataset_text_field="text", seed=42)
p = inspect.signature(SFTConfig.__init__).parameters
kw["max_seq_length" if "max_seq_length" in p else "max_length"] = 1024
if "dataset_num_proc" in p: kw["dataset_num_proc"] = 2
if "packing" in p: kw["packing"] = False
cfg = SFTConfig(**kw)
tp = inspect.signature(SFTTrainer.__init__).parameters
tkw = {"tokenizer" if "tokenizer" in tp else "processing_class": tokenizer}
trainer = SFTTrainer(model=model, train_dataset=train_ds, eval_dataset=val_ds, args=cfg, **tkw)
t0 = time.time()
trainer.train()
model.save_pretrained("%(out)s/tool_use_lora_%(tag)s")
tokenizer.save_pretrained("%(out)s/tool_use_lora_%(tag)s")
import importlib.metadata as md
json.dump({"size": SIZE, "n_examples": len(sft_examples), "train_minutes": (time.time()-t0)/60,
           "log_history": trainer.state.log_history,
           "versions": {k: md.version(k) for k in ["unsloth","trl","transformers","peft","torch","datasets"]}},
          open("%(out)s/train_%(tag)s.json","w"), indent=1, default=str)
print("TRAIN DONE", flush=True)
'''
tag = SIZE.replace(".", "p")
script = TRAIN % {"repo": REPO, "size": SIZE, "out": OUT, "tag": tag}
open(f"{OUT}/train_{tag}.py", "w").write(script)
env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")  # Unsloth trains on one GPU
r = subprocess.run([sys.executable, "-u", f"{OUT}/train_{tag}.py"], env=env)
assert r.returncode == 0, "training failed"
json.dump({"unsloth_install": unsloth_note}, open(f"{OUT}/install_{tag}.json", "w"))

r = subprocess.run([sys.executable, "-u", "eval/v5/run_local_modes_v5.py", "--task", "tf",
                    "--modes", "M0,M1,M2", "--base-model", f"Qwen/Qwen2.5-{SIZE}-Instruct",
                    "--lora", f"{OUT}/tool_use_lora_{tag}",
                    "--out", f"{OUT}/local_tf_{tag}_v5.jsonl"])
shutil.rmtree(REPO, ignore_errors=True)
shutil.rmtree(f"{OUT}/sft_ckpt", ignore_errors=True)
sys.exit(r.returncode)
