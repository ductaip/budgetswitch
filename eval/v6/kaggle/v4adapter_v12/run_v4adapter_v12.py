"""
Kaggle kernel (v6). JOB is substituted by eval/v6/kaggle/make_kernels.py:

  retrain   Step 3 + 4: retrain the 3B LoRA on the v1.1 TRAIN split with the v4 recipe
            (same as eval/v5/journal/backbones/train_eval_backbone.py: Unsloth 4-bit,
            r=16/alpha=32 on 7 projections, 3 epochs, bs 4 x acc 4, lr 2e-4 cosine,
            seed 42, M0-style + M1-think examples with gold-history replay),
            then M0/M1/M2 teacher-forced + M0/M1 rollout on the v1.1 EVAL split.
  v4adapter Step 4: the v4 adapter (tool_use_lora_v2, from the v5 bundle) on v1.1 eval
            queries, teacher-forced + rollout (isolates "better queries" from "retrained").

Mounts: ductaiphan/budgetswitch-v6-bundle (repo with v1.1 data + v5/v6 code) and
ductaiphan/budgetswitch-v5-bundle (only for the v4 adapter).
"""
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys

JOB = "v4adapter_v12"  # retrain | v4adapter | v4adapter_v12
OUT = "/kaggle/working"
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout, flush=True)

# eval/v6/analyze_v6.py exists only in the v6 bundle (the v5 bundle is mounted for the v4 adapter)
hit = glob.glob("/kaggle/input/**/eval/v6/analyze_v6.py", recursive=True)
assert hit, "v6 bundle not mounted"
repo_src = os.path.dirname(os.path.dirname(os.path.dirname(hit[0])))
assert os.path.exists(f"{repo_src}/data/vitool/pilot_eval_{'v1.2' if JOB.endswith('_v12') else 'v1.1'}.jsonl"), repo_src
REPO = f"{OUT}/bs"
shutil.copytree(repo_src, REPO, dirs_exist_ok=True)
os.chdir(REPO)
VER = "v1.2" if JOB.endswith("_v12") else "v1.1"
EVAL = f"{REPO}/data/vitool/pilot_eval_{VER}.jsonl"
TRAIN_FILE = "data/vitool/sft_train_v1.1.jsonl"
print("eval sha256", hashlib.sha256(open(EVAL, "rb").read()).hexdigest(), flush=True)

if JOB == "retrain":
    pin = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "unsloth==2026.6.2"],
                         capture_output=True, text=True)
    note = "pinned 2026.6.2"
    if pin.returncode != 0:
        print("pin failed:", pin.stderr[-1500:], flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "unsloth"], check=True)
        note = "latest (pin to 2026.6.2 failed)"
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft", "bitsandbytes",
                    "sentence-transformers", "jsonschema", "trl", "datasets"], check=True)
    src = open("eval/v5/journal/backbones/train_eval_backbone.py").read()
    TRAIN = src.split("TRAIN = r'''", 1)[1].split("'''", 1)[0]
    TRAIN = TRAIN.replace('load_trajectories("data/vitool/sft_train.jsonl")',
                          f'load_trajectories("{TRAIN_FILE}")')
    assert TRAIN_FILE in TRAIN
    script = TRAIN % {"repo": REPO, "size": "3B", "out": OUT, "tag": "v6"}
    open(f"{OUT}/train_v6.py", "w").write(script)
    r = subprocess.run([sys.executable, "-u", f"{OUT}/train_v6.py"],
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES="0"))
    assert r.returncode == 0, "training failed"
    lora = f"{OUT}/tool_use_lora_v6"
    tag = "v6adapter"
    json.dump({"unsloth_install": note, "train_file": TRAIN_FILE,
               "train_sha256": hashlib.sha256(open(TRAIN_FILE, "rb").read()).hexdigest()},
              open(f"{OUT}/install_v6.json", "w"))
else:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft", "bitsandbytes",
                    "sentence-transformers", "jsonschema"], check=True)
    hit = glob.glob("/kaggle/input/**/tool_use_lora_v2/adapter_model.safetensors", recursive=True)
    assert hit, "v4 adapter not mounted"
    lora = os.path.dirname(hit[0])
    tag = "v4adapter"

rc = 0
for task in ("tf", "rollout"):
    r = subprocess.run([sys.executable, "-u", "eval/v5/run_local_modes_v5.py", "--task", task,
                        "--modes", "M0,M1,M2", "--data", EVAL, "--lora", lora,
                        "--out", f"{OUT}/local_{task}_{tag}_{VER.replace('.', '')}.jsonl"])
    rc = rc or r.returncode
shutil.rmtree(REPO, ignore_errors=True)
shutil.rmtree(f"{OUT}/sft_ckpt", ignore_errors=True)
sys.exit(rc)
