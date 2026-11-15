# Kaggle kernel: BudgetSwitch v5 — task=tf. Generated from eval/v5/kaggle/.
import glob, os, shutil, subprocess, sys
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "peft", "bitsandbytes",
                "sentence-transformers", "jsonschema"], check=True)
hit = glob.glob("/kaggle/input/**/eval/v5/common_v5.py", recursive=True)
assert hit, "bundle not mounted"
repo = os.path.dirname(os.path.dirname(os.path.dirname(hit[0])))
lora = glob.glob("/kaggle/input/**/tool_use_lora_v2/adapter_model.safetensors", recursive=True)
lora = os.path.dirname(lora[0])
shutil.copytree(repo, "/kaggle/working/bs", dirs_exist_ok=True)
os.chdir("/kaggle/working/bs")
r = subprocess.run([sys.executable, "-u", "eval/v5/run_local_modes_v5.py", "--task", "tf",
                    "--lora", lora, "--out", "/kaggle/working/local_tf_v5.jsonl"])
shutil.rmtree("/kaggle/working/bs", ignore_errors=True)
sys.exit(r.returncode)
