"""Copy the code + v1.1 data needed on Kaggle into <dest>/repo and write dataset metadata.
Usage: venv/bin/python eval/v6/kaggle/build_bundle.py <dest_dir>"""
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
dest = Path(sys.argv[1]).resolve()
repo = dest / "repo"
if dest.exists():
    shutil.rmtree(dest)
repo.mkdir(parents=True)
globs = ["data/*.py", "data/vitool/*.py", "data/vitool/*.jsonl", "data/vitool/*.json", "data/vitool/*.npy",
         "env/**/*.py", "modes/**/*.py", "pilot/*.py", "verifiers/**/*.py",
         "eval/v5/*.py", "eval/v5/journal/backbones/train_eval_backbone.py",
         "eval/v6/*.py", "eval/v6/kaggle/V6_BUNDLE", "requirements.txt"]
n = 0
for g in globs:
    for p in ROOT.glob(g):
        if "__pycache__" in p.parts:
            continue
        q = repo / p.relative_to(ROOT)
        q.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, q)
        n += 1
secret = re.compile(rb"gsk_[A-Za-z0-9]{10,}|sk-[A-Za-z0-9]{20,}|GROQ_API_KEY\s*=\s*\S{10,}")
hits = [str(p) for p in repo.rglob("*") if p.is_file() and secret.search(p.read_bytes())]
assert not hits, f"secret-like strings in {hits}"
(dest / "dataset-metadata.json").write_text(json.dumps({
    "title": "budgetswitch-v6-bundle", "id": "ductaiphan/budgetswitch-v6-bundle",
    "licenses": [{"name": "CC0-1.0"}]}, indent=2))
print(f"{n} files -> {repo}; no secrets")
