"""Write eval/v6/kaggle/{retrain,v4adapter}/ (script + kernel-metadata.json) from kernel_template.py."""
import json
from pathlib import Path

H = Path(__file__).resolve().parent
tpl = (H / "kernel_template.py").read_text()
for job in ("retrain", "v4adapter", "v4adapter_v12"):
    d = H / job
    d.mkdir(exist_ok=True)
    (d / f"run_{job}.py").write_text(tpl.replace('JOB = "__JOB__"', f'JOB = "{job}"'))
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"ductaiphan/budgetswitch-v6-{job.replace('_', '-')}", "title": f"budgetswitch-v6-{job.replace('_', '-')}",
        "code_file": f"run_{job}.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": ["ductaiphan/budgetswitch-v6-bundle", "ductaiphan/budgetswitch-v5-bundle"],
        "competition_sources": [], "kernel_sources": []}, indent=2))
print("ok")
