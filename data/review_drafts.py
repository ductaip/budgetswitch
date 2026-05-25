"""
Draft Review Helper — efficient human review of generated trajectories.

The LLM generates drafts; this tool helps the team review them quickly and
mark each as approved / needs-fix / rejected. Approved trajectories become
the gold dataset.

============================================================
WHY HUMAN REVIEW IS MANDATORY:
The LLM drafts plausible-looking trajectories, but:
- expected_params may not match what the tool actually expects
- queries may be unnatural or ambiguous
- difficulty labels may be miscalibrated
Human review IS the ground-truth step. Do not skip it.
============================================================

Place file at: data/review_drafts.py

Usage (interactive CLI review):
    python data/review_drafts.py --in data/vitool/drafts_banking.jsonl \
                                 --out data/vitool/gold_banking.jsonl

Or generate an HTML review sheet for offline review:
    python data/review_drafts.py --in data/vitool/drafts_banking.jsonl --html review_banking.html
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.vitool.schema import (
    Trajectory, load_trajectories, save_trajectories, validate_trajectory,
)


def load_catalog(path="data/vitool/tool_catalog.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# Validation stats (run before review to triage)
# ============================================================

def triage_drafts(drafts: list[Trajectory], catalog: dict) -> dict:
    """
    Categorize drafts to prioritize review effort.
    
    Returns dict with: valid (no errors), flagged (validation errors),
    and per-difficulty / per-step-count breakdown.
    """
    valid, flagged = [], []
    for t in drafts:
        errors = validate_trajectory(t, catalog)
        if errors:
            flagged.append((t, errors))
        else:
            valid.append(t)

    from collections import Counter
    diff_counts = Counter(
        s.difficulty.value for t in drafts for s in t.steps
    )
    step_counts = Counter(t.num_steps for t in drafts)

    return {
        "total": len(drafts),
        "valid": valid,
        "flagged": flagged,
        "n_valid": len(valid),
        "n_flagged": len(flagged),
        "difficulty_distribution": dict(diff_counts),
        "step_count_distribution": dict(step_counts),
    }


def print_triage_report(triage: dict):
    print(f"\n{'='*55}")
    print(f"DRAFT TRIAGE REPORT")
    print(f"{'='*55}")
    print(f"Total drafts: {triage['total']}")
    print(f"  Valid (passes schema check): {triage['n_valid']}")
    print(f"  Flagged (validation errors): {triage['n_flagged']}")
    print(f"\nDifficulty distribution: {triage['difficulty_distribution']}")
    print(f"Step-count distribution: {triage['step_count_distribution']}")

    if triage["flagged"]:
        print(f"\n--- FLAGGED DRAFTS (review these first) ---")
        for t, errors in triage["flagged"][:10]:
            print(f"  {t.traj_id}: {errors[0]}")
        if len(triage["flagged"]) > 10:
            print(f"  ... and {len(triage['flagged']) - 10} more")
    print(f"{'='*55}\n")


# ============================================================
# HTML review sheet (for offline / collaborative review)
# ============================================================

def generate_html_review(drafts: list[Trajectory], catalog: dict, out_path: str):
    """Generate a self-contained HTML sheet for reviewing drafts."""
    rows = []
    for t in drafts:
        errors = validate_trajectory(t, catalog)
        status_color = "#ffe0e0" if errors else "#e0ffe0"
        error_html = (
            f'<div style="color:#c00;font-size:0.85em">⚠️ {"; ".join(errors)}</div>'
            if errors else ""
        )
        steps_html = ""
        for i, s in enumerate(t.steps):
            steps_html += (
                f'<div style="margin-left:1em;font-family:monospace;font-size:0.85em">'
                f'[{i}] <b>{s.expected_tool}</b>('
                f'{json.dumps(s.expected_params, ensure_ascii=False)}) '
                f'<span style="color:#888">[{s.difficulty.value}]</span></div>'
            )
        rows.append(
            f'<tr style="background:{status_color}">'
            f'<td style="padding:8px;vertical-align:top">{t.traj_id}</td>'
            f'<td style="padding:8px;vertical-align:top"><b>{t.query}</b>{error_html}'
            f'{steps_html}</td>'
            f'<td style="padding:8px;vertical-align:top">'
            f'<label><input type="checkbox" class="approve" data-id="{t.traj_id}"> OK</label><br>'
            f'<label><input type="checkbox" class="fix" data-id="{t.traj_id}"> Fix</label><br>'
            f'<label><input type="checkbox" class="reject" data-id="{t.traj_id}"> Reject</label>'
            f'</td></tr>'
        )

    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
<meta charset="utf-8">
<title>BudgetSwitch Draft Review</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1em; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th {{ background: #333; color: white; padding: 8px; text-align: left; position: sticky; top: 0; }}
  td {{ border-bottom: 1px solid #ddd; }}
  .summary {{ background: #f5f5f5; padding: 1em; border-radius: 8px; margin-bottom: 1em; }}
  button {{ padding: 8px 16px; font-size: 1em; cursor: pointer; }}
</style>
</head>
<body>
<h1>BudgetSwitch Draft Review</h1>
<div class="summary">
  <b>{len(drafts)} drafts</b> to review.
  Green = passes schema check, Red = has validation errors (review first).<br>
  Check OK / Fix / Reject for each, then click "Export decisions" to download a JSON
  you can feed back into the pipeline.
</div>
<button onclick="exportDecisions()">Export decisions (JSON)</button>
<table>
<thead><tr><th>ID</th><th>Query &amp; Steps</th><th>Decision</th></tr></thead>
<tbody>
{"".join(rows)}
</tbody>
</table>
<script>
function exportDecisions() {{
  const decisions = {{approve: [], fix: [], reject: []}};
  document.querySelectorAll('.approve:checked').forEach(c => decisions.approve.push(c.dataset.id));
  document.querySelectorAll('.fix:checked').forEach(c => decisions.fix.push(c.dataset.id));
  document.querySelectorAll('.reject:checked').forEach(c => decisions.reject.push(c.dataset.id));
  const blob = new Blob([JSON.stringify(decisions, null, 2)], {{type: 'application/json'}});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'review_decisions.json';
  a.click();
}}
</script>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ HTML review sheet saved to {out_path}")
    print(f"   Open in browser, review, check boxes, click 'Export decisions'.")


# ============================================================
# Apply review decisions
# ============================================================

def apply_decisions(
    drafts: list[Trajectory],
    decisions_path: str,
    gold_out: str,
) -> None:
    """
    Apply a review_decisions.json (from HTML export) to produce gold dataset.
    
    Only 'approve' trajectories go to gold. 'fix' are saved separately for
    manual editing. 'reject' are dropped.
    """
    with open(decisions_path, encoding="utf-8") as f:
        decisions = json.load(f)

    by_id = {t.traj_id: t for t in drafts}
    approved = [by_id[i] for i in decisions.get("approve", []) if i in by_id]
    needs_fix = [by_id[i] for i in decisions.get("fix", []) if i in by_id]

    save_trajectories(approved, gold_out)
    if needs_fix:
        fix_path = gold_out.replace(".jsonl", "_needs_fix.jsonl")
        save_trajectories(needs_fix, fix_path)
        print(f"  {len(needs_fix)} trajectories need fixing -> {fix_path}")

    print(f"✅ {len(approved)} approved trajectories -> {gold_out}")
    print(f"   Rejected: {len(decisions.get('reject', []))}")


def main():
    parser = argparse.ArgumentParser(description="Review draft trajectories")
    parser.add_argument("--in", dest="input", required=True, help="drafts JSONL")
    parser.add_argument("--html", help="generate HTML review sheet at this path")
    parser.add_argument("--decisions", help="apply review_decisions.json")
    parser.add_argument("--out", help="gold output JSONL (with --decisions)")
    parser.add_argument("--catalog", default="data/vitool/tool_catalog.json")
    args = parser.parse_args()

    catalog = load_catalog(args.catalog)
    drafts = load_trajectories(args.input)

    # Always show triage
    triage = triage_drafts(drafts, catalog)
    print_triage_report(triage)

    if args.html:
        generate_html_review(drafts, catalog, args.html)
    elif args.decisions and args.out:
        apply_decisions(drafts, args.decisions, args.out)
    else:
        print("Tip: use --html review.html to generate a review sheet,")
        print("     or --decisions review_decisions.json --out gold.jsonl to apply.")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Self-test with synthetic drafts
        from data.vitool.schema import Trajectory, Step, Language, Difficulty
        catalog = load_catalog()
        drafts = [
            Trajectory(traj_id="banking_0001", query="Kiểm tra số dư 12345678",
                       language=Language.VI, source="groq_draft", domain="banking",
                       steps=[Step("check_balance", {"account_number": "12345678"},
                                   difficulty=Difficulty.EASY)]),
            Trajectory(traj_id="banking_0002", query="Chuyển tiền lung tung",
                       language=Language.VI, source="groq_draft", domain="banking",
                       steps=[Step("nonexistent_tool", {})]),  # will be flagged
        ]
        triage = triage_drafts(drafts, catalog)
        print_triage_report(triage)
        generate_html_review(drafts, catalog, "/tmp/test_review.html")
        print("\n✅ Review helper self-test passed.")
    else:
        main()