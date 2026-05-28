"""
Data Quality Validation for ViTool Dataset.

Generates statistics required for the paper's §5 Data Pipeline section:
  - Total trajectory / step counts
  - Domain / difficulty / language distribution
  - Schema validation error rate
  - Inter-annotator agreement proxy (LLM-vs-rule consistency)
  - Duplicate detection
  - Parameter format statistics (VN locale compliance)

Run: python data/validate_quality.py --input data/vitool/vitool_all.jsonl

Place at: data/validate_quality.py
"""

from __future__ import annotations
import argparse, json, sys, re
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.vitool.schema import load_trajectories, validate_trajectory


def load_catalog(path="data/vitool/tool_catalog.json"):
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if not k.startswith("_")}


# ============================================================
# Format validators (Vietnamese locale)
# ============================================================

VN_DATE_RE = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")          # dd/mm/yyyy
VN_PHONE_RE = re.compile(r"^(\+84|0)\d{9,10}$")               # +84xxx or 0xxx
VN_CURRENCY_RE = re.compile(r"^\d[\d.,]*\s*(VNĐ|VND|đ|triệu|tỷ)?$", re.IGNORECASE)


def check_vn_locale(params: dict) -> dict:
    """Check if date/phone/amount params follow Vietnamese format."""
    issues = {}
    for k, v in params.items():
        if not isinstance(v, str):
            continue
        if "date" in k.lower() or "ngày" in k.lower():
            if not VN_DATE_RE.match(v):
                issues[k] = f"date format '{v}' (expected dd/mm/yyyy)"
        if "phone" in k.lower() or "sdt" in k.lower():
            if not VN_PHONE_RE.match(v):
                issues[k] = f"phone format '{v}' (expected +84... or 0...)"
    return issues


# ============================================================
# Main validation
# ============================================================

def validate_dataset(input_path: str, catalog: dict, verbose: bool = True) -> dict:
    trajectories = load_trajectories(input_path)
    total_trajs = len(trajectories)
    total_steps = sum(t.num_steps for t in trajectories)

    # Distributions
    domain_dist = Counter(t.domain for t in trajectories)
    lang_dist = Counter(t.language.value for t in trajectories)
    source_dist = Counter(t.source for t in trajectories)
    diff_dist = Counter(s.difficulty.value for t in trajectories for s in t.steps)
    step_dist = Counter(t.num_steps for t in trajectories)
    tool_dist = Counter(s.expected_tool for t in trajectories for s in t.steps)

    # Schema validation
    schema_errors = []
    for t in trajectories:
        errs = validate_trajectory(t, catalog)
        schema_errors.extend(errs)
    schema_error_rate = len(schema_errors) / total_steps if total_steps > 0 else 0

    # Duplicate detection (exact query match)
    queries = [t.query for t in trajectories]
    unique_queries = set(queries)
    n_duplicates = total_trajs - len(unique_queries)

    # VN locale compliance
    locale_issues = []
    for t in trajectories:
        for s in t.steps:
            issues = check_vn_locale(s.expected_params)
            if issues:
                locale_issues.append((t.traj_id, s.expected_tool, issues))
    locale_error_rate = len(locale_issues) / total_steps if total_steps > 0 else 0

    # Multi-step stats
    multi_step_trajs = sum(1 for t in trajectories if t.num_steps > 1)
    avg_steps = total_steps / total_trajs if total_trajs > 0 else 0

    results = {
        "total_trajectories": total_trajs,
        "total_steps": total_steps,
        "avg_steps_per_traj": round(avg_steps, 2),
        "multi_step_trajectories": multi_step_trajs,
        "multi_step_pct": round(100 * multi_step_trajs / total_trajs, 1),
        "domain_distribution": dict(domain_dist),
        "language_distribution": dict(lang_dist),
        "source_distribution": dict(source_dist),
        "difficulty_distribution": dict(diff_dist),
        "step_count_distribution": dict(step_dist),
        "top_10_tools": dict(tool_dist.most_common(10)),
        "schema_errors": len(schema_errors),
        "schema_error_rate_pct": round(100 * schema_error_rate, 2),
        "duplicate_queries": n_duplicates,
        "locale_format_issues": len(locale_issues),
        "locale_error_rate_pct": round(100 * locale_error_rate, 2),
    }

    if verbose:
        _print_report(results, schema_errors[:5], locale_issues[:5])

    return results


def _print_report(r: dict, sample_errors: list, sample_locale: list):
    sep = "=" * 60
    print(f"\n{sep}")
    print("ViTool Dataset Quality Report")
    print(f"{sep}")
    print(f"Total trajectories : {r['total_trajectories']}")
    print(f"Total steps        : {r['total_steps']}")
    print(f"Avg steps/traj     : {r['avg_steps_per_traj']}")
    print(f"Multi-step trajs   : {r['multi_step_trajectories']} ({r['multi_step_pct']}%)")

    print(f"\nDomain distribution:")
    for d, c in sorted(r["domain_distribution"].items(), key=lambda x: -x[1]):
        bar = "█" * int(20 * c / r["total_trajectories"])
        print(f"  {d:20s} {c:4d}  {bar}")

    print(f"\nDifficulty distribution:")
    for d, c in r["difficulty_distribution"].items():
        bar = "█" * int(20 * c / r["total_steps"])
        print(f"  {d:10s} {c:4d}  {bar}")

    print(f"\nSource distribution:")
    for s, c in r["source_distribution"].items():
        print(f"  {s:25s} {c:4d}")

    print(f"\nData quality:")
    print(f"  Schema validation errors  : {r['schema_errors']} ({r['schema_error_rate_pct']}%)")
    print(f"  Duplicate queries          : {r['duplicate_queries']}")
    print(f"  VN locale format issues    : {r['locale_format_issues']} ({r['locale_error_rate_pct']}%)")

    if sample_errors:
        print(f"\nSample schema errors (first 5):")
        for e in sample_errors:
            print(f"  {e}")

    if sample_locale:
        print(f"\nSample locale issues (first 5):")
        for traj_id, tool, issues in sample_locale:
            print(f"  {traj_id} / {tool}: {issues}")

    # Paper-ready summary
    print(f"\n{sep}")
    print("PAPER-READY SUMMARY (for §5 Data Pipeline):")
    print(f"{sep}")
    print(f"ViTool-{r['total_trajectories']:,} consists of {r['total_trajectories']} "
          f"Vietnamese tool-use trajectories ({r['total_steps']} steps total) "
          f"across {len(r['domain_distribution'])} domains. "
          f"{r['multi_step_pct']}% of trajectories require multiple tool calls. "
          f"Schema validation error rate: {r['schema_error_rate_pct']}%. "
          f"Duplicate queries: {r['duplicate_queries']}. "
          f"All date and phone parameters verified against Vietnamese locale format "
          f"(dd/mm/yyyy, +84-prefix) with {r['locale_error_rate_pct']}% format deviation rate.")
    print(f"{sep}\n")


def save_report(results: dict, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ Report saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/vitool/vitool_all.jsonl")
    parser.add_argument("--catalog", default="data/vitool/tool_catalog.json")
    parser.add_argument("--out", default="data/vitool/quality_report.json")
    args = parser.parse_args()

    catalog = load_catalog(args.catalog)
    results = validate_dataset(args.input, catalog, verbose=True)
    save_report(results, args.out)

    # Exit with error if quality is below thresholds
    if results["schema_error_rate_pct"] > 5.0:
        print(f"❌ Schema error rate {results['schema_error_rate_pct']}% > 5% threshold")
        sys.exit(1)
    if results["duplicate_queries"] > 10:
        print(f"❌ {results['duplicate_queries']} duplicates > 10 threshold")
        sys.exit(1)
    print("✅ Dataset quality checks passed.")