#!/usr/bin/env bash
# Rebuild REPORT_v6.md = narrative + generated tables (rerun after M3 jobs finish).
set -e
venv/bin/python eval/v6/analyze_v6.py > /dev/null
venv/bin/python eval/v6/routers_v6.py > /dev/null 2>&1
venv/bin/python eval/v6/report_tables_v6.py > /dev/null
cat eval/v6/REPORT_v6_narrative.md eval/v6/report_tables_v6.md > eval/v6/REPORT_v6.md
echo "REPORT_v6.md rebuilt"
