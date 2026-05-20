#!/bin/bash
# verify_no_oracle_leak.sh
# Pre-commit hook to enforce CRITICAL RULE 1.
# 
# Checks that no Python CODE (excluding docstrings and comments) references
# oracle_mode, expected_mode, correct_mode, or target_mode.
#
# Usage:
#   bash scripts/verify_no_oracle_leak.sh
#
# Exit code: 0 if clean, 1 if violations found.

set -e

cd "$(dirname "$0")/.."

PATTERNS="oracle_mode|expected_mode|correct_mode|target_mode"
SEARCH_DIRS="training verifiers modes baselines eval"

echo "=========================================="
echo "BudgetSwitch CRITICAL RULE 1 Verification"
echo "=========================================="
echo "Searching for forbidden mode-supervision patterns..."
echo "Patterns: $PATTERNS"
echo "Directories: $SEARCH_DIRS"
echo ""

# Find matches in .py files, excluding lines that are comments/docstrings
violations=0

for dir in $SEARCH_DIRS; do
    if [ ! -d "$dir" ]; then
        continue
    fi
    
    # Find all matches
    matches=$(grep -rn -E "$PATTERNS" --include="*.py" "$dir" 2>/dev/null || true)
    
    if [ -z "$matches" ]; then
        continue
    fi
    
    # Filter out lines that are clearly in comments or docstrings
    # (lines whose first non-whitespace char is # or that are inside """...""")
    while IFS= read -r line; do
        file=$(echo "$line" | cut -d: -f1)
        line_num=$(echo "$line" | cut -d: -f2)
        content=$(echo "$line" | cut -d: -f3-)
        
        # Strip leading whitespace
        trimmed=$(echo "$content" | sed 's/^[[:space:]]*//')
        
        # Skip if line starts with #
        if [[ "$trimmed" == \#* ]]; then
            continue
        fi
        
        # Skip if line is inside a docstring (heuristic: line has """ before the match)
        # More robust check: use AST. For now, check if the file's matching line
        # is wrapped in """ ... """ block by counting """ before this line.
        triple_quote_count=$(head -n "$line_num" "$file" | grep -c '"""' || true)
        if (( triple_quote_count % 2 == 1 )); then
            # Inside a docstring
            continue
        fi
        
        # This is a real code reference — FLAG IT
        echo "❌ VIOLATION: $file:$line_num"
        echo "   $content"
        violations=$((violations + 1))
    done <<< "$matches"
done

echo ""
if [ "$violations" -eq 0 ]; then
    echo "✅ PASSED: No oracle_mode references in code."
    echo "   (Matches in docstrings/comments are intentional warnings.)"
    exit 0
else
    echo "❌ FAILED: $violations code-level violations of CRITICAL RULE 1."
    echo ""
    echo "Reward functions and verifiers MUST NOT reference per-step mode labels."
    echo "See CRITICAL_RULES.md for details and acceptable alternatives."
    exit 1
fi
