# Public release checklist — github.com/ductaip/budgetswitch (prepared 2026-11-13, NOT pushed)

## 1. BLOCKER: a real Groq API key is in git history

- File: `notebooks/week2-budgetswitch.ipynb`, hard-coded as the default of
  `os.environ.get("GROQ_API_KEY", "gsk_Uj…")` (56 chars; not one of the keys in
  the current `.env`).
- Commit `1f595d9` (2026-05-27), which **is on `origin/main`**. The GitHub repo
  currently answers 404 to anonymous requests (private), so it has not been
  public, but anyone with access to the repo can read it.
- gitleaks 8.24.2 did **not** flag it (found by a manual `gsk_` scan).

Before making the repo public:
1. **Revoke the key** in the Groq console (whichever account issued it).
2. Remove it from the working tree (replace the default with `""` or drop the
   cell output/source) and **from history**, e.g.
   ```bash
   pip install git-filter-repo
   printf 'regex:gsk_[A-Za-z0-9]{20,}==>REDACTED_GROQ_KEY\n' > /tmp/replacements.txt
   git filter-repo --replace-text /tmp/replacements.txt     # rewrites all commits
   git push --force origin main                              # rewrites the remote
   ```
   This rewrites history (new commit hashes, force push) — your decision; not
   done here. Alternative: publish a fresh repository with a single squashed
   commit of the cleaned tree.
3. Never `git push --all` / `--mirror`: local refs `refs/agents/…/checkpoints/*`
   (agent snapshots, also containing the notebook) must not be pushed.

## 2. Scan results

| scan | result |
|---|---|
| gitleaks, full git history | 2 findings, both false positives (empty `GROQ_API_KEY=` in `.env.example`; the name in a docstring) → recorded in `.gitleaksignore`; rescan: no leaks |
| gitleaks, files that would be published (tracked + untracked, not ignored) | same 2 false positives only |
| manual regex (Groq `gsk_`, HF `hf_`, OpenAI/Anthropic `sk-`, GitHub `ghp_`, AWS, Kaggle key) over all commits and the publishable tree | **1 real key: the notebook above**; nothing else |
| `.env` (4 real keys) | git-ignored, never committed |
| `~/.kaggle/kaggle.json` | outside the repo |

## 3. What changed locally (uncommitted)

- `README.md` rewritten for v1.2 (history v1 → v1.1 → v1.2, v1.2 numbers).
- `DATASHEET.md` new (the old README referenced it but it did not exist).
- `.gitignore`: adapters (`**/tool_use_lora*/`), kernel outputs, caches,
  `*.bak`, `*.zip`, `eval/v6/paper_pack/`.
- `requirements.txt`: + scikit-learn, scipy, adjustText.
- `.gitleaksignore` (reviewed false positives).
- `data/build_dataset.py` (G3 wired in, `--gate-report`, Gate 1.5 extended).

## 4. Files to add (suggested)

```
data/grounding_gate.py  data/build_dataset.py  tests/test_grounding_gate.py
data/vitool/vitool_v1.2.jsonl  data/vitool/sft_train_v1.2.jsonl  data/vitool/pilot_eval_v1.2.jsonl
data/vitool/vitool_v1.1.jsonl  data/vitool/sft_train_v1.1.jsonl  data/vitool/pilot_eval_v1.1.jsonl
DATASHEET.md  README.md  requirements.txt  .gitignore  .gitleaksignore  .env.example
eval/v5/  eval/v6/          (ignored: adapters, caches, backups, zips)
```
Consider NOT publishing: `CLAUDE_CODE_TASK_v5.md` (internal brief),
`paper/vitool_camera_ready.tex` until after the camera-ready deadline, and the
already-tracked `docs/ADVISOR_EMAIL_TEMPLATES.md`, `docs/W*_RETROSPECTIVE.md`,
`docs/BudgetSwitch_Proposal.*` (internal planning notes).

## 5. Verified

- `pytest tests/test_grounding_gate.py`: 26 passed; `build_dataset.py --selftest`: pass.
- `build_dataset.py --gate-report` on v1.2: 373/373 pass every gate.
- Publishable tree ≈ 23 MB.
