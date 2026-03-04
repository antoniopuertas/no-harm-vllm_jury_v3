# GitHub Prep Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Clean the repo for public GitHub publication: remove ephemeral artifacts, junk config backups, obsolete scripts, and fix portability inconsistencies.

**Architecture:** Pure file operations — delete, gitignore, and edit in place. No source code changes. No tests needed (these are config/script fixes).

**Tech Stack:** bash, git

---

### Task 1: Add .gitignore

**Files:**
- Create: `.gitignore`

**Step 1: Create `.gitignore`**

```
# Python
__pycache__/
*.py[cod]
*.egg-info/
.venv/
.env

# Run artifacts
logs/*.log
data/results/
data/datasets/

# OS
.DS_Store
```

**Step 2: Verify it covers the right files**

Run: `git check-ignore -v logs/full_eval_20260220_143120.log data/results/vllm/full_runs/Jury_v3/`
Expected: both lines show `.gitignore` as the source

**Step 3: Commit**

```bash
git add .gitignore
git commit -m "chore: add .gitignore"
```

---

### Task 2: Remove data/results/ contents, keep directory skeleton

**Files:**
- Delete: everything under `data/results/` except `data/results/README.md`
- Create: `data/results/.gitkeep`

**Step 1: Remove result files**

```bash
rm -rf data/results/vllm data/results/parse_logs
```

**Step 2: Add .gitkeep so the directory is tracked**

```bash
touch data/results/.gitkeep
```

**Step 3: Commit**

```bash
git add -A data/results/
git commit -m "chore: remove computed results (gitignored)"
```

---

### Task 3: Remove logs/ contents, keep directory

**Files:**
- Delete: all `logs/*.log`
- Create: `logs/.gitkeep`

**Step 1: Remove log files**

```bash
rm logs/*.log
touch logs/.gitkeep
```

**Step 2: Commit**

```bash
git add -A logs/
git commit -m "chore: remove run logs (gitignored)"
```

---

### Task 4: Remove docs/plans/

**Files:**
- Delete: `docs/plans/` directory (entire)

**Step 1: Remove the directory**

```bash
rm -rf docs/plans/
```

**Step 2: Commit**

```bash
git add -A docs/plans/
git commit -m "chore: remove internal planning docs"
```

---

### Task 5: Remove backup config files

**Files:**
- Delete: `config/vllm_jury_config_v2.3.yaml`
- Delete: `config/vllm_jury_config_v3.0_backup.yaml`
- Delete: `config/vllm_jury_config_with_ministral_backup.yaml`

These are superseded or contain paths from a different machine (`/nfs/staging/puertao/noharm/`).

**Step 1: Delete them**

```bash
rm config/vllm_jury_config_v2.3.yaml \
   config/vllm_jury_config_v3.0_backup.yaml \
   config/vllm_jury_config_with_ministral_backup.yaml
```

**Step 2: Commit**

```bash
git add -A config/
git commit -m "chore: remove obsolete config backups"
```

---

### Task 6: Remove junk scripts

**Files:**
- Delete: `scripts/WORKING_COMMANDS.txt`
- Delete: `scripts/run_1000_instances.txt`
- Delete: `scripts/QUICK_START_DUAL_GPU.txt`
- Delete: `scripts/STOP_AND_FIX.sh`
- Delete: `scripts/test_olmo_dual_gpu.py`
- Delete: `scripts/run_full_vllm_evaluation_v3.py`

Rationale:
- The three `.txt` files are session notes; the info is in the README.
- `STOP_AND_FIX.sh` is a one-off debugging workaround for a specific hang.
- `test_olmo_dual_gpu.py` is a one-off GPU sanity check.
- `run_full_vllm_evaluation_v3.py` is an older entry point; README points to `run_full_vllm_evaluation.py`.

**Step 1: Delete them**

```bash
rm scripts/WORKING_COMMANDS.txt \
   scripts/run_1000_instances.txt \
   scripts/QUICK_START_DUAL_GPU.txt \
   scripts/STOP_AND_FIX.sh \
   scripts/test_olmo_dual_gpu.py \
   scripts/run_full_vllm_evaluation_v3.py
```

**Step 2: Commit**

```bash
git add -A scripts/
git commit -m "chore: remove session notes and one-off debug scripts"
```

---

### Task 7: Remove stray log from tests/

**Files:**
- Delete: `tests/phase2_maira2_test5_20260301_181948.log`

**Step 1: Delete it**

```bash
rm tests/phase2_maira2_test5_20260301_181948.log
```

**Step 2: Commit**

```bash
git add -A tests/
git commit -m "chore: remove stray log from tests/"
```

---

### Task 8: Fix dead comment in config.yaml

**Files:**
- Modify: `config.yaml:34`

The comment `# Using the 5 actual models from README_Qwen2.5.md` references an archived file.

**Step 1: Remove the comment**

In `config.yaml`, delete line 34:
```
# Using the 5 actual models from README_Qwen2.5.md
```

**Step 2: Commit**

```bash
git add config.yaml
git commit -m "fix: remove dead reference to archived README_Qwen2.5.md"
```

---

### Task 9: Fix `--num_samples` → `--instances` in run_medmcqa_1000.sh

**Files:**
- Modify: `scripts/run_medmcqa_1000.sh`

The script calls `--num_samples 1000` but `run_full_vllm_evaluation.py` uses `--instances`.

**Step 1: Fix the flag**

In `scripts/run_medmcqa_1000.sh`, change:
```bash
    --num_samples 1000 \
```
to:
```bash
    --instances 1000 \
```

**Step 2: Commit**

```bash
git add scripts/run_medmcqa_1000.sh
git commit -m "fix: use --instances flag (was --num_samples, which doesn't exist)"
```

---

### Task 10: Fix hardcoded absolute paths in shell scripts

**Files:**
- Modify: `scripts/run_1000_dual_gpu.sh`
- Modify: `scripts/run_1000_dual_gpu_safe.sh`
- Modify: `scripts/run_medmcqa_1000.sh`

Replace the hardcoded `/home/puertao/llm/no-harm-vllm_jury_v3` prefix with a computed root.
Add these two lines near the top of each script (after `set -e`):

```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
```

Then replace all occurrences of `/home/puertao/llm/no-harm-vllm_jury_v3` with `$REPO_ROOT` in those files.

**`run_1000_dual_gpu.sh` — before:**
```bash
BASE_OUTPUT_DIR="/home/puertao/llm/no-harm-vllm_jury_v3/data/results/vllm"
CONFIG="/home/puertao/llm/no-harm-vllm_jury_v3/config/vllm_jury_config_dual_gpu.yaml"
```
**after:**
```bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
BASE_OUTPUT_DIR="$REPO_ROOT/data/results/vllm"
CONFIG="$REPO_ROOT/config/vllm_jury_config_dual_gpu.yaml"
```

Apply the same pattern to `run_1000_dual_gpu_safe.sh` and `run_medmcqa_1000.sh`.

**Step 2: Commit**

```bash
git add scripts/run_1000_dual_gpu.sh scripts/run_1000_dual_gpu_safe.sh scripts/run_medmcqa_1000.sh
git commit -m "fix: replace hardcoded absolute paths with repo-relative \$REPO_ROOT"
```
