# Design: H100 / GB10 Result Differentiation

**Date:** 2026-04-07  
**Status:** Approved

---

## Problem

`harm_dimensions_v2` results from two different hardware platforms (H100 and GB10 Blackwell) were mixed in the same output directory, making it impossible to compare or reproduce runs per hardware.

---

## Goal

Route evaluation results to hardware-specific subdirectories and move existing GB10 results into place, with no risk of accidentally writing to the wrong location.

---

## Directory Structure

```
data/results/vllm/harm_dimensions_v2/
├── GB10/          ← existing results moved here via git mv
│   ├── medmcqa_consolidated.json
│   ├── medmcqa_full_results/
│   ├── medqa_consolidated.json
│   ├── medqa_full_results/
│   ├── pubmedqa_consolidated.json
│   └── pubmedqa_full_results/
└── H100/          ← new H100 runs output here
    └── (populated on first run)
```

---

## Script Changes

Scripts affected:
- `scripts/run_harm_v2_sequential.sh`
- `scripts/run_harm_v2_1000.sh`
- `scripts/run_medqa_medmcqa.sh`

Each script gains a `--gpu <H100|GB10>` flag:
- **Required** — no default, script exits with usage error if omitted.
- Sets `OUTPUT_DIR` to `$REPO_ROOT/data/results/vllm/harm_dimensions_v2/$GPU_LABEL`.
- `--gpu GB10` auto-selects `config/vllm_jury_config_gb10.yaml`.
- `--gpu H100` uses `config/vllm_jury_config.yaml`.
- A `--config` flag can still override config selection explicitly.

Example usage:
```bash
bash scripts/run_harm_v2_sequential.sh --gpu H100
bash scripts/run_harm_v2_sequential.sh --gpu GB10
```

---

## .gitignore Changes

Current `data/` blanket-ignores everything. Update to un-ignore the two hardware subdirs:

```
data/
!data/results/
!data/results/vllm/
!data/results/vllm/harm_dimensions_v2/
!data/results/vllm/harm_dimensions_v2/GB10/
!data/results/vllm/harm_dimensions_v2/GB10/**
!data/results/vllm/harm_dimensions_v2/H100/
!data/results/vllm/harm_dimensions_v2/H100/**
```

---

## Migration

- `git mv data/results/vllm/harm_dimensions_v2/{medmcqa,medqa,pubmedqa}* data/results/vllm/harm_dimensions_v2/GB10/` for all tracked files.
- `git mv data/results/vllm/harm_dimensions_v2/Jury_v3 data/results/vllm/harm_dimensions_v2/GB10/Jury_v3` (if present).

---

## Out of Scope

- `full_runs/` directory (H100 original runs, old dimension schema) — left as-is, local only.
- Visualization scripts — not changed in this iteration; they accept `--results_dir` already.
