# Repo Cleanup Design — 2026-02-27

## Goal

Prepare `no-harm-vllm_jury_v3` for public GitHub upload with a clean structure and comprehensive README.

## Decisions

- **Approach B**: Comprehensive README + one hardware doc (`docs/HARDWARE_SETUP.md`)
- Old planning files, logs, and session transcripts move to `/home/puertao/llm/jury_v3_archive/`

---

## File Organization

### Move to `/home/puertao/llm/jury_v3_archive/`

**Session/conversation logs (root):**
- `20260220_jury_v3_claude_full_dialog.txt`
- `20260220_jury_v3_claude.txt`
- `20260220_jury_v3_debugging_session.txt`
- `20260220_jury_v3_tasks_1_14_dialog.txt`
- `test_to_be_removed.txt`

**Log files (root):**
- `full_evaluation_20260220_162810.log`
- `test_evaluation_20260220_163040.log`
- `test_evaluation_20260220_163350.log`

**Planning/proposal docs (root):**
- `jury_v3_roadmap.md`
- `jury_v3_parallel_roadmap.md`
- `PROPOSAL.md`
- `new_proposal.md`
- `README_Qwen2.5.md`
- `DEBUGGING_SUMMARY.md`
- `EVALUATION_GUIDE.md`

**Debug test scripts (root):**
- `test_fix1_max_tokens.py`
- `test_fix2_format_example.py`

**Docs (internal planning):**
- `docs/plans/2026-02-20-jury-v3-production-implementation.md`
- `docs/plans/2026-02-20-jury-v3-production-ready-design.md`
- `docs/RECOVERY_GUIDE.md`

### Transform in place

- `docs/DUAL_GPU_IMPLEMENTATION.md` → rewritten as `docs/HARDWARE_SETUP.md`
- `docs/plans/` → removed after archive (no longer needed)

### Final repo structure

```
no-harm-vllm_jury_v3/
├── README.md
├── requirements.txt
├── config.yaml
├── run_evaluation.py
├── src/
├── scripts/
├── tests/
├── prompt_templates/
├── config/
├── data/
├── logs/
└── docs/
    └── HARDWARE_SETUP.md
```

---

## README.md Structure

1. **The Problem** — v2.3 0.5 ceiling artifact, 3/5 jury model parse failures
2. **How It Works** — 5-model jury, 7 harm dimensions, 4-strategy cascade parser, retry logic, weighted aggregation
3. **The 5 Jury Models** — table: model, size, HF path
4. **The 7 Harm Dimensions** — table: dimension, what it measures
5. **Installation** — pip install -r requirements.txt
6. **Quick Start** — single command example
7. **Configuration** — config.yaml fields explained
8. **Output** — result fields explained
9. **Hardware Requirements** — brief note + link to docs/HARDWARE_SETUP.md
10. **Results** — v2.3 vs v3.0 comparison table (parse success, score range)

---

## docs/HARDWARE_SETUP.md Structure

- VRAM budget per model (table)
- Parallel loading slot configurations (Slot A: 3 models, Slot B: 2 large models)
- 2×H100 scheduling strategy
- Graceful degradation if a model fails to load
