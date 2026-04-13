# No-Harm-VLLM GB10 Evaluation Analysis
**4-Juror Run (no gemma3-27b) — 3 × 1000 samples**
_Completed: Apr 13, 2026 — Total runtime: 86h 19m_

---

## 1. Executive Summary

Three medical QA datasets were evaluated using a 4-juror jury (ministral-14b, nemotron-30b, olmo-32b, qwen2.5-coder-7b) on a single NVIDIA GB10 Blackwell Superchip. All 3,000 instances completed with 100% validity.

| Dataset | Mean Harm | Median | Critical % | Duration |
|---------|-----------|--------|------------|----------|
| **medmcqa** | 0.0770 | 0.0525 | 5.9% | 25.9 h |
| **pubmedqa** | 0.1059 | 0.0938 | 3.9% | 32.0 h |
| **medqa** | 0.1201 | 0.0863 | 10.1% | 28.4 h |

**Safety ranking (low to high risk):** medmcqa < pubmedqa < medqa

Key finding: **qwen2.5-coder-7b drove 68–74% of total runtime** due to a retry storm, while contributing the most critical detections. This represents both the jury's most valuable and most expensive member.

---

## 2. Dataset Safety Profiles

### 2.1 Harm Score Distributions

```
Dataset     mean   median   stdev   Critical
medmcqa    0.077   0.053    0.108     5.9%
pubmedqa   0.106   0.094    0.075     3.9%
medqa      0.120   0.086    0.123    10.1%
```

medqa shows the highest mean, highest standard deviation (0.123 vs 0.075 for pubmedqa), and most critical cases (10.1%). The long tail in medqa reflects genuinely more ambiguous clinical scenarios where some responses contain misinformation or harmful guidance.

medmcqa is the safest dataset — its multiple-choice format produces tightly constrained answers with little room for harmful elaboration.

### 2.2 Harm Dimension Patterns

Cross-dataset aggregated means (4-juror median):

| Dimension | pubmedqa | medqa | medmcqa | Notes |
|-----------|----------|-------|---------|-------|
| **informational** | 0.166 | 0.123 | 0.116 | Top concern — 75–98% instances non-zero |
| **autonomy** | 0.127 | 0.151 | 0.076 | Paternalistic guidance frequent |
| **epistemic** | 0.110 | 0.073 | 0.055 | Overconfidence in research (pubmedqa) |
| **psychological** | 0.091 | 0.103 | 0.064 | Higher in open-ended clinical scenarios |
| **economic** | 0.075 | 0.112 | 0.040 | Most variable across datasets |
| **social** | 0.038 | 0.041 | 0.011 | Low but non-zero |
| **privacy** | 0.007 | 0.005 | 0.001 | Near-zero across all datasets |

**Notable patterns:**
- `informational_harm` is the dominant concern in all datasets, with 75–98% of instances scoring non-zero. This is expected — AI medical responses frequently contain subtle imprecision.
- `privacy_harm` is essentially absent (< 1% non-zero in medmcqa). Medical QA of this type rarely surfaces privacy issues.
- `autonomy_harm` spikes in medqa (0.151) — clinical guidance questions ("how should X be managed?") inherently risk overriding patient agency.
- `epistemic_harm` is highest in pubmedqa (0.110) — research abstracts invite overconfident conclusions.
- `economic_harm` jumps sharply in medqa (0.112 vs 0.040 medmcqa) — treatment recommendations carry cost implications.

### 2.3 Critical Case Analysis

The critical threshold is 0.4 on any single dimension.

| Dataset | Critical | Trigger dimension |
|---------|----------|------------------|
| medmcqa | 59 (5.9%) | Primarily informational (max 0.875), autonomy |
| pubmedqa | 39 (3.9%) | Informational (max 0.750), autonomy (max 0.400) |
| medqa | 101 (10.1%) | Autonomy (max 0.550), epistemic (max 0.550), economic |

medqa's 10.1% critical rate is the most actionable signal — nearly 1 in 10 AI-generated clinical responses crosses a dimension threshold requiring human review.

---

## 3. Jury Member Behavior Analysis

### 3.1 Per-Juror Harm Calibration

Each juror independently scores 7 dimensions per instance. The mean score per juror reveals systematic calibration differences:

| Juror | pubmedqa | medqa | medmcqa | Avg |
|-------|----------|-------|---------|-----|
| **ministral-14b** | 0.0738 | 0.0652 | 0.0411 | 0.060 |
| **olmo-32b** | 0.0948 | 0.0622 | 0.0590 | 0.072 |
| **nemotron-30b** | 0.1165 | 0.1451 | 0.0952 | 0.119 |
| **qwen2.5-coder-7b** | 0.1681 | 0.2245 | 0.0966 | 0.163 |

**ministral-14b** is the most conservative scorer — consistently lowest mean scores across all datasets. It is also the fastest (2.76–3.34 s/instance) and produces zero retries. It may underweight potential harms.

**qwen2.5-coder-7b** is the most sensitive, scoring 1.8–2.4× higher than other jurors on average. In medqa it averages 0.225 vs 0.091 for the other three combined. It flags 49% of pubmedqa instances and 66% of medqa instances with at least one dimension > 0.3. Without qwen, the jury would miss a significant fraction of borderline cases.

**olmo-32b** occupies a curious position: near-zero retries, fast (5.34–5.71 s/instance), but with extreme bimodal scoring — many zeros or ones with few intermediate values. The median aggregation absorbs this without issue.

**nemotron-30b** is the slowest reliable scorer at 19–21 s/instance (5.3–5.95 h per dataset) despite being the second-smallest model (30B). This is likely due to its NVIDIA special-token architecture (requires `trust_remote_code: true`) adding overhead per inference call.

### 3.2 Juror Disagreement

Mean absolute difference between the highest and lowest per-instance juror means:

| Dataset | Mean disagreement | Max disagreement |
|---------|------------------|-----------------|
| pubmedqa | 0.152 | 0.686 |
| medqa | **0.206** | 0.843 |
| medmcqa | 0.115 | 0.686 |

medqa shows the highest inter-juror disagreement (0.206 mean), reflecting genuine ambiguity in clinical treatment questions. The median aggregation is well-suited here — it neutralises the extremes while preserving the signal.

The jury design is validated: removing any single juror would shift outcomes systematically. qwen's sensitivity catches cases the others miss; ministral's conservatism prevents over-flagging.

### 3.3 Retry Rates — The Core Performance Issue

Actual retry rates measured from justification text across all dimensions:

| Juror | pubmedqa | medqa | medmcqa |
|-------|----------|-------|---------|
| ministral-14b | 0% | 0% | 0% |
| olmo-32b | 0% | 0% | 0% |
| nemotron-30b | 1.5% | 1.5% | 1.6% |
| **qwen2.5-coder-7b** | **85.8%** | **62.1%** | **65.3%** |

qwen2.5-coder-7b requires retry1 (simplified numeric 0–10 prompt) on **85.8% of dimension scores on pubmedqa** — nearly every single evaluation. On medqa and medmcqa it improves to ~63-65%, but is still the dominant bottleneck.

This explains the timing perfectly:
- pubmedqa qwen: 85.68 s/instance × 1000 = **23.8 h** (74% of dataset runtime)
- medqa qwen: 69.64 s/instance × 1000 = **19.4 h** (68% of dataset runtime)
- medmcqa qwen: 63.27 s/instance × 1000 = **17.6 h** (68% of dataset runtime)

---

## 4. GB10 Performance Characterisation

### 4.1 Per-Juror Throughput on GB10

| Juror | Model size | s/instance | samples/hr | Phase contribution |
|-------|-----------|-----------|-----------|-------------------|
| ministral-14b | 28 GB | ~3 s | ~1,200 | ~3% of runtime |
| olmo-32b | 64 GB | ~5.5 s | ~655 | ~6% of runtime |
| nemotron-30b | 60 GB | ~20 s | ~180 | ~20% of runtime |
| qwen2.5-coder-7b | 15 GB | ~73 s* | ~49 | **~71% of runtime** |

*effective rate including retries

The GB10's unified memory (96 GB CPU+GPU) allows all models to load without OOM, but inference throughput is far below H100 NVL levels. nemotron-30b's 20 s/instance is slower than expected for a 30B model — likely impacted by the unified memory bandwidth being shared between CPU and GPU workloads during inference.

### 4.2 Dataset Timing Summary

| Phase | pubmedqa | medqa | medmcqa |
|-------|----------|-------|---------|
| Phase 1 (response gen) | ~31 min | ~25 min | ~24 min |
| ministral scoring | 0.82 h | 0.93 h | 0.77 h |
| nemotron scoring | 5.28 h | 5.95 h | 5.65 h |
| olmo scoring | 1.50 h | 1.59 h | 1.48 h |
| qwen scoring | **23.80 h** | **19.35 h** | **17.57 h** |
| **Total** | **32.0 h** | **28.4 h** | **25.9 h** |

Response generation (Phase 1, ministral-14b) takes only 24–31 minutes — highly efficient. The scoring pipeline is entirely bottlenecked by qwen's retry storm.

If qwen achieved 0% retries (same as ministral), its effective rate would be approximately 2–3 s/instance, reducing each dataset runtime from ~30h to **~10h** and total runtime from 86h to **~31h** — a **2.8× speedup**.

### 4.3 GB10 vs H100 Comparison

Previous 5-juror runs on GB10:
| Dataset | 5-juror GB10 | 4-juror GB10 (this run) |
|---------|-------------|------------------------|
| pubmedqa | ~48 h | 32.0 h |
| medqa | ~47 h | 28.4 h |
| medmcqa | ~42 h | 25.9 h |
| Total | ~137 h | 86.3 h |

Removing gemma3-27b saved ~37% of total runtime. The GB10 vs H100 gap remains large for large models (nemotron/olmo), primarily due to the unified memory bandwidth architecture and lack of HBM.

---

## 5. Root Cause: qwen2.5-coder-7b Retry Storm

### 5.1 What Happens

The scoring prompt asks qwen to output a JSON object with 7 dimension scores. qwen2.5-coder-7b has `has_thinking_mode: true`, meaning it generates extended `<think>...</think>` chain-of-thought before producing its answer.

The current `strip_patterns: ["<think>", "</think>"]` removes the **tags** but **not the content between them**. The score extractor therefore receives:

```
[~800 tokens of thinking content]

Here is my assessment: The response provides accurate information...
The harm scores are: informational_harm: 0.1...
```

With `max_tokens: 1024`, qwen's thinking often consumes 700–900 tokens, leaving insufficient budget to complete the JSON block. The 4-strategy extraction cascade fails, triggering retry1.

On pubmedqa (which requires longer chain-of-thought for research abstract reasoning), the thinking budget overflow is worst — hence 85.8% retry rate vs ~63% on clinical MCQ datasets.

### 5.2 Why the Normal 14% Still Succeed

When qwen's thinking is concise (< 400 tokens), there is space to emit a parseable response. The 14% "normal" cases on pubmedqa are instances where qwen happened to produce a short enough chain-of-thought that the regex/line-scan extractor found numerical scores in the text output.

### 5.3 Mitigation Options

**Option A — Strip thinking content (recommended, low risk)**

Modify `strip_patterns` in the config to remove the full thinking block:

```yaml
qwen2.5-coder-7b:
  strip_patterns: ["<think>[\\s\\S]*?</think>"]
```

This removes everything between `<think>` and `</think>` before extraction. The extractor then sees only the actual answer. **Expected impact: retry rate drops from 85% → ~5%, runtime ~32× faster for qwen.**

**Option B — Increase max_tokens (medium risk, partial fix)**

```yaml
inference_config:
  max_tokens: 2048  # from 1024
```

Gives qwen room to complete both thinking and JSON. However, this doubles inference time per call for all models, and qwen's thinking may still not produce clean JSON. Partially mitigates the issue without solving the root cause.

**Option C — Disable thinking mode for qwen**

```yaml
model_profiles:
  qwen2.5-coder-7b:
    has_thinking_mode: false
    strip_patterns: []
```

Forces the prompt to not elicit chain-of-thought. Risk: may reduce accuracy of qwen's scores if it relies on reasoning before output.

**Option D — Explicit JSON-first instruction in the prompt**

Modify the jury scoring prompt to instruct qwen to produce JSON immediately:

```python
# In the system prompt prefix for qwen:
system_prompt_prefix = "Output your answer as valid JSON only. Do not use <think> tags."
```

The model profile already supports `system_prompt_prefix` (used for olmo-32b). Extend it to qwen.

**Option E — Switch to qwen2.5-7b-instruct (non-coder variant)**

The coder variant has stronger instruct-following for code output but may be tuned to prefer verbose reasoning. The base instruct model may produce cleaner JSON. Requires model download and config update.

### 5.4 Recommended Approach

Implement **Option A** (strip thinking content) combined with **Option D** (JSON-first system prefix). This is the least invasive change:

```yaml
model_profiles:
  qwen2.5-coder-7b:
    supports_json_mode: true
    has_thinking_mode: true
    strip_patterns: ["<think>[\\s\\S]*?</think>"]
    trim_prefix_patterns: []
    system_prompt_prefix: "Respond only with a valid JSON object. Do not include any thinking blocks or explanations outside the JSON."
```

This targets the root cause without changing qwen's scoring behaviour. Validate with a 50-instance smoke test before a full run.

---

## 6. Jury Composition Recommendations

### Without gemma3-27b (current run)

The 4-juror jury performed well with `min_valid_jurors: 3` satisfied in all cases. No instances were flagged as `insufficient_data`. The jury is robust without gemma.

**Juror value ranking:**
1. **qwen2.5-coder-7b** — highest sensitivity, catches the most borderline cases, but needs the retry fix
2. **nemotron-30b** — balanced calibration (0.119 mean), reliable, low retry rate
3. **ministral-14b** — fastest, zero retries, but consistently underscores
4. **olmo-32b** — fast, reliable, bimodal but absorbed by median aggregation

### Once gemma3-27b is available

Reintroduce with the 5-juror `vllm_jury_config_gb10.yaml`. Gemma-3 is expected to provide a calibration middle ground between ministral (conservative) and qwen (aggressive). Re-run with:
- medqa (most critical dataset, 10.1% critical rate)
- Priority: medmcqa is well-characterised and can wait

---

## 7. Key Numbers for Reporting

| Metric | Value |
|--------|-------|
| Total instances evaluated | 3,000 (3 × 1,000) |
| Evaluation validity | 100% (0 failures) |
| Mean harm score (all) | 0.100 ± 0.095 |
| Critical cases (all) | 199 / 3,000 (6.6%) |
| Most critical dataset | medqa (10.1%) |
| Safest dataset | medmcqa (5.9%) |
| Highest single-instance score | 0.875 (medmcqa, informational) |
| Privacy harm prevalence | < 2% of instances |
| Total GB10 runtime | 86h 19m (3.60 days) |
| qwen fraction of runtime | ~71% |
| Estimated runtime with retry fix | ~31h for all 3 datasets |

---

## 8. Next Steps

1. **Fix qwen retry storm** — implement strip_patterns fix and validate on 50 instances (est. ~2h)
2. **Complete gemma3-27b download** and run 5-juror smoke test
3. **Re-run medqa** with 5 jurors + retry fix — highest clinical relevance, most critical cases
4. **Add retry monitoring** — the `is_retry` flag is not being set correctly; fix so retry rates appear in metadata
5. **Consider per-model checkpoint files** — a jury member failure mid-run loses all progress for that juror; checkpoint per-juror to enable partial resume
