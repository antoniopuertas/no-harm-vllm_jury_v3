# No-Harm-VLLM — GB10 5-Juror Evaluation: Deep Analysis
**5-Juror Run — 3 × 1000 samples — NVIDIA GB10 Blackwell Superchip**
_Datasets: pubmedqa · medqa · medmcqa | Jurors: ministral-14b · nemotron-30b · olmo-32b · qwen2.5-coder-7b · gemma3-27b_

---

## 1. Executive Summary

Three medical QA datasets (3,000 instances total) were evaluated by a 5-model LLM jury on a single NVIDIA GB10 Blackwell Superchip. All instances completed with 100% validity. The jury exposed a clear safety gradient across datasets — medqa is the most clinically risky, medmcqa the safest — and revealed sharply contrasting behaviour across the five jury models, from extreme conservatism (ministral-14b) to aggressive harm flagging (qwen2.5-coder-7b, gemma3-27b).

| Dataset | Mean Harm | Median | Stdev | Critical % | Max Score |
|---------|-----------|--------|-------|------------|-----------|
| **pubmedqa** | 0.109 | 0.105 | 0.054 | 0.9% | 0.700 |
| **medqa** | 0.141 | 0.108 | 0.140 | 9.3% | 0.800 |
| **medmcqa** | 0.084 | 0.060 | 0.117 | 4.3% | 0.900 |

**Safety ranking (low → high risk):** medmcqa < pubmedqa < medqa

The most critical finding: **medqa generates a 9.3% Critical case rate**, meaning nearly 1 in 10 AI-generated clinical responses requires human review under these evaluation criteria. medqa's higher stdev (0.140) confirms genuine ambiguity — it is not a uniformly high-risk dataset but contains a long tail of genuinely dangerous responses.

---

## 2. Dataset Safety Profiles

### 2.1 Score Distributions

```
Dataset     mean    median   stdev    min    max    Critical
pubmedqa    0.109   0.105    0.054    0.000  0.700   0.9%
medqa       0.141   0.108    0.140    0.000  0.800   9.3%
medmcqa     0.084   0.060    0.117    0.000  0.900   4.3%
```

A notable structural observation: **all three datasets show zero Moderate or High cases** — every instance is either Low or Critical. This is a consequence of the scoring logic: the `critical_threshold` (any single dimension ≥ 0.4) produces a hard jump directly to Critical, bypassing intermediate categories. The median aggregation across 5 jurors suppresses minor disagreements, so borderline cases typically resolve to a Low verdict with the jury median pulling below 0.4, rather than landing in Moderate. This bimodal outcome distribution (Low / Critical) is expected behaviour given the aggregation design.

### 2.2 Harm Dimension Profiles

#### pubmedqa
Research abstracts generate elevated **informational** (mean 0.164, 97.0% non-zero) and **epistemic** (0.128, 96.5% non-zero) harm. Both near-universally fire above zero — almost every abstract-based AI response contains some degree of information imprecision or overconfident scientific interpretation. **Autonomy** harm is also high (0.148, 95.5%) because research summaries frequently imply clinical recommendations. **Privacy** harm is near-absent (0.006, only 4.8% non-zero) as expected: research questions do not expose patient data.

#### medqa
The most harm-diverse dataset. **Autonomy** dominates (mean 0.170, 90.9% non-zero) — clinical management questions by design involve guidance that may override patient agency. **Informational** harm is slightly lower than pubmedqa (0.148 vs 0.164) but produces more critical cases because clinical errors are action-consequential. **Economic** harm is the highest of the three datasets (0.131) reflecting the cost implications of treatment recommendations. **Epistemic** harm is paradoxically lower than pubmedqa (0.097 vs 0.128) because clinical MCQ answers tend to express more certainty.

#### medmcqa
The safest dataset by all measures. The multiple-choice format constrains responses to specific options, limiting elaboration. **Social** harm is near-absent (0.014, only 13.5% non-zero) and **privacy** harm is essentially zero (0.001, 0.6% non-zero). **Informational** harm is the primary concern (0.121, 68.4% non-zero) even here — even constrained answers contain medical imprecision.

#### Cross-dataset dimension summary
| Dimension | pubmedqa | medqa | medmcqa | Dominant pattern |
|-----------|----------|-------|---------|-----------------|
| informational | 0.164 | 0.148 | 0.121 | Ubiquitous across all formats |
| autonomy | 0.148 | 0.170 | 0.092 | Highest in clinical guidance (medqa) |
| epistemic | 0.128 | 0.097 | 0.074 | Highest in research abstracts (pubmedqa) |
| psychological | 0.110 | 0.124 | 0.078 | Gradual increase with clinical specificity |
| economic | 0.093 | 0.131 | 0.049 | Spikes in treatment recommendations |
| social | 0.053 | 0.056 | 0.014 | Format-dependent, low overall |
| privacy | 0.006 | 0.003 | 0.001 | Near-zero across all datasets |

### 2.3 Critical Case Anatomy

The critical threshold fires when the 5-juror median on any single dimension reaches ≥ 0.4. Across all three datasets, **informational harm is the dominant trigger**:

| Dataset | Critical | Top trigger | Share |
|---------|----------|-------------|-------|
| pubmedqa | 9 (0.9%) | informational | 55.6% |
| medqa | 93 (9.3%) | informational | 60.2% |
| medmcqa | 43 (4.3%) | informational | 88.4% |

medqa has the most diverse triggers: economic harm is the second trigger (22.6%) and autonomy the third (10.8%) — reflecting genuine multi-dimensional risk in clinical responses. medmcqa's critical cases are almost purely informational (88.4%), suggesting that when the constrained format produces a harmful response, it is specifically a factual error.

**Critical case consensus:** Every single critical instance in all three datasets was flagged by at least 3 jurors independently (the minimum consensus threshold). No critical verdict was produced by minority opinion. In medqa, 34.4% of critical cases were flagged by 4 jurors simultaneously and 2.2% by all 5, indicating the most dangerous responses are clearly identifiable across diverse model perspectives.

```
Critical instance juror agreement:
  pubmedqa: 3 jurors=55.6%, 4 jurors=44.4%, 5 jurors=0%
  medqa:    3 jurors=63.4%, 4 jurors=34.4%, 5 jurors=2.2%
  medmcqa:  3 jurors=60.5%, 4 jurors=39.5%, 5 jurors=0%
```

---

## 3. Jury Model Behaviour Analysis

### 3.1 Calibration Landscape

The five jury members span a 2.8× range of mean harm scores across all datasets:

| Juror | pubmedqa | medqa | medmcqa | Overall | Stdev (pubmedqa) |
|-------|----------|-------|---------|---------|-----------------|
| ministral-14b | 0.074 | 0.065 | 0.041 | **0.060** | 0.027 |
| olmo-32b | 0.095 | 0.062 | 0.059 | 0.072 | 0.053 |
| nemotron-30b | 0.117 | 0.145 | 0.095 | 0.119 | 0.047 |
| qwen2.5-coder-7b | 0.168 | 0.225 | 0.097 | 0.163 | 0.144 |
| gemma3-27b | 0.170 | 0.211 | 0.126 | **0.169** | 0.060 |

This is not random variation — each model exhibits a **systematic, reproducible bias** that persists across all three datasets and all seven dimensions.

### 3.2 ministral-14b — The Conservative Anchor

ministral-14b is the jury's most conservative member. Its overall mean (0.060) is 2.8× lower than gemma3-27b (0.169). Critically, it almost never fires above the moderate threshold:
- **0.1% sensitivity** on pubmedqa (only 1/1000 instances with any dim > 0.3)
- **1.0% sensitivity** on medqa, **1.2%** on medmcqa
- 24.1% of medmcqa instances receive an **all-zero score** across all 7 dimensions simultaneously — the model essentially declares perfect safety for nearly a quarter of medmcqa responses
- Maximum observed mean score across any instance: 0.26 — it never approaches the critical threshold on its own

**Score distribution:** 73% of all instances score in the 0.0–0.1 band; 0% ever reach 0.3–0.5 or above. ministral's dynamic range is essentially capped at 0.19.

**Per-dimension signature:** ministral is proportionally least deficient on informational (0.088) and autonomy (0.098) harm — its highest-scoring dimensions. It is essentially blind to social harm (0.032) and privacy harm (0.0004).

**Role in the jury:** ministral acts as a conservative floor. Its low scores prevent the median from being inflated by noise, but its consistent underscoring means it rarely contributes to critical verdicts. Removing ministral from the jury would change 4.6% of pubmedqa verdicts — the highest removal impact of any single juror — because its near-zero scores actively pull the median down in borderline cases.

**Throughput:** 2.93 s/instance on GB10 (fastest scorer). Zero retries. Reliable, deterministic output.

### 3.3 nemotron-30b — The Balanced Middle

nemotron-30b occupies the calibration middle: mean 0.119 across all datasets, neither systematically conservative nor aggressive. Its scores cluster in the 0.1–0.3 range (55.2% of instances), with essentially no all-zero instances on pubmedqa and medqa (0.0%). It is the only juror with near-zero all-zero rate across all datasets.

**Score distribution:** 42.9% of instances in 0.0–0.1, 55.2% in 0.1–0.3, 1.4% in 0.3–0.5 — a roughly bell-shaped curve centred at ~0.12. The stdev of instance means is moderate (0.047 on pubmedqa), indicating consistent scoring without extreme swings.

**Per-dimension signature:** nemotron is the strongest informational harm detector (0.243 mean across all datasets) — notably higher than any other juror on this dimension. It also scores meaningfully on autonomy (0.161) and economic (0.118) harm. It is relatively muted on social (0.047) and privacy (0.040) harm.

**Sensitivity:** 34.7% of pubmedqa, 56.5% of medqa, and 35.3% of medmcqa instances are flagged with at least one dimension > 0.3 — the second highest sensitivity rate after qwen. nemotron flags consistently across dataset types, unlike qwen which shows high variance.

**Reliability:** 1.4–1.7% retry rate across datasets (lowest among non-ministral jurors). nemotron's output is parseable JSON in 98%+ of cases with no thinking-mode interference.

**Throughput:** 18.72 s/instance — by far the slowest reliable scorer, 6.4× slower than the next slowest (olmo at 5.41 s). This is the GB10 bottleneck for nemotron and is architecture-related: nemotron's NVIDIA special-token format (`<|...|>`) adds per-token overhead and requires `trust_remote_code: true`, suggesting additional processing during generation. On a machine with dedicated HBM (H100), this overhead is less pronounced.

### 3.4 olmo-32b — The Bimodal Outlier

olmo-32b has a distinctive scoring pattern that sets it apart from all other jurors: extreme bimodality. It either gives zero to nearly all dimensions or gives substantial scores to several — with very little in between.

**All-zero rate:** 9.2% pubmedqa, 20.2% medqa, **33.3% medmcqa** — the highest all-zero rate of any juror. One third of medmcqa instances receive a flat 0.0 across all dimensions from olmo. This is not a failure mode; the model is functioning correctly. olmo genuinely treats many constrained MCQ responses as completely harmless.

**Score distribution:** 20.9% of all instances score exactly 0.0, 47.3% in 0.0–0.1, 31.5% in 0.1–0.3, but only 0.2% reach 0.3–0.5 and a tiny 0.1% go above 0.5. The distribution has a clear mass at zero with a secondary mass around 0.15. This bimodal shape means olmo is essentially a binary juror: it either sees no harm or moderate harm, rarely anything in between.

**Pairwise correlation:** olmo has the lowest correlation with qwen (0.115) — the two models fundamentally disagree on which instances are harmful. olmo's correlation with ministral (0.538) and gemma (0.463) is higher, suggesting its zero-or-not judgments align better with the conservative anchor than the aggressive flaggers.

**Role in the jury:** olmo's bimodal pattern is absorbed gracefully by the median aggregation. Its zeros pull the median down for safe instances without inflating it for risky ones. Removing olmo changes 11.1% of medqa verdicts — the second-highest removal impact after ministral — because its selective zero-scoring actively shapes the median in borderline cases.

**Throughput:** 5.41 s/instance, consistent with gemma (4.99–5.56 s/inst). Near-zero retries despite `has_thinking_mode: true` — olmo successfully outputs JSON with minimal interference from its thinking patterns.

### 3.5 qwen2.5-coder-7b — The Sensitive Extremist

qwen2.5-coder-7b is the jury's most sensitive and most volatile member. It scores 2.7× higher than ministral overall (0.163 vs 0.060) and shows the highest per-instance variance of any juror.

**Sensitivity:** 49.0% of pubmedqa, 66.2% of medqa, 23.5% of medmcqa instances are flagged with at least one dimension > 0.3 — far higher than any other juror. In medqa, two thirds of all responses receive a high-concern flag from qwen alone.

**Score distribution:** qwen is the only juror with a meaningful mass above 0.3: 14.0% of instances score 0.3–0.5 and 3.3% score above 0.5. Its stdev on medqa instance means is 0.146 — five times higher than ministral (0.037) — confirming that qwen's assessments swing dramatically across instances rather than clustering narrowly.

**Qwen vs gemma divergence:** qwen and gemma are the two "aggressive" flaggers, but they disagree substantially. On pubmedqa, qwen flags 44.4% of instances that gemma does not, while gemma flags only 1.3% that qwen misses. On medqa, qwen-only flagging is 43.2% vs gemma-only 7.2%. This means qwen is far more aggressive than gemma but captures different signal — removing either changes different sets of verdicts.

**Pairwise correlation with other jurors:** qwen has the lowest correlation with olmo (0.115) and is also relatively uncorrelated with nemotron (0.260). qwen essentially operates on its own axis, detecting harm in instances that the other four jurors rate as low risk.

**Retry storm:** qwen2.5-coder-7b generates `<think>...</think>` chain-of-thought blocks unconditionally. With the evaluation's 1024-token budget, these blocks exhaust the token window before JSON output is generated, causing parsing failure on virtually every primary response. Retry rates: 85.8% on pubmedqa, 62.1% on medqa, 65.3% on medmcqa (from the 4-juror reference run). Scores were ultimately obtained via Retry 1 (0–10 numeric scale, divided by 10). The scores are numerically valid but have reduced resolution (0.1 steps vs continuous) compared to full JSON output. This is an architectural property of the model, not fixable by prompt engineering alone.

**Size paradox:** qwen is the smallest model at 15 GB, yet contributes the highest sensitivity and most volatile scoring. This is likely a training artefact of the coder-7B variant, which is tuned for technical precision and tends to apply rigorous scrutiny to every response regardless of context.

### 3.6 gemma3-27b — The Calibrated Aggressive Scorer

gemma3-27b is the second-highest scorer overall (mean 0.169) but differs fundamentally from qwen in the nature of its aggressiveness. Where qwen is volatile (high stdev), gemma is consistent.

**Score distribution:** 71.2% of all instances score in the 0.1–0.3 band — a tight, well-calibrated distribution centred at ~0.15. Only 6.7% reach 0.3–0.5 and 0.3% above 0.5. gemma almost never scores zero (2.6% all-zero overall) and almost never scores extreme — it applies a steady, moderate-to-significant level of concern to nearly every response.

**Stdev comparison:** gemma's stdev on medqa is 0.097 — compared to qwen's 0.146. gemma sees meaningful variance across instances (it is responsive to content) but without qwen's extremes.

**Per-dimension signature:** gemma is the strongest scorer on psychological harm (0.219), autonomy harm (0.240), and epistemic harm (0.194) — dimensions related to patient relationship dynamics. gemma appears particularly sensitive to content that could affect patient mental state or decision-making power. Its informational harm mean (0.216) is comparable to qwen's (0.217) but with much lower variance.

**Sensitivity vs qwen:** gemma flags 30.2% of medqa instances with any dim > 0.3, compared to qwen's 66.2%. gemma is genuinely selective — it reserves high scores for instances it has reason to be concerned about, rather than flagging broadly.

**Pairwise correlation:** gemma has the highest correlation with ministral (0.710) — a counterintuitive finding given they score at opposite ends of the calibration spectrum. This suggests they agree on *which* instances are risky, even if not on *how* risky. gemma also correlates well with nemotron (0.605). qwen has the lowest correlation with gemma (0.359), confirming they detect different types of harm.

**Throughput:** 4.99–5.56 s/instance — highly consistent across all three datasets despite being the second-largest model at 54 GB. gemma's inference is efficient relative to its size: nemotron (60 GB) takes 3.4× longer per instance.

---

## 4. Jury Dynamics and Robustness

### 4.1 Inter-Juror Disagreement

The mean absolute spread between the highest and lowest juror means per instance:

| Dataset | Mean spread | Max spread | Stdev of spread |
|---------|-------------|------------|-----------------|
| pubmedqa | 0.177 | 0.686 | 0.104 |
| medqa | **0.237** | 0.843 | 0.114 |
| medmcqa | 0.141 | 0.686 | 0.092 |

medqa shows the highest inter-juror disagreement — instances of genuine clinical ambiguity produce the largest divergence between models. A spread of 0.843 means the highest-scoring juror gave a mean of ~0.86 across 7 dimensions while the lowest gave ~0.01 for the same instance. These extreme cases represent genuinely contested harm assessments where model perspective dominates over content signal.

**High-disagreement instances** (spread > 0.5):
- pubmedqa: 20 instances (mean spread among these: 0.561)
- medqa: 30 instances (mean spread: 0.576)
- medmcqa: 10 instances (mean spread: 0.583)

These 60 instances across all datasets represent the "genuinely contested" cases — where the jury itself cannot reach consensus. The median aggregation handles them correctly by defaulting to the middle, but they warrant human attention regardless of the final verdict.

### 4.2 Pairwise Juror Correlation

Pearson correlation on per-instance mean scores (all 3,000 instances pooled):

```
                  ministral  nemotron    olmo    qwen   gemma
ministral-14b       1.000     0.559    0.538   0.335   0.710
nemotron-30b        0.559     1.000    0.387   0.260   0.605
olmo-32b            0.538     0.387    1.000   0.115   0.463
qwen2.5-coder-7b    0.335     0.260    0.115   1.000   0.359
gemma3-27b          0.710     0.605    0.463   0.359   1.000
```

Several structural observations:

1. **gemma is the most correlated juror** (mean correlation with others: 0.534). It forms the jury's connective tissue — it agrees moderately with everyone, never becoming an outlier.

2. **qwen is the most independent juror** (mean correlation with others: 0.267). qwen operates largely on its own signal, which is why it catches cases the other four miss.

3. **olmo-qwen correlation (0.115)** is near-zero — these two models essentially disagree randomly with each other. This is the widest calibration gap in the jury.

4. **ministral-gemma (0.710)** is the strongest pairwise correlation despite their very different absolute score levels. They agree on *which* instances are riskier, even though ministral's scores are consistently lower.

5. All correlations are positive, which validates the jury design: every model has some genuine shared signal about harm, even the most divergent pair (qwen-olmo at 0.115).

### 4.3 Jury Robustness Analysis

How many final verdicts change if any single juror is removed:

| Removed | pubmedqa | medqa | medmcqa |
|---------|----------|-------|---------|
| ministral-14b | 4.6% | 12.3% | 6.0% |
| olmo-32b | 3.3% | 11.1% | 3.7% |
| qwen2.5-coder-7b | 0.7% | 7.7% | 3.4% |
| gemma3-27b | 3.2% | 7.2% | 3.8% |
| nemotron-30b | 1.2% | 5.9% | 2.3% |

**Key insight:** ministral-14b causes the most verdict instability when removed (4.6–12.3%), despite — or rather because of — being the most conservative juror. When ministral's near-zero scores are removed, the median of the remaining 4 shifts upward, flipping borderline Low cases to Critical. The jury depends on ministral not as a harm detector but as a stabilising anchor.

**medqa is the most jury-sensitive dataset.** Removing any single juror changes 5.9–12.3% of medqa verdicts, versus 0.7–4.6% for pubmedqa. This reflects medqa's genuine ambiguity — its verdicts sit closer to decision boundaries, where small shifts in median values flip outcomes.

**qwen's low removal impact (0.7% on pubmedqa)** is initially surprising. The explanation: qwen's extreme scores are already being pulled toward the median and, since qwen is outlier-high, removing it rarely changes the 5-juror median enough to flip a verdict in the downward direction.

---

## 5. GB10 Hardware Performance

### 5.1 Scoring Throughput by Model

Measured from `gb10_5juror` run (pubmedqa, April 14–15 2026):

| Model | Size | s/instance | samples/hr | Fraction of runtime |
|-------|------|-----------|-----------|---------------------|
| ministral-14b (Phase 1+scoring) | 28 GB | 2.93 | 1,229 | ~9% |
| gemma3-27b | 54 GB | 4.99–5.56 | 660 | ~16% |
| olmo-32b | 64 GB | 5.41 | 665 | ~17% |
| qwen2.5-coder-7b | 15 GB | ~61* | ~59 | — |
| nemotron-30b | 60 GB | 18.72 | 192 | **~58%** |

*qwen rate with retry storm (unusable for clean timing); clean estimate ~2–3 s/inst if thinking disabled

**nemotron-30b is the GB10 bottleneck** under normal conditions (no qwen retry storm). At 18.72 s/instance, it is 3.4× slower than gemma (which is 9 GB larger) and 6.4× slower than olmo (which is also larger). This anomaly is due to nemotron's NVIDIA special-token architecture requiring `trust_remote_code: true`, which adds per-token processing overhead that is unusually expensive on the GB10's unified memory architecture.

### 5.2 Size vs Speed Paradox

The GB10 results reveal a strong disconnect between model size and inference speed:

```
Model          Size    s/inst    MB/s effective
gemma3-27b     54 GB   5.0 s    ~10.8 GB/s
olmo-32b       64 GB   5.4 s    ~11.9 GB/s
ministral-14b  28 GB   2.9 s    ~9.6 GB/s
nemotron-30b   60 GB   18.7 s   ~3.2 GB/s  ← 3–4× underperforming
qwen2.5-coder  15 GB   2.9 s*   ~5.2 GB/s  (*no retry)
```

gemma3-27b achieves near-equivalent throughput to olmo-32b despite being 10 GB smaller — both achieve approximately 10–12 GB/s effective memory bandwidth utilisation. nemotron achieves only 3.2 GB/s effective, suggesting it is CPU-bound rather than memory-bandwidth-bound on the GB10 architecture.

### 5.3 GB10 Unified Memory Characteristics

The GB10 Blackwell Superchip uses a unified CPU+GPU memory architecture with 96 GB total addressable LPDDR5X memory. All models loaded sequentially without OOM at `gpu_memory_utilization: 0.5`. Key observations:

**What works well:**
- Sequential model loading: every model in the 15–64 GB range loads reliably within the 96 GB budget
- No tensor parallelism required: `tensor_parallel_size: 1` for all models
- Sustained multi-hour inference: all models ran 1000+ instance scoring passes without memory pressure or crashes
- Response generation is fast: 1000 responses generated in ~34 minutes using ministral-14b in batch mode (32/batch)

**GB10 limitations:**
- Memory bandwidth ceiling for large models: gemma (54 GB) and olmo (64 GB) push the GB10 to near-capacity at `gpu_memory_utilization: 0.5`, leaving limited headroom
- nemotron overhead: unified memory is shared between CPU and GPU — models requiring CPU-side processing (nemotron's special tokens) suffer disproportionately on this architecture
- Sequential-only execution: all 5 jurors must score sequentially; no parallelism is possible with a single GB10 chip
- `nvidia-smi` does not report memory usage (`Memory-Usage: Not Supported`) — cannot monitor VRAM utilisation in real time
- Total GB10 runtime for all 3 datasets (via gemma merge + partial fresh run): approximately **90+ hours** for a full 5-juror evaluation including qwen's retry storm

### 5.4 Phase 1 Response Generation

Ministral-14b generates 1000 responses in approximately 34 minutes on GB10 (including container load time), at roughly 32 responses per 60-second batch. This phase is fast and reliable and represents a negligible fraction of total evaluation time. The bottleneck is always jury scoring, not response generation.

### 5.5 Practical Runtime Estimates (GB10)

| Configuration | Estimated time / 1000 samples |
|---------------|-------------------------------|
| 5-juror, nemotron bottleneck (no qwen retry) | ~11 h |
| 5-juror, qwen retry storm active | ~33 h |
| 4-juror (no gemma) | ~28 h (measured) |
| Gemma-only scoring on existing results | ~1.5 h |
| 3-dataset full evaluation (5-juror, clean) | ~33 h |
| 3-dataset full evaluation (5-juror, with qwen retry) | ~90 h |

---

## 6. Key Findings and Structural Insights

### The Calibration Spectrum Serves a Purpose

The jury's 2.8× spread in mean scores is not a flaw — it is the feature. Without ministral's conservatism (0.060), the median would be artificially inflated; without qwen and gemma's aggressiveness (0.163–0.169), borderline cases would be missed. The Pearson correlations (all positive, ranging 0.115–0.710) confirm that every model contributes independent signal.

### The Bimodal Verdict Problem

The absence of Moderate and High verdicts — every instance is either Low or Critical — is a limitation of the current aggregation design. The hard critical threshold (single dimension ≥ 0.4) combined with median aggregation creates a bimodal output that may lose useful gradient information. Instances scoring 0.38 on a dimension look identical to instances scoring 0.05, yet the former is far closer to requiring review.

### Informational Harm Dominates

Across all datasets, informational harm is the most prevalent dimension (non-zero in 68–97% of instances depending on dataset) and the leading trigger for Critical verdicts (55–88% of all critical cases). AI medical responses almost universally contain some degree of informational imprecision — this is not a sign that the jury is over-sensitive but that medical language generation inherently involves simplified or potentially misleading statements.

### Privacy Harm is Essentially Absent

Privacy harm scored non-zero in fewer than 5% of instances across all datasets (0.6% in medmcqa). Medical QA evaluation of this format (question → AI response) does not expose patient data, so this result is expected and validates that the jury is not producing false positives on a dimension that should be near-zero.

### qwen's Independence is Valuable but Costly

qwen2.5-coder-7b has the lowest mean correlation with any other juror (0.267 average) and flags cases the other four models miss. In medqa, qwen-only flags represent 43.2% of all instances — these are cases the rest of the jury rates as low risk, but qwen identifies as potentially harmful. Whether these are true positives or false positives requires human validation, but the signal itself is unique and would be lost by removing qwen from the jury. The cost is architectural: qwen's thinking mode makes it the most expensive and least reliable scorer by output format.

---

## 7. Statistics Reference for Documentation

### Dataset-level
| Metric | pubmedqa | medqa | medmcqa | All |
|--------|----------|-------|---------|-----|
| Instances evaluated | 1,000 | 1,000 | 1,000 | 3,000 |
| Mean harm score | 0.109 | 0.141 | 0.084 | 0.111 |
| Median harm score | 0.105 | 0.108 | 0.060 | — |
| Score stdev | 0.054 | 0.140 | 0.117 | — |
| Critical cases | 9 (0.9%) | 93 (9.3%) | 43 (4.3%) | 145 (4.8%) |
| Max observed score | 0.700 | 0.800 | 0.900 | 0.900 |
| Min observed score | 0.000 | 0.000 | 0.000 | 0.000 |

### Juror-level (overall)
| Juror | Mean score | Stdev | Sensitivity (any dim>0.3) | All-zero rate |
|-------|------------|-------|--------------------------|---------------|
| ministral-14b | 0.060 | 0.033 | 0.8% | 10.8% |
| olmo-32b | 0.072 | 0.057 | 9.5% | 20.9% |
| nemotron-30b | 0.119 | 0.055 | 42.2% | 0.6% |
| qwen2.5-coder-7b | 0.163 | 0.133 | 46.2% | 10.7% |
| gemma3-27b | 0.169 | 0.080 | 17.2% | 2.6% |

### GB10 throughput
| Model | s/instance | samples/hr |
|-------|-----------|-----------|
| ministral-14b | 2.93 | 1,229 |
| gemma3-27b | 5.21 | 691 |
| olmo-32b | 5.41 | 665 |
| qwen2.5-coder-7b | ~61* | ~59 |
| nemotron-30b | 18.72 | 192 |

*With retry storm. Clean estimate: ~2–3 s/inst.

### Jury robustness (% verdicts changed by single juror removal)
| Removed | pubmedqa | medqa | medmcqa |
|---------|----------|-------|---------|
| ministral-14b | 4.6% | 12.3% | 6.0% |
| olmo-32b | 3.3% | 11.1% | 3.7% |
| qwen2.5-coder-7b | 0.7% | 7.7% | 3.4% |
| gemma3-27b | 3.2% | 7.2% | 3.8% |
| nemotron-30b | 1.2% | 5.9% | 2.3% |

### Inter-juror agreement
| Pair | Pearson r |
|------|-----------|
| ministral-14b / gemma3-27b | 0.710 |
| nemotron-30b / gemma3-27b | 0.605 |
| ministral-14b / nemotron-30b | 0.559 |
| ministral-14b / olmo-32b | 0.538 |
| olmo-32b / gemma3-27b | 0.463 |
| nemotron-30b / olmo-32b | 0.387 |
| qwen2.5-coder-7b / gemma3-27b | 0.359 |
| ministral-14b / qwen2.5-coder-7b | 0.335 |
| nemotron-30b / qwen2.5-coder-7b | 0.260 |
| olmo-32b / qwen2.5-coder-7b | 0.115 |
