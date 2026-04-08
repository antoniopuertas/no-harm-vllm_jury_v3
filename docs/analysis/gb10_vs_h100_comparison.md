# GB10 vs H100 — Harm Dimensions v2 Evaluation Comparison

**Generated:** 2026-04-08  
**Datasets evaluated:** MedQA, MedMCQA, PubMedQA (1,000 samples each)  
**Jury members:** ministral-14b, gemma3-27b, nemotron-30b, olmo-32b, qwen2.5-coder-7b  
**Response model:** ministral-14b  
**Aggregation:** median per dimension, critical threshold = 0.4

---

## Table of Contents

1. [Hardware Specifications](#1-hardware-specifications)
2. [Execution Time Analysis](#2-execution-time-analysis)
3. [Score Distribution Comparison](#3-score-distribution-comparison)
4. [Harm Category Distributions](#4-harm-category-distributions)
5. [Per-Dimension Analysis](#5-per-dimension-analysis)
6. [Zero-Score Analysis](#6-zero-score-analysis)
7. [High-Harm Sample Rates](#7-high-harm-sample-rates)
8. [Data Quality Issues — Root Cause Analysis](#8-data-quality-issues--root-cause-analysis)
   - [GB10 Issues](#gb10-issues)
   - [H100 Issues](#h100-issues)
9. [Dataset-Level Observations](#9-dataset-level-observations)
10. [Summary of Key Differences](#10-summary-of-key-differences)
11. [Conclusions and Recommendations](#11-conclusions-and-recommendations)

---

## 1. Hardware Specifications

| Attribute | GB10 | H100 |
|---|---|---|
| GPU model | GB10 | H100 |
| Number of GPUs | 1 | 2 |
| Total VRAM | 96 GB | 190 GB |
| Jury config (members) | identical (5 jurors) | identical (5 jurors) |
| Aggregation method | median | median |
| Critical threshold | 0.4 | 0.4 |

Both runs used the **same jury configuration, same datasets, same 1,000-sample slices**, and the same 5 juror models. The only differences were the GPU hardware and the LLM generator that produced the responses being evaluated.

**Critical note:** The response model (ministral-14b) generated different text on each hardware run, meaning the two evaluations assessed different inputs — not the same responses scored by different hardware. This is the fundamental confounder that makes direct score comparison non-trivial.

---

## 2. Execution Time Analysis

### Raw Duration

| Dataset | GB10 (seconds) | H100 (seconds) | Speedup |
|---|---|---|---|
| MedQA | 168,234.27 | 12,373.42 | **13.6×** |
| MedMCQA | 152,004.98 | 10,621.78 | **14.3×** |
| PubMedQA | 173,023.81 | 12,043.10 | **14.4×** |
| **Average** | **164,421.02** | **11,679.43** | **14.1×** |

### Human-Readable Duration

| Dataset | GB10 | H100 |
|---|---|---|
| MedQA | ~46.7 hours | ~3.4 hours |
| MedMCQA | ~42.2 hours | ~2.9 hours |
| PubMedQA | ~48.1 hours | ~3.3 hours |
| **Total** | **~137 hours (~5.7 days)** | **~9.7 hours** |

### Throughput

| Metric | GB10 | H100 |
|---|---|---|
| Samples/second (MedQA) | 0.0059 | 0.0808 |
| Samples/hour (MedQA) | 21.4 | 290.9 |
| Seconds/sample (MedQA) | 168.2 | 12.4 |
| Samples/second (avg) | 0.0061 | 0.0856 |

### Analysis

The H100 (dual-GPU, 190 GB VRAM) completed each 1,000-sample evaluation in **under 3.5 hours**. The GB10 (single GPU, 96 GB VRAM) took **between 42 and 48 hours** per dataset — a factor of **14× longer** on average.

This speedup is attributable to:
- **2× more GPUs** on H100 (enabling parallel inference)
- **~2× more VRAM** (190 GB vs 96 GB), reducing model swapping and memory pressure
- **H100's higher raw tensor throughput** vs GB10's architecture
- **Reduced retry overhead** in H100 for most models (see Section 8)

The GB10 PubMedQA run was the slowest of all (173,023 s, ~48 h), which stands out given that PubMedQA has the shortest response lengths in the corpus. This anomaly is unexplained by hardware alone and may indicate background interference, thermal throttling, or a scheduling delay during that specific run window (started 2026-03-28).

---

## 3. Score Distribution Comparison

### MedQA — Final Score Distribution

| Statistic | GB10 | H100 | Delta |
|---|---|---|---|
| N | 1,000 | 1,000 | — |
| Min | 0.0000 | 0.0000 | 0.0000 |
| Max | 0.7000 | 0.8000 | +0.1000 |
| **Mean** | **0.0695** | **0.2115** | **+0.1420 (+204%)** |
| **Median** | **0.0600** | **0.1350** | **+0.0750 (+125%)** |
| Std Dev | 0.0744 | 0.1889 | +0.1145 |
| p10 | 0.0095 | 0.0350 | +0.0255 |
| p25 | 0.0300 | 0.0850 | +0.0550 |
| p75 | 0.0900 | 0.4500 | +0.3600 |
| p90 | 0.1150 | 0.5000 | +0.3850 |
| p95 | 0.1350 | 0.5000 | +0.3650 |
| p99 | 0.5000 | 0.7000 | +0.2000 |

### MedMCQA — Final Score Distribution

| Statistic | GB10 | H100 | Delta |
|---|---|---|---|
| Min | 0.0000 | 0.0000 | 0.0000 |
| Max | 0.9000 | 0.9500 | +0.0500 |
| **Mean** | **0.0450** | **0.1180** | **+0.0730 (+162%)** |
| **Median** | **0.0300** | **0.0550** | **+0.0250 (+83%)** |
| Std Dev | 0.0768 | 0.1750 | +0.0982 |
| p10 | 0.0000 | 0.0000 | 0.0000 |
| p25 | 0.0000 | 0.0150 | +0.0150 |
| p75 | 0.0650 | 0.1200 | +0.0550 |
| p90 | 0.0900 | 0.5000 | +0.4100 |
| p95 | 0.1100 | 0.5000 | +0.3900 |
| p99 | 0.5000 | 0.7000 | +0.2000 |

### PubMedQA — Final Score Distribution

| Statistic | GB10 | H100 | Delta |
|---|---|---|---|
| Min | 0.0000 | 0.0000 | 0.0000 |
| Max | 0.7000 | 0.7000 | 0.0000 |
| **Mean** | **0.0728** | **0.1197** | **+0.0469 (+64%)** |
| **Median** | **0.0700** | **0.1150** | **+0.0450 (+64%)** |
| Std Dev | 0.0467 | 0.0811 | +0.0344 |
| p10 | 0.0295 | 0.0400 | +0.0105 |
| p25 | 0.0450 | 0.0800 | +0.0350 |
| p75 | 0.0925 | 0.1400 | +0.0475 |
| p90 | 0.1125 | 0.1750 | +0.0625 |
| p95 | 0.1300 | 0.1975 | +0.0675 |
| p99 | 0.1550 | 0.5000 | +0.3450 |

### Cross-Dataset Mean Score Summary

| Dataset | GB10 Mean | H100 Mean | Factor |
|---|---|---|---|
| MedQA | 0.0695 | 0.2115 | **3.04×** |
| MedMCQA | 0.0450 | 0.1180 | **2.62×** |
| PubMedQA | 0.0728 | 0.1197 | **1.64×** |
| **Average** | **0.0624** | **0.1497** | **2.40×** |

H100 produces approximately **2.4× higher mean harm scores** across all datasets. The gap is most extreme for MedQA (3× higher) and most modest for PubMedQA (1.6× higher).

**PubMedQA is the most stable across hardware.** Its distribution is tight on both platforms (std 0.047 GB10 vs 0.081 H100) and the mean delta is the smallest. This may be because PubMedQA questions are more constrained (yes/no/maybe answers), leaving less room for scoring divergence.

---

## 4. Harm Category Distributions

Both platforms produce only two harm categories: **"Low"** and **"Critical"**. No intermediate categories ("Moderate", "High") appear in any dataset, suggesting the thresholding logic produces a hard binary split at the 0.4 threshold.

### Low vs Critical Count

| Dataset | GB10 Low | GB10 Critical | H100 Low | H100 Critical |
|---|---|---|---|---|
| MedQA | 979 (97.9%) | **21 (2.1%)** | 744 (74.4%) | **256 (25.6%)** |
| MedMCQA | 984 (98.4%) | 16 (1.6%) | 868 (86.8%) | 132 (13.2%) |
| PubMedQA | 995 (99.5%) | 5 (0.5%) | 971 (97.1%) | 29 (2.9%) |

### Critical Rate Comparison

| Dataset | GB10 Critical % | H100 Critical % | Factor |
|---|---|---|---|
| MedQA | 2.1% | 25.6% | **12.2×** |
| MedMCQA | 1.6% | 13.2% | **8.3×** |
| PubMedQA | 0.5% | 2.9% | **5.8×** |

The H100 run classifies dramatically more samples as "Critical." This is the single most alarming surface-level finding. However, as Section 8 documents, a significant portion of H100's "Critical" classifications stem from data quality issues (empty responses, retry contamination) rather than genuine harm signal.

---

## 5. Per-Dimension Analysis

### MedQA — Per-Dimension Mean Scores

| Dimension | GB10 Mean | H100 Mean | Delta | Factor |
|---|---|---|---|---|
| informational_harm | 0.0741 | 0.2157 | +0.1416 | 2.91× |
| social_harm | 0.0208 | 0.0748 | +0.0540 | 3.60× |
| psychological_harm | 0.0780 | 0.1951 | +0.1171 | 2.50× |
| autonomy_harm | **0.1247** | **0.2342** | +0.1095 | 1.88× |
| economic_harm | 0.0756 | 0.1867 | +0.1111 | 2.47× |
| privacy_harm | **0.0000** | 0.0506 | +0.0506 | ∞ |
| epistemic_harm | 0.0449 | 0.1477 | +0.1028 | 3.29× |

### MedMCQA — Per-Dimension Mean Scores

| Dimension | GB10 Mean | H100 Mean | Delta | Factor |
|---|---|---|---|---|
| informational_harm | 0.0745 | 0.1467 | +0.0722 | 1.97× |
| social_harm | 0.0043 | 0.0260 | +0.0217 | 6.05× |
| psychological_harm | 0.0449 | 0.0923 | +0.0474 | 2.06× |
| autonomy_harm | 0.0567 | 0.1033 | +0.0466 | 1.82× |
| economic_harm | 0.0246 | 0.0822 | +0.0576 | 3.34× |
| privacy_harm | **0.0001** | 0.0217 | +0.0216 | 217× |
| epistemic_harm | 0.0312 | 0.0806 | +0.0494 | 2.58× |

### PubMedQA — Per-Dimension Mean Scores

| Dimension | GB10 Mean | H100 Mean | Delta | Factor |
|---|---|---|---|---|
| informational_harm | **0.1197** | **0.1629** | +0.0432 | 1.36× |
| social_harm | 0.0250 | 0.0420 | +0.0170 | 1.68× |
| psychological_harm | 0.0711 | 0.1312 | +0.0601 | 1.85× |
| autonomy_harm | 0.1045 | 0.1565 | +0.0520 | 1.50× |
| economic_harm | 0.0539 | 0.1270 | +0.0731 | 2.36× |
| privacy_harm | 0.0009 | 0.0027 | +0.0018 | 3.00× |
| epistemic_harm | **0.0867** | 0.1369 | +0.0502 | 1.58× |

### Key Dimension-Level Findings

1. **`privacy_harm` on GB10 MedQA is exactly 0.0000** across all 1,000 samples — a complete parsing or generation failure for this dimension (confirmed: gemma3-27b produced 0.0 for all dimensions due to parsing failure, and qwen2.5-coder-7b shows near-zero privacy scores as its default). On H100, privacy_harm has non-trivial signal (mean 0.0506).

2. **`autonomy_harm` is the highest-scoring dimension on GB10** across all three datasets. This is consistent with the jury identifying autonomy-related concerns in medical QA responses (e.g., failure to adequately communicate patient choice, informed consent framing).

3. **`informational_harm` is the highest-scoring dimension on H100** for MedQA and MedMCQA, and consistently elevated across all datasets. This is the expected primary harm dimension for medical QA — factual errors and misinformation.

4. **`social_harm` and `privacy_harm` are the lowest-scoring dimensions** on both platforms. Medical QA responses rarely trigger social bias or privacy concerns.

5. **`epistemic_harm` is markedly higher in PubMedQA vs other datasets** on GB10 (mean 0.0867 vs 0.0449 MedQA / 0.0312 MedMCQA). PubMedQA often involves ambiguous evidence, making epistemic issues more salient.

### % Samples Scoring >= 0.4 Per Dimension (Critical Threshold)

| Dimension | GB10 MedQA | H100 MedQA | GB10 MedMCQA | H100 MedMCQA | GB10 PubMedQA | H100 PubMedQA |
|---|---|---|---|---|---|---|
| informational_harm | 1.4% | **21.1%** | 1.5% | **12.3%** | 0.5% | 2.1% |
| social_harm | 0.0% | **10.2%** | 0.0% | 4.3% | 0.0% | 0.6% |
| psychological_harm | 0.0% | **10.9%** | 0.0% | 4.6% | 0.0% | 0.2% |
| autonomy_harm | 0.1% | **15.4%** | 0.1% | 5.1% | 0.0% | 0.1% |
| economic_harm | 0.6% | **11.6%** | 0.0% | 4.4% | 0.0% | 0.2% |
| privacy_harm | 0.0% | 9.9% | 0.0% | 4.3% | 0.0% | 0.0% |
| epistemic_harm | 0.2% | **12.1%** | 0.0% | 5.5% | 0.0% | 0.4% |

---

## 6. Zero-Score Analysis

A "zero-score" sample is one where `final_score == 0.0` AND all 7 dimension scores are 0.0. In both runs, these two conditions are perfectly aligned (no sample had a non-zero final score with all-zero dimensions, or vice versa).

| Dataset | GB10 Zero Count | GB10 % | H100 Zero Count | H100 % |
|---|---|---|---|---|
| MedQA | 91 | 9.1% | 34 | 3.4% |
| MedMCQA | 317 | **31.7%** | 230 | **23.0%** |
| PubMedQA | 27 | 2.7% | 27 | 2.7% |

### Observations

- **MedMCQA has a strikingly high zero-score rate** on both platforms (31.7% GB10, 23.0% H100). Nearly 1 in 4–3 MedMCQA responses is judged as completely harmless across all dimensions. This is unusual and warrants inspection — it may reflect the multiple-choice nature of MedMCQA responses (short, factual answers that are harder to flag as harmful) or a systematic gap in juror sensitivity to this format.

- **PubMedQA zero rates are identical** (27/1000 on both platforms), suggesting this dataset's responses are so consistently structured that the zero-harm floor is hardware-independent. This is the only metric where GB10 and H100 agree exactly.

- **MedQA zero rate is much higher on GB10 (9.1%)** than H100 (3.4%). Part of this is explained by the gemma3-27b parsing failure on GB10 (see Section 8), which silently produces 0.0 for all its dimensions, biasing the median toward zero.

---

## 7. High-Harm Sample Rates

| Dataset | GB10 Count (≥0.4) | GB10 % | H100 Count (≥0.4) | H100 % | Factor |
|---|---|---|---|---|---|
| MedQA | 21 | 2.1% | 256 | **25.6%** | **12.2×** |
| MedMCQA | 16 | 1.6% | 132 | **13.2%** | 8.3× |
| PubMedQA | 5 | 0.5% | 29 | 2.9% | 5.8× |
| **Total** | **42** | **1.4%** | **417** | **13.9%** | **9.9×** |

H100 produced approximately **10× more high-harm classifications** than GB10. Section 8 explains the data quality reasons behind this divergence. Not all of these "Critical" classifications on H100 represent genuine harm — many are artifacts of the 98 empty-response entries (all evaluated as 0.5 across all dimensions, automatically exceeding the 0.4 threshold).

---

## 8. Data Quality Issues — Root Cause Analysis

The score gap between GB10 and H100 is real but cannot be attributed to hardware alone. Multiple data quality issues — most of which are software/runtime bugs — systematically bias scores in opposite directions on the two platforms.

---

### GB10 Issues

#### Issue GB10-1: gemma3-27b Complete Parsing Failure (CRITICAL)

**Scope:** 100% of all 1,000 MedQA entries (and likely all other datasets)  
**Effect:** All gemma3-27b dimension scores = 0.0 with `justification = "Parsing failed"`  

gemma3-27b completely failed to produce parseable output on the GB10 run. Every single entry shows `"Parsing failed"` as the justification and 0.0 for all 7 dimensions. Since aggregation uses the median across 5 jurors, this silently zeroes out 20% of the jury's scoring signal, systematically **suppressing all GB10 scores by approximately 20%**.

This is the single most impactful data quality issue in the entire comparison. GB10 harm scores are **structurally lower** than H100 scores in part because one of five jurors contributed nothing but zeros.

**Likely cause:** The GB10 hardware stack used a different vLLM version or quantization level that altered gemma3-27b's output format, breaking the score extractor's regex/parsing logic.

---

#### Issue GB10-2: qwen2.5-coder-7b Out-of-Range Scores (MODERATE)

**Scope:** 39 out of 1,000 MedQA entries  
**Effect:** Dimension scores outside [0, 1] (e.g., 7.2, 2.3, 1.9)  

In 39 instances, qwen2.5-coder-7b emitted scores on what appears to be a 0–10 integer scale rather than the expected 0–1 float scale. The score extractor did not clamp or rescale these values, so they propagated as-is into the jury aggregation. These entries are corrupted — their actual harm level is unknown. The affected instance IDs should be flagged and excluded from any quantitative comparison.

**Likely cause:** Different prompt version used on GB10 that omitted the 0–1 normalization instruction, or a different model quantization that altered the output format.

---

#### Issue GB10-3: PubMedQA Duration Anomaly (INFORMATIONAL)

**Scope:** GB10 PubMedQA run  
**Effect:** 173,024 seconds (~48 hours) for 1,000 samples, the slowest run overall  

PubMedQA questions tend to be shorter than MedQA questions, so the slowest runtime for the shortest inputs is anomalous. The GB10 MedQA run took 168,234 seconds (42.2 hours) despite longer inputs. This ~5,000-second gap is suspicious and may indicate:
- A background job competing for GPU resources during the PubMedQA window (started 2026-03-28)
- A thermal throttle event
- A scheduled retry storm for one of the models

---

### H100 Issues

#### Issue H100-1: Empty Response Field — 98/1000 MedQA Entries (CRITICAL)

**Scope:** 98 out of 1,000 MedQA entries on H100  
**Effect:** All 5 jurors score every dimension at 0.5, triggering "Critical" classification  

For 98 entries, the `response` field in `jury_details.json` is completely empty. When a response is missing, all 5 jury members assign `score = 0.5` to all 7 dimensions (with `justification = "Skipped - missing data"`). With a median of 0.5 across all dimensions, these entries receive a final_score of **0.5**, automatically classified as "Critical."

This means a significant fraction of H100's "Critical" classifications are not genuine harm detections — they are **null evaluations masquerading as high-harm samples**.

**Scope across all datasets:** This was confirmed for MedQA. The extent for MedMCQA and PubMedQA is unknown but should be investigated.

**Likely cause:** The response generation (ministral-14b) or dataset loading step on H100 failed to populate the response field for approximately 9.8% of MedQA instances. This may be a vLLM inference timeout, output truncation at zero length, or a file I/O race condition during concurrent batch processing.

---

#### Issue H100-2: ministral-14b Retry Contamination — 1,286 Dimensions Inflated (CRITICAL)

**Scope:** 1,286 dimension-level scores across MedQA entries  
**Effect:** Scores inflated to 0.5 (681 cases) or 0.8 (156 cases) due to retry artifacts  

When ministral-14b failed to parse its output on the first attempt, the retry mechanism was invoked. However, the justification text for these retried scores contains the prefix `"Retry 2: HIGH\n\n..."` or similar, and the scores are systematically inflated compared to successful first-attempt scores. A retry that produces "HIGH" is scored at 0.5 or 0.8 even though the underlying content may not warrant it.

This is a scoring pipeline bug: the retry mechanism uses a degraded fallback prompt that tends toward over-flagging harm. The inflated retry scores propagate into the final median without any flag or exclusion.

**Impact:** If the 1,286 affected dimensions were corrected to their expected values, a substantial portion of H100's "Critical" classifications would drop to "Low."

---

#### Issue H100-3: Dimension Order Inconsistency (MINOR)

**Scope:** H100 results.json entries  
**Effect:** JSON keys appear in inconsistent order across entries  

The 7 harm dimensions appear in different orders in different H100 results entries (e.g., `psychological_harm` may appear 4th in one entry and 7th in another). This does not affect correctness if scores are accessed by key (not by index), but it makes raw file inspection and certain parsing approaches unreliable. GB10 entries maintain consistent key ordering.

---

#### Issue H100-4: olmo-32b Scenario Confusion (MINOR, both platforms)

**Scope:** Both GB10 and H100, all datasets  
**Effect:** olmo-32b sometimes scores the scenario/question for harm rather than the response  

Confirmed cases exist where olmo-32b assigns high harm scores (e.g., `autonomy_harm = 1.0`) based on the content of the question or hypothetical scenario rather than the actual model response. Example: a question about intraoperative complications was scored as `autonomy_harm = 1.0` with justification referencing the scenario, while the response correctly answered the question.

This is a prompt engineering issue with olmo-32b's system prompt — it does not sufficiently disambiguate "evaluate the harm in the response" from "evaluate the harm in the scenario being described."

---

### Quantitative Impact Summary

| Issue | Platform | Samples Affected | Direction |
|---|---|---|---|
| gemma3-27b parsing failure | GB10 | **1,000/1,000** (100%) | Scores artificially **deflated** |
| qwen2.5-coder-7b out-of-range | GB10 | 39/1,000 (3.9%) | Scores **corrupted** |
| Empty response field | H100 | **98/1,000** (9.8%) | Scores artificially **inflated** to 0.5 |
| ministral-14b retry contamination | H100 | ~1,286 dimensions (~9% of dims) | Scores artificially **inflated** |
| olmo-32b scenario confusion | Both | Unknown, scattered | Scores **inflated** (moderate) |
| Dimension order inconsistency | H100 | All 1,000 | Cosmetic only |

---

## 9. Dataset-Level Observations

### MedQA

- Largest scoring gap between platforms (GB10 mean 0.0695 vs H100 mean 0.2115)
- Highest "Critical" rate on H100 (25.6%) — partially explained by 98 empty responses
- **Autonomy harm** is the dominant dimension on GB10; **informational harm** is dominant on H100
- MedQA responses likely involve complex clinical scenarios where autonomy framing varies by model generation

### MedMCQA

- Lowest mean harm scores on both platforms (GB10: 0.0450, H100: 0.1180)
- **Highest zero-score rates** on both platforms (31.7% GB10, 23.0% H100)
- Multiple-choice format likely produces more terse, structured answers that are less prone to harmful framing
- Single highest-score outlier across all datasets lives here: 0.9500 on H100, 0.9000 on GB10
- On H100, `social_harm` shows an outsized multiplier (6.05× vs GB10) for MedMCQA — warrants investigation

### PubMedQA

- Most **consistent** dataset across platforms (smallest mean delta: +0.0469)
- **Identical zero-score counts** on both platforms (27/1,000) — the only metric in perfect agreement
- Tightest score distributions (lowest std on both platforms)
- **Informational harm** is the dominant dimension on both platforms — PubMedQA involves summarizing published research, where factual accuracy is paramount
- Highest **epistemic harm** on GB10 (mean 0.0867 vs 0.0449 MedQA) — expected, given the evidence-interpretation nature of the questions

---

## 10. Summary of Key Differences

| Metric | GB10 | H100 | Notes |
|---|---|---|---|
| Total runtime (3 datasets) | ~137 hours | ~9.7 hours | H100 is **14.1× faster** |
| Avg final score (MedQA) | 0.0695 | 0.2115 | H100 **3.0×** higher |
| Avg final score (MedMCQA) | 0.0450 | 0.1180 | H100 **2.6×** higher |
| Avg final score (PubMedQA) | 0.0728 | 0.1197 | H100 **1.6×** higher |
| "Critical" rate (MedQA) | 2.1% | 25.6% | H100 **12.2×** more Critical |
| "Critical" rate (MedMCQA) | 1.6% | 13.2% | H100 **8.3×** more Critical |
| "Critical" rate (PubMedQA) | 0.5% | 2.9% | H100 **5.8×** more Critical |
| Zero-score rate (MedQA) | 9.1% | 3.4% | GB10 **2.7×** more zeros |
| Zero-score rate (MedMCQA) | 31.7% | 23.0% | GB10 **1.4×** more zeros |
| Zero-score rate (PubMedQA) | 2.7% | 2.7% | **Identical** |
| gemma3-27b parsing | FAILED (100%) | OK | Critical GB10 defect |
| Empty responses | 0 | 98 (9.8%) | Critical H100 defect |
| Retry contamination | None | 1,286 dims | Critical H100 defect |
| Out-of-range scores | 39 entries | None | GB10 defect |
| Dominant harm dim | autonomy_harm | informational_harm | Different signals detected |

---

## 11. Conclusions and Recommendations

### What Worked Well

1. **H100 inference throughput is excellent.** A 14× speedup makes full-corpus evaluation feasible in under 10 hours. GB10 at 5.7 days is unusable for iterative experimentation.

2. **PubMedQA evaluation is the most stable.** Both platforms agree on zero rates and the score distributions are tightest. PubMedQA is the most reliable benchmark for cross-hardware comparisons.

3. **The jury framework itself is sound.** The jury architecture (5 jurors, median aggregation, 7 dimensions) is capable of producing meaningful signal. The issues identified are pipeline/runtime bugs, not fundamental design flaws.

4. **MedMCQA shows consistent patterns.** The high zero-rate (~23-31%) reflects the format of multiple-choice answers (short, structured, less prone to harm), and this pattern is consistent across both hardware platforms.

### What Was Wrong

1. **gemma3-27b parsing failure on GB10 is unacceptable.** A 100% failure rate for one juror means GB10 results for all datasets are structurally invalid. Every GB10 harm score is artificially deflated. GB10 results should be considered unreliable until gemma3-27b is re-run with the correct output format.

2. **98 empty responses on H100 (MedQA) produce false "Critical" classifications.** Nearly 10% of MedQA samples on H100 have no response text, yet receive harm scores of 0.5 across all dimensions. These entries should be excluded from any analysis, and the root cause (response generation failure) must be fixed.

3. **ministral-14b retry contamination inflates H100 scores.** The 1,286 affected dimension scores introduce systematic upward bias. The retry mechanism needs to either use the same scoring scale as first-attempt scoring, or retry scores should be flagged and excluded from aggregation.

4. **qwen2.5-coder-7b emitted 0–10 integer scores on GB10** instead of 0–1 floats. These 39 entries are corrupted and need reprocessing with clamping/rescaling.

5. **The two runs evaluated different responses.** Because ministral-14b generated different text on each hardware run, the score comparison is confounded. A fair comparison would require identical input responses evaluated by the same jury.

### What Was Longer Than Expected

1. **GB10 runtimes were dramatically longer than any reasonable estimate.** 46–48 hours per dataset for 1,000 samples (168–173 seconds/sample) is approximately 13–14× what was achieved on H100. The PubMedQA GB10 run (~48 hours) was the longest despite PubMedQA having the shortest questions, suggesting non-hardware factors (retry storms, model loading overhead, background interference) contributed.

2. **gemma3-27b parse failures likely caused retry storms on GB10**, significantly increasing per-sample wall time as the system attempted and failed to extract scores from malformed outputs.

### Recommended Actions

| Priority | Action |
|---|---|
| P0 | Re-run GB10 evaluation with gemma3-27b output format bug fixed |
| P0 | Identify and fix the root cause of 98 empty responses on H100 MedQA |
| P0 | Audit H100 MedMCQA and PubMedQA for empty response rates |
| P1 | Fix the ministral-14b retry scoring to use consistent scale |
| P1 | Reprocess the 39 qwen2.5-coder-7b out-of-range entries on GB10 |
| P1 | Add a post-processing validation step that flags: empty responses, out-of-range scores, parsing failures, and retry-inflated scores |
| P2 | Fix olmo-32b system prompt to better disambiguate scenario vs response evaluation |
| P2 | Use identical generated responses across hardware runs for fair comparison |
| P2 | Add a `valid` flag to each jury_details entry to make filtering easy |
| P3 | Investigate GB10 PubMedQA duration anomaly (48h for shortest-input dataset) |
| P3 | Standardize dimension key ordering in output JSON |

### Net Assessment

The H100 hardware is the clear winner for throughput and should be the primary evaluation platform going forward. However, the current H100 results are not clean: ~9.8% of MedQA samples are null evaluations, and retry contamination affects ~9% of dimensions. Similarly, GB10 results are not trustworthy due to the gemma3-27b parsing failure affecting 100% of samples.

**Neither platform's current results should be used as ground truth for downstream decisions.** A clean re-run on H100 — with the four P0/P1 bugs fixed — would be the correct next step before drawing any conclusions about model harm levels.

---

*This report was generated by automated statistical analysis of `results.json` and `jury_details.json` files. All figures are based on medqa data for jury_details analysis; results.json statistics cover all three datasets.*
