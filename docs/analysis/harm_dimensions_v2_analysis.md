# Harm Dimensions v2 — Deep Analysis

**Run date:** 2026-03-23 to 2026-03-24
**Datasets:** PubMedQA, MedQA, MedMCQA (1 000 samples each)
**Jury:** ministral-14b, gemma3-27b, nemotron-30b, olmo-32b, qwen2.5-coder-7b
**Aggregation:** median | **Critical threshold:** 0.4
**Hardware:** 2× H100 (190 GB total VRAM)

---

## 1. Run Overview

| Dataset   | Samples | Duration | s/sample | Status         |
|-----------|---------|----------|----------|----------------|
| PubMedQA  | 1 000   | 3h 31m   | 12.7 s   | Clean, 0 errors|
| MedQA     | 1 000   | 3h 33m   | 12.8 s   | Clean, 0 errors|
| MedMCQA   | 1 000   | 3h 04m   | 11.1 s   | Clean, 0 errors|

---

## 2. Juror Scoring Profiles

### 2.1 Scoring Style and Granularity

The most revealing characteristic of each juror is not the mean score but the **discrete values they use** and the **entropy** of their scoring distribution.

| Juror              | Unique values | Entropy (bits) | Top-3 values cover | Scoring style        |
|--------------------|---------------|----------------|--------------------|----------------------|
| nemotron-30b       | 23            | 2.420          | 80.0%              | Continuous, graduated|
| gemma3-27b         | 12            | 2.359          | 76.9%              | Spread, 4-level      |
| ministral-14b      | 12            | 1.797          | 87.8%              | Discrete, 3-level    |
| qwen2.5-coder-7b   | 7             | 1.648          | 89.1%              | Coarse, 4 values     |
| olmo-32b           | 7             | 1.337          | 98.6%              | Near-binary          |

**nemotron-30b** is the most granular scorer: uses 23 distinct values including 0.05, 0.15, 0.25, 0.35, 0.65 — it reasons in fine increments. **olmo-32b** is near-binary: 98.6% of scores fall on just three values (0.0 / 0.2 / 0.5), meaning it effectively classifies rather than scores.

Score distribution by value (top entries, combined across all datasets and dimensions):

**ministral-14b:** 0.0 (58.5%), 0.2 (19.4%), 0.5 (9.9%), 0.1 (7.1%), 0.3 (3.5%), 0.8 (1.2%)
**gemma3-27b:** 0.0 (29.3%), 0.1 (24.8%), 0.3 (22.8%), 0.2 (12.5%), 0.5 (7.9%), 0.6 (1.3%)
**nemotron-30b:** 0.1 (47.0%), 0.0 (26.2%), 0.15 (6.9%), 0.5 (5.3%), 0.05 (2.2%), 0.25 (2.1%)
**olmo-32b:** 0.0 (60.9%), 0.2 (30.7%), 0.5 (6.9%), 0.7 (0.7%), others < 0.5%
**qwen2.5-coder-7b:** 0.0 (57.3%), 0.2 (19.5%), 0.3 (12.3%), 0.5 (10.9%), others < 0.2%

---

### 2.2 Per-Juror Overall Means

Combined mean score across all dimensions, all three datasets:

| Juror              | PubMedQA | MedQA  | MedMCQA | Combined | Std across datasets |
|--------------------|----------|--------|---------|----------|---------------------|
| gemma3-27b         | 0.1524   | 0.2384 | 0.1359  | 0.1756   | 0.0449              |
| nemotron-30b       | 0.1057   | 0.1852 | 0.1200  | 0.1370   | 0.0346              |
| qwen2.5-coder-7b   | 0.1325   | 0.1725 | 0.0868  | 0.1306   | 0.0350              |
| ministral-14b      | 0.1007   | 0.1825 | 0.0720  | 0.1184   | 0.0468              |
| olmo-32b           | 0.1013   | 0.1182 | 0.0909  | 0.1035   | 0.0113              |

**gemma3-27b** is the most stringent juror by a clear margin (combined 0.176 vs jury average ~0.130). **olmo-32b** is the most lenient and by far the most dataset-stable (std 0.011 vs next-best 0.035), which reflects its near-binary scoring: it does not differentiate much by content type.

**ministral-14b** shows the largest swing between MedQA (0.183) and MedMCQA (0.072) despite being the response generator — suggesting its juror role is not trivially correlated with leniency toward its own outputs.

---

### 2.3 Dimension Signatures

Each juror has a distinct dimension profile. The table below shows each juror's mean for each dimension relative to the jury-wide mean for that dimension (ratio > 1.15 = systematic over-scorer, < 0.85 = under-scorer):

#### ministral-14b — Calibrated on informational/autonomy; strongly under-scores economic and psychological

| Dimension          | Juror mean | Jury mean | Ratio  |
|--------------------|------------|-----------|--------|
| informational_harm | 0.2072     | 0.1948    | 1.06×  |
| autonomy_harm      | 0.1857     | 0.1838    | 1.01×  |
| privacy_harm       | 0.0403     | 0.0394    | 1.02×  |
| epistemic_harm     | 0.1298     | 0.1451    | 0.89×  |
| psychological_harm | 0.1145     | 0.1547    | **0.74×** |
| economic_harm      | 0.0789     | 0.1421    | **0.56×** |

Ministral treats harm as primarily about factual accuracy (informational) and patient agency (autonomy) while consistently downweighting the financial and psychological dimensions. This is notable because it is also the response generator — it scores its own responses conservatively on the dimensions most aligned with medical advice risk.

#### gemma3-27b — Global over-scorer; most amplified on social and epistemic

| Dimension          | Juror mean | Jury mean | Ratio  |
|--------------------|------------|-----------|--------|
| social_harm        | 0.1113     | 0.0711    | **1.56×** |
| epistemic_harm     | 0.2053     | 0.1451    | **1.42×** |
| psychological_harm | 0.2126     | 0.1547    | **1.37×** |
| autonomy_harm      | 0.2396     | 0.1838    | **1.30×** |
| economic_harm      | 0.1814     | 0.1421    | **1.28×** |
| informational_harm | 0.2406     | 0.1948    | **1.23×** |
| privacy_harm       | 0.0381     | 0.0394    | 0.97×  |

Gemma over-scores on 6 of 7 dimensions — the only exception is privacy (near-neutral). This is a structurally strict juror with no blind spots toward the dimensions that other jurors downweight. Its anomalously high social_harm score (1.56×) suggests it interprets demographic framing more broadly than peers.

#### nemotron-30b — Over-scores on informational and privacy; otherwise near-neutral

| Dimension          | Juror mean | Jury mean | Ratio  |
|--------------------|------------|-----------|--------|
| privacy_harm       | 0.0547     | 0.0394    | **1.39×** |
| informational_harm | 0.2509     | 0.1948    | **1.29×** |
| social_harm        | 0.0745     | 0.0711    | 1.05×  |
| epistemic_harm     | 0.1414     | 0.1451    | 0.97×  |
| autonomy_harm      | 0.1764     | 0.1838    | 0.96×  |
| economic_harm      | 0.1268     | 0.1421    | 0.89×  |
| psychological_harm | 0.1343     | 0.1547    | 0.87×  |

Nemotron has the highest informational_harm mean of any juror in the combined set (0.251 vs jury mean 0.195), reflecting heightened sensitivity to factual errors. Its privacy score (1.39× jury mean) is the highest among all jurors — notably, nemotron is the only model that systematically flags privacy concerns where others score near zero.

#### olmo-32b — Systematic under-scorer on social, psychological, epistemic

| Dimension          | Juror mean | Jury mean | Ratio  |
|--------------------|------------|-----------|--------|
| informational_harm | 0.1861     | 0.1948    | 0.96×  |
| economic_harm      | 0.1253     | 0.1421    | 0.88×  |
| autonomy_harm      | 0.1487     | 0.1838    | **0.81×** |
| epistemic_harm     | 0.1014     | 0.1451    | **0.70×** |
| privacy_harm       | 0.0274     | 0.0394    | **0.70×** |
| psychological_harm | 0.0966     | 0.1547    | **0.62×** |
| social_harm        | 0.0388     | 0.0711    | **0.55×** |

Olmo under-scores on 5 of 7 dimensions. Its near-binary scoring (0/0.2/0.5) means it cannot express nuanced harm — it effectively classifies responses as either clean, mildly concerning, or flagged. The systematic under-scoring on social and psychological dimensions reduces its influence on the jury median for these dimensions. It is, however, unusually stable (std 0.011 across datasets) and its discriminative ability on Critical cases is the highest in MedQA (4.31× ratio on Critical vs Low, see §4).

#### qwen2.5-coder-7b — Strong over-scorer on psychological and economic; major under-scorer on informational

| Dimension          | Juror mean | Jury mean | Ratio  |
|--------------------|------------|-----------|--------|
| economic_harm      | 0.1982     | 0.1421    | **1.39×** |
| psychological_harm | 0.2153     | 0.1547    | **1.39×** |
| epistemic_harm     | 0.1475     | 0.1451    | 1.02×  |
| privacy_harm       | 0.0365     | 0.0394    | 0.93×  |
| autonomy_harm      | 0.1685     | 0.1838    | 0.92×  |
| social_harm        | 0.0590     | 0.0711    | **0.83×** |
| informational_harm | 0.0892     | 0.1948    | **0.46×** |

Qwen's dimension profile is the most distinctive: it severely under-scores informational_harm (0.46× jury mean), the dimension all other jurors score at or above the average, while simultaneously over-scoring psychological and economic harm. Being a code-specialized model, it may parse medical factual claims differently and instead focus on downstream harm pathways (psychological impact, financial cost).

---

### 2.4 Juror Bias vs Jury Median

Percentage of cases where each juror scores above or below the jury median per dimension:

| Juror              | info↑ | info↓ | psyc↑ | psyc↓ | auto↑ | auto↓ | econ↑ | econ↓ | epis↑ | epis↓ |
|--------------------|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| ministral-14b      | 24.1% |  9.2% |  6.4% | 37.7% | 15.7% | 19.0% |  2.9% | 43.4% | 10.8% | 19.5% |
| gemma3-27b         | 38.7% |  6.6% | 49.5% |  1.7% | 58.6% |  2.7% | 36.1% |  1.7% | 54.1% |  2.5% |
| nemotron-30b       | 38.4% | 21.0% | 23.1% | 29.1% | 29.6% | 30.4% | 18.5% | 31.8% | 36.1% | 22.7% |
| olmo-32b           | 21.0% | 26.2% |  6.6% | 38.0% | 10.6% | 23.4% |  9.9% | 19.9% | 11.1% | 28.9% |
| qwen2.5-coder-7b   |  8.5% | 59.9% | 55.3% |  4.6% | 17.2% | 23.6% | 46.0% |  1.9% | 27.8% | 24.3% |

Key patterns:
- **gemma3-27b** is above the median more than below on every single dimension — uniquely biased upward across the board.
- **qwen2.5-coder-7b** is below the informational median in 59.9% of cases while above the psychological median in 55.3% — a strong directional reversal relative to all other jurors.
- **ministral-14b** is below economic median 43.4% of the time and below psychological median 37.7% — consistent systematic discounting of these dimensions.
- **olmo-32b** is below the psychological median 38.0% of cases, reinforcing its pattern of under-scoring human-impact dimensions.
- **nemotron-30b** is the most balanced — its above/below rates are closest to symmetric across dimensions.

---

### 2.5 Juror Influence on the Final Score

#### Deciding vote frequency (juror score = jury median)

| Juror              | PubMedQA | MedQA | MedMCQA | Combined |
|--------------------|----------|-------|---------|----------|
| ministral-14b      | 68.2%    | 62.7% | 80.6%   | **70.5%**|
| olmo-32b           | 67.9%    | 62.8% | 79.4%   | **70.0%**|
| qwen2.5-coder-7b   | 47.0%    | 57.5% | 70.7%   | 58.4%    |
| gemma3-27b         | 55.7%    | 50.3% | 64.3%   | 56.8%    |
| nemotron-30b       | 47.0%    | 51.9% | 57.2%   | 52.0%    |

**ministral-14b and olmo-32b are the median anchors** — their score lands on the jury median in ~70% of all dimension evaluations. This happens for different reasons: ministral scores conservatively on most dimensions, landing in the middle of the distribution; olmo uses only three values, making collisions with the median more likely mechanically. Despite being the most extreme scorer, **gemma3-27b** is not the least influential — it lands on the median 56.8% of the time because the jury as a whole has moved upward.

#### Lone outlier frequency (furthest from median, >0.05 distance, unique position)

| Juror              | PubMedQA | MedQA | MedMCQA | Combined |
|--------------------|----------|-------|---------|----------|
| qwen2.5-coder-7b   | 19.7%    |  9.8% |  9.1%   | **12.9%**|
| gemma3-27b         |  9.4%    | 15.2% | 11.4%   | **12.0%**|
| nemotron-30b       |  9.1%    | 10.5% | 13.9%   | **11.2%**|
| ministral-14b      |  8.1%    | 13.1% |  3.9%   |  8.4%    |
| olmo-32b           |  2.0%    |  4.4% |  2.3%   | **2.9%** |

**qwen2.5-coder-7b** is the lone outlier most often overall (12.9%), driven by PubMedQA where it dissents in nearly 1 in 5 dimension evaluations. **olmo-32b** is almost never the lone outlier (2.9%) — its coarse scoring clusters with others by chance. **gemma3-27b** is the lone outlier most often in MedQA (15.2%), where its strict stance on clinical content diverges most from peers.

---

### 2.6 Pairwise Juror Agreement

Pearson correlation between juror score vectors (all dimensions, all datasets):

|                    | ministral | gemma3   | nemotron | olmo     | qwen     |
|--------------------|-----------|----------|----------|----------|----------|
| **ministral-14b**  | 1.000     | 0.602    | 0.560    | 0.574    | 0.460    |
| **gemma3-27b**     | 0.602     | 1.000    | 0.609    | 0.633    | 0.561    |
| **nemotron-30b**   | 0.560     | 0.609    | 1.000    | 0.594    | 0.430    |
| **olmo-32b**       | 0.574     | 0.633    | 0.594    | 1.000    | 0.554    |
| **qwen2.5-coder**  | 0.460     | 0.561    | 0.430    | 0.554    | 1.000    |

Mean absolute difference between juror pairs:

|                    | ministral | gemma3   | nemotron | olmo     | qwen     |
|--------------------|-----------|----------|----------|----------|----------|
| **ministral-14b**  | 0.000     | 0.103    | 0.097    | 0.072    | 0.109    |
| **gemma3-27b**     | 0.103     | 0.000    | 0.092    | 0.097    | 0.105    |
| **nemotron-30b**   | 0.097     | 0.092    | 0.000    | 0.092    | 0.118    |
| **olmo-32b**       | 0.072     | 0.097    | 0.092    | 0.000    | 0.085    |
| **qwen2.5-coder**  | 0.109     | 0.105    | 0.118    | 0.085    | 0.000    |

**gemma3-27b and olmo-32b** have the highest correlation (r=0.633) despite being at opposite ends of the severity spectrum — they agree on *which* items are high-harm even if their absolute scores differ. **qwen2.5-coder-7b** has the lowest correlation with nemotron-30b (r=0.430), consistent with their opposite informational_harm signatures (qwen: 0.46× jury mean; nemotron: 1.29× jury mean). **ministral-14b and qwen2.5-coder-7b** have the greatest mean absolute distance (0.109), reflecting their opposite dimension specializations.

---

### 2.7 Reasoning Depth (Justification Length)

| Juror              | Mean chars | Median | Std  |
|--------------------|------------|--------|------|
| qwen2.5-coder-7b   | 351        | 356    | 123  |
| ministral-14b      | 335        | 347    | 159  |
| gemma3-27b         | 314        | 308    | 106  |
| olmo-32b           | 290        | 296    |  84  |
| nemotron-30b       | 135        | 163    | 106  |

**nemotron-30b writes the shortest justifications** (mean 135 chars, about half of the others) despite having the most granular scoring. Its justifications are terse and verdict-like. **olmo-32b** has the lowest variance in justification length (std 84) — consistent with its near-binary, classification-style output. **ministral-14b** has the most variable justifications (std 159), suggesting it elaborates more when it detects meaningful harm.

---

## 3. Discriminative Ability: Critical vs Low Samples

For each juror, the ratio of mean score on Critical samples (final score ≥ 0.4) to mean score on Low samples:

| Juror              | PubMedQA ratio | MedQA ratio | MedMCQA ratio | Mean ratio |
|--------------------|----------------|-------------|---------------|------------|
| ministral-14b      | 2.12×          | 2.64×       | **6.59×**     | 3.78×      |
| olmo-32b           | 1.74×          | **4.31×**   | **4.96×**     | 3.67×      |
| qwen2.5-coder-7b   | **2.20×**      | 2.59×       | 4.92×         | 3.24×      |
| gemma3-27b         | 1.72×          | 2.11×       | 3.50×         | 2.44×      |
| nemotron-30b       | 1.62×          | 2.44×       | 3.48×         | 2.51×      |

**ministral-14b and olmo-32b have the strongest discrimination** between Low and Critical samples (3.78× and 3.67× mean ratio). This is especially pronounced in MedMCQA where ministral achieves a 6.59× ratio — suggesting that despite scoring conservatively overall, it reliably escalates when genuinely problematic content is present.

**gemma3-27b** has the weakest discrimination (2.44×). Because it scores high across the board, the signal-to-noise ratio between Critical and Low samples is compressed. Its absolute Critical scores are the highest, but the contrast with its already-elevated Low scores is relatively small.

This creates an important asymmetry: **gemma identifies the most critical cases in absolute terms but is the least efficient discriminator**; **ministral and olmo are the most efficient discriminators** but from a lower baseline.

---

## 4. Per-Dimension Analysis (Supporting Reference)

Dimension means (all 1 000 samples per dataset):

### 4.1 PubMedQA

| Dimension           | Mean   | Std    | Max    | % > 0 |
|---------------------|--------|--------|--------|-------|
| informational_harm  | 0.1638 | 0.0943 | 0.7000 | 89.8% |
| autonomy_harm       | 0.1566 | 0.0838 | 0.5000 | 85.5% |
| epistemic_harm      | 0.1367 | 0.0864 | 0.5000 | 82.7% |
| psychological_harm  | 0.1299 | 0.0892 | 0.5000 | 78.3% |
| economic_harm       | 0.1245 | 0.0879 | 0.5000 | 76.4% |
| social_harm         | 0.0416 | 0.0839 | 0.5000 | 25.4% |
| privacy_harm        | 0.0013 | 0.0144 | 0.3000 |  1.0% |

### 4.2 MedQA

| Dimension           | Mean   | Std    | Max    | % > 0 |
|---------------------|--------|--------|--------|-------|
| autonomy_harm       | 0.2357 | 0.1404 | 0.8000 | 90.3% |
| informational_harm  | 0.2172 | 0.1811 | 0.8000 | 81.4% |
| psychological_harm  | 0.1988 | 0.1355 | 0.5000 | 90.5% |
| economic_harm       | 0.1874 | 0.1444 | 0.5000 | 80.8% |
| epistemic_harm      | 0.1478 | 0.1579 | 0.7000 | 68.7% |
| social_harm         | 0.0776 | 0.1554 | 0.7000 | 30.6% |
| privacy_harm        | 0.0532 | 0.1528 | 0.7000 | 11.6% |

### 4.3 MedMCQA

| Dimension           | Mean   | Std    | Max    | % > 0 |
|---------------------|--------|--------|--------|-------|
| informational_harm  | 0.1549 | 0.1814 | 0.9000 | 62.2% |
| autonomy_harm       | 0.1080 | 0.1346 | 0.5000 | 52.4% |
| psychological_harm  | 0.0957 | 0.1324 | 0.5000 | 48.4% |
| epistemic_harm      | 0.0863 | 0.1383 | 0.9500 | 44.1% |
| economic_harm       | 0.0888 | 0.1364 | 0.5000 | 40.2% |
| social_harm         | 0.0319 | 0.1165 | 0.5000 |  8.5% |
| privacy_harm        | 0.0283 | 0.1150 | 0.5000 |  5.8% |

---

## 5. Dataset-Level Context (Summary)

| Dataset  | Mean score | Critical % | Dominant dimension | Notes                                     |
|----------|------------|------------|---------------------|-------------------------------------------|
| PubMedQA | 0.1193     | 2.9%       | informational       | Abstract-level research, tight distribution|
| MedQA    | 0.2153     | 26.6%      | autonomy            | Clinical vignettes, highest risk           |
| MedMCQA  | 0.1249     | 14.7%      | informational       | Bimodal: many clean, some high-harm        |

---

## 6. Dimension Correlations (New Run)

### PubMedQA
| Pair                          | r     |
|-------------------------------|-------|
| informational ↔ epistemic     | 0.633 |
| autonomy ↔ economic           | 0.560 |
| autonomy ↔ epistemic          | 0.507 |
| psychological ↔ autonomy      | 0.457 |

### MedQA
| Pair                          | r     |
|-------------------------------|-------|
| social ↔ privacy              | 0.924 |
| psychological ↔ economic      | 0.792 |
| social ↔ psychological        | 0.782 |
| social ↔ epistemic            | 0.758 |

### MedMCQA
| Pair                          | r     |
|-------------------------------|-------|
| social ↔ privacy              | 0.979 |
| autonomy ↔ economic           | 0.844 |
| psychological ↔ economic      | 0.829 |
| psychological ↔ autonomy      | 0.786 |

The **social–privacy** near-perfect correlation in MedQA (r=0.924) and MedMCQA (r=0.979) means these two dimensions are effectively a single signal: whenever a response is flagged for social bias, it is also flagged for privacy exposure, and vice versa.

---

## 7. Cross-Dataset Consistency

Dimension ranking stability per juror (Kendall τ between datasets):

| Juror              | pub↔med | pub↔mcq | med↔mcq | Interpretation                           |
|--------------------|---------|---------|---------|------------------------------------------|
| gemma3-27b         | 0.905   | 0.905   | **1.000** | Perfect ranking stability med↔mcq       |
| qwen2.5-coder-7b   | 0.905   | 0.905   | **1.000** | Perfect ranking stability med↔mcq       |
| ministral-14b      | 0.714   | 0.905   | 0.810   | Stable                                   |
| olmo-32b           | 0.619   | 0.905   | 0.714   | Moderate pub↔med variation               |
| nemotron-30b       | 0.714   | 0.905   | 0.619   | Moderate med↔mcq variation               |

**gemma3-27b and qwen2.5-coder-7b** maintain perfectly consistent dimension rankings between MedQA and MedMCQA — their internal hierarchy of which dimensions matter most is dataset-invariant. **nemotron-30b** shows the most unstable ranking between MedQA and MedMCQA (τ=0.619), likely because its high informational sensitivity interacts differently with exam-style vs clinical-vignette content.

---

## 8. Summary: Model Characterization

| Juror              | Severity  | Specialization                        | Discriminability | Stability | Scoring style   |
|--------------------|-----------|---------------------------------------|------------------|-----------|-----------------|
| gemma3-27b         | Highest   | All dims; esp. social, epistemic      | Low (2.4×)       | High      | 4-level spread  |
| nemotron-30b       | Medium    | Informational, privacy                | Medium (2.5×)    | High      | Continuous/fine |
| qwen2.5-coder-7b   | Medium    | Psychological, economic; anti-info    | Medium (3.2×)    | High      | 4-value coarse  |
| ministral-14b      | Low-Med   | Informational, autonomy; anti-econ    | Highest (3.8×)   | Low       | 3-level discrete|
| olmo-32b           | Lowest    | Informational (only); under-scores rest | High (3.7×)    | Highest   | Near-binary     |

**Design implication:** The jury achieves good coverage through complementary weaknesses. Gemma provides a high sensitivity floor; ministral and olmo provide the sharpest signal at the critical threshold; nemotron catches privacy concerns that others miss; qwen catches psychological and economic harms that ministral systematically discounts. The median aggregation effectively suppresses gemma's global inflation while retaining its ability to swing borderline cases above the critical threshold when supported by at least two other jurors.
