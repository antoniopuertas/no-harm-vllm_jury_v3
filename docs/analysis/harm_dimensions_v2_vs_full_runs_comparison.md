# Comparison: harm_dimensions_v2 vs full_runs

**Old run:** `data/results/vllm/full_runs/` — completed 2026-02-27 to 2026-02-28
**New run:** `data/results/vllm/harm_dimensions_v2/` — completed 2026-03-23 to 2026-03-24
**Jury:** identical (ministral-14b, gemma3-27b, nemotron-30b, olmo-32b, qwen2.5-coder-7b)
**Aggregation:** median | **Critical threshold:** 0.4 | **Hardware:** 2× H100

---

## 1. Sample Coverage

| Dataset  | Old samples | New samples | Common IDs |
|----------|-------------|-------------|------------|
| PubMedQA | 1 000       | 1 000       | 1 000      |
| MedQA    | 1 273       | 1 000       | 1 000      |
| MedMCQA  | 1 000       | 1 000       | 1 000      |

MedQA was reduced from 1 273 to 1 000 samples. The 1 000 common IDs are used for all per-sample delta comparisons below.

---

## 2. Runtime & Throughput

| Dataset  | Old duration | Old s/sample | New duration | New s/sample | Speedup |
|----------|-------------|--------------|-------------|--------------|---------|
| PubMedQA | 9.96 h      | 35.8 s       | 3.52 h      | 12.7 s       | **2.83×** |
| MedQA    | 14.90 h     | 42.2 s       | 3.56 h      | 12.8 s       | **4.19×** |
| MedMCQA  | 10.37 h     | 37.3 s       | 3.07 h      | 11.1 s       | **3.38×** |

The new pipeline is **2.8–4.2× faster**, reducing total compute from ~35 GPU-hours to ~10 GPU-hours across all three datasets. The biggest gain is in MedQA, where the old run had the highest per-sample latency (42 s), likely due to the instruct-variant olmo-32b being replaced — the old olmo (think variant) generated extended chain-of-thought that bloated token counts and triggered high retry rates.

---

## 3. Final Score Drift (new − old)

For the common set of instance IDs:

| Dataset  | Mean delta | Median delta | Std of delta | Increased | Decreased | Same |
|----------|------------|--------------|--------------|-----------|-----------|------|
| PubMedQA | +0.0334    | +0.0300      | 0.0604       | 829 (82.9%) | 108 (10.8%) | 63 |
| MedQA    | +0.0409    | +0.0237      | 0.1299       | 656 (65.6%) | 143 (14.3%) | 201 |
| MedMCQA  | +0.0285    | +0.0050      | 0.1260       | 529 (52.9%) | 125 (12.5%) | 346 |

Scores are **systematically higher in the new run** across all datasets. PubMedQA shows the most consistent upward shift (82.9% of samples increased). MedMCQA has a lower median delta (+0.005) with a larger share of unchanged samples (346), indicating many low-harm items are stable while higher-harm items moved more.

---

## 4. Mean Final Score Comparison

| Dataset  | Old mean | New mean | Absolute delta | Relative change |
|----------|----------|----------|----------------|-----------------|
| PubMedQA | 0.0859   | 0.1193   | +0.0334        | +38.9%          |
| MedQA    | 0.1705   | 0.2153   | +0.0448        | +26.3%          |
| MedMCQA  | 0.0964   | 0.1249   | +0.0285        | +29.6%          |

---

## 5. Harm Category Reclassifications

### PubMedQA (1 000 common IDs)
| Transition      | Count |
|-----------------|-------|
| Low → Low       | 963   |
| Low → Critical  |  16   |
| Critical → Low  |   8   |
| Critical → Crit |  13   |

Net change: **+8 new Critical cases** (2.1% → 2.9%)

### MedQA (1 000 common IDs)
| Transition      | Count |
|-----------------|-------|
| Low → Low       | 710   |
| Low → Critical  |  82   |
| Critical → Low  |  24   |
| Critical → Crit | 184   |

Net change: **+58 new Critical cases** (19.8% → 26.6%). This is the largest reclassification, with 82 samples newly exceeding the 0.4 threshold, overwhelmingly driven by the autonomy_harm increase.

### MedMCQA (1 000 common IDs)
| Transition      | Count |
|-----------------|-------|
| Low → Low       | 838   |
| Low → Critical  |  44   |
| Critical → Low  |  15   |
| Critical → Crit | 103   |

Net change: **+29 new Critical cases** (11.8% → 14.7%)

---

## 6. Per-Dimension Delta (new − old, mean across common samples)

### PubMedQA

| Dimension           | Old mean | New mean | Delta   | Std of delta |
|---------------------|----------|----------|---------|--------------|
| autonomy_harm       | 0.0852   | 0.1566   | +0.0713 | 0.0662       |
| informational_harm  | 0.1118   | 0.1638   | +0.0520 | 0.0702       |
| epistemic_harm      | 0.0985   | 0.1367   | +0.0382 | 0.0626       |
| psychological_harm  | 0.1107   | 0.1299   | +0.0193 | 0.0613       |
| economic_harm       | 0.1100   | 0.1245   | +0.0145 | 0.0744       |
| social_harm         | 0.0289   | 0.0416   | +0.0127 | 0.0503       |
| privacy_harm        | 0.0016   | 0.0013   | -0.0003 | 0.0187       |

### MedQA

| Dimension           | Old mean | New mean | Delta   | Std of delta |
|---------------------|----------|----------|---------|--------------|
| autonomy_harm       | 0.1401   | 0.2357   | +0.0946 | 0.0928       |
| economic_harm       | 0.1549   | 0.1874   | +0.0317 | 0.0689       |
| informational_harm  | 0.1878   | 0.2172   | +0.0252 | 0.1145       |
| social_harm         | 0.0608   | 0.0776   | +0.0146 | 0.0500       |
| epistemic_harm      | 0.1250   | 0.1478   | +0.0194 | 0.0793       |
| psychological_harm  | 0.1862   | 0.1988   | +0.0119 | 0.0735       |
| privacy_harm        | 0.0498   | 0.0532   | +0.0011 | 0.0138       |

### MedMCQA

| Dimension           | Old mean | New mean | Delta   | Std of delta |
|---------------------|----------|----------|---------|--------------|
| autonomy_harm       | 0.0584   | 0.1080   | +0.0496 | 0.0752       |
| informational_harm  | 0.1157   | 0.1549   | +0.0393 | 0.1248       |
| economic_harm       | 0.0809   | 0.0888   | +0.0079 | 0.0703       |
| epistemic_harm      | 0.0772   | 0.0863   | +0.0091 | 0.0842       |
| psychological_harm  | 0.0877   | 0.0957   | +0.0080 | 0.0571       |
| social_harm         | 0.0311   | 0.0319   | +0.0008 | 0.0331       |
| privacy_harm        | 0.0287   | 0.0283   | -0.0004 | 0.0278       |

**The single most changed dimension across all datasets is `autonomy_harm`:**
- PubMedQA: +0.0713 (+83.7% relative)
- MedQA: +0.0946 (+67.5% relative)
- MedMCQA: +0.0496 (+84.9% relative)

`privacy_harm` is the only dimension that did not increase (negligible delta ≈ −0.0003), remaining near zero.

---

## 7. olmo-32b: Outlier Fix

In the old run, olmo-32b (think variant) produced extreme score outliers due to its extended chain-of-thought leaking into the scoring output:

| Dataset  | Old olmo overall_mean | Old olmo std | Anomalous samples |
|----------|-----------------------|--------------|-------------------|
| PubMedQA | 0.0210                | 0.1861       | —                 |
| MedQA    | 0.0825                | **0.7527**   | —                 |
| MedMCQA  | 0.0809                | **2.3390**   | 2 samples (scores 7.0, 194.8) |

The medmcqa old run had 2 samples with olmo privacy/epistemic scores of 7.0 and 194.8 — clearly parse failures where the model's CoT text was interpreted as the score.

In the new run (olmo instruct variant), these anomalies are eliminated:

| Dataset  | New olmo overall_mean | New olmo std |
|----------|-----------------------|--------------|
| PubMedQA | 0.1013                | 0.1133       |
| MedQA    | 0.1182                | 0.1759       |
| MedMCQA  | 0.0909                | 0.1664       |

Standard deviations are now consistent with other jurors. The switch from olmo think → olmo instruct also contributed substantially to the overall runtime speedup.

---

## 8. Inter-Juror Disagreement

| Dataset  | Old mean disagree | New mean disagree | Change   |
|----------|-------------------|-------------------|----------|
| PubMedQA | 0.0534            | 0.0480            | −10.1%   |
| MedQA    | 0.0664            | 0.0628            | −5.4%    |
| MedMCQA  | 0.0570            | 0.0422            | −26.0%   |

Disagreement decreased in all datasets. The largest improvement is in MedMCQA, where the old olmo outliers were inflating the disagreement metric. With clean olmo scores, the jury converges more reliably.

The old MedQA and MedMCQA runs also had extreme max disagreement values (3.85 and 11.13) driven entirely by the olmo parse failures; the new runs cap at 0.21 and 0.21 respectively.

---

## 9. Per-Juror Score Shift (old → new, overall mean)

### PubMedQA
| Juror              | Old mean | New mean | Delta   |
|--------------------|----------|----------|---------|
| ministral-14b      | 0.0647   | 0.1007   | +0.0360 |
| gemma3-27b         | 0.1174   | 0.1524   | +0.0350 |
| nemotron-30b       | 0.0994   | 0.1057   | +0.0063 |
| olmo-32b           | 0.0210   | 0.1013   | **+0.0803** |
| qwen2.5-coder-7b   | 0.1352   | 0.1325   | −0.0027 |

### MedQA
| Juror              | Old mean | New mean | Delta   |
|--------------------|----------|----------|---------|
| ministral-14b      | 0.1180   | 0.1825   | +0.0645 |
| gemma3-27b         | 0.2057   | 0.2384   | +0.0327 |
| nemotron-30b       | 0.1720   | 0.1852   | +0.0132 |
| olmo-32b           | 0.0825   | 0.1182   | +0.0357 |
| qwen2.5-coder-7b   | 0.1607   | 0.1725   | +0.0118 |

### MedMCQA
| Juror              | Old mean | New mean | Delta   |
|--------------------|----------|----------|---------|
| ministral-14b      | 0.0623   | 0.0720   | +0.0097 |
| gemma3-27b         | 0.1123   | 0.1359   | +0.0236 |
| nemotron-30b       | 0.1212   | 0.1200   | −0.0012 |
| olmo-32b           | 0.0809   | 0.0909   | +0.0100 |
| qwen2.5-coder-7b   | 0.0883   | 0.0868   | −0.0015 |

**olmo-32b** shows the largest gain in PubMedQA (+0.0803) — its old scores were suppressed by the think-variant parse failures. **ministral-14b** shows the largest gain in MedQA (+0.0645), suggesting the v2 prompt or scoring rubric change also affected how this juror evaluates clinical responses.

---

## 10. Summary of Key Differences

| Aspect                    | full_runs (old)                        | harm_dimensions_v2 (new)               |
|---------------------------|----------------------------------------|----------------------------------------|
| **Runtime**               | 10–15 h per dataset                    | 3–3.5 h per dataset (3–4× faster)     |
| **Final scores**          | Lower (mean 0.086–0.171)               | Higher (mean 0.119–0.215)              |
| **Critical rate**         | PubMedQA 2.1%, MedQA 19.8%, MCQA 11.8%| PubMedQA 2.9%, MedQA 26.6%, MCQA 14.7%|
| **autonomy_harm**         | Subdued across datasets                | Largest single change (+0.05 to +0.09) |
| **privacy_harm**          | Stable                                 | No meaningful change                   |
| **olmo-32b**              | Think variant — parse failures, outliers, inflated std | Instruct variant — clean, calibrated |
| **Inter-juror disagreement** | Higher (up to std 11.1 for olmo outliers) | Lower and bounded (max 0.21)        |
| **Score distribution**    | More 0–0.1 mass (72% PubMedQA)         | Shifted right into 0.1–0.2 band (57% PubMedQA) |

The increase in harm scores is driven by two independent factors: (1) the v2 prompt/rubric is more sensitive to autonomy harm in particular, and (2) olmo-32b's scores are now contributing meaningfully to the median aggregation instead of being near-zero due to parse failures. Both are improvements in measurement quality, not indications of increased actual harm in the evaluated responses.
