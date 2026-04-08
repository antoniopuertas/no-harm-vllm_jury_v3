# SOTA Literature Review: No-Harm-VLLM

**Date:** 2026-04-01
**Scope:** State-of-the-art research relevant to the No-Harm-VLLM jury-based medical AI safety evaluation system.
**Coverage:** 2019–2026 | arXiv, ACL, NeurIPS, EMNLP, ICLR, Nature

---

## Overview

This report surveys the state of the art across five areas directly relevant to No-Harm-VLLM: a jury-based, 5-model LLM evaluation framework that scores AI-generated medical responses across 7 harm dimensions (informational, social, psychological, autonomy, economic, privacy, epistemic) via reliability-weighted median aggregation, targeting medical QA datasets (MedQA, MedMCQA, PubMedQA).

---

## Section 1 — LLM-as-Judge / Evaluation Frameworks

### 1.1 Foundational Multi-Judge Works

**"Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena"**
- Authors: Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, et al. (Berkeley / LMSYS)
- Year: 2023 | Venue: NeurIPS 2023 (Datasets & Benchmarks)
- arXiv: 2306.05685
- Relevance: The seminal paper establishing the LLM-as-judge paradigm. Demonstrates that strong LLMs (GPT-4) match human preferences at over 80% agreement and identifies key biases (position, verbosity, self-preference) that any multi-judge system must account for. No-Harm-VLLM's use of 5 independent jurors directly addresses the single-judge bias problem this paper describes.

---

**"Replacing Judges with Juries: Evaluating LLM Generations with a Panel of Diverse Models" (PoLL)**
- Authors: Pat Verga, Sebastian Hofstatter, Sophia Althammer, et al.
- Year: 2024 | Venue: ACL 2024
- arXiv: 2404.18796
- Relevance: **Closest methodological parallel to No-Harm-VLLM's jury design.** PoLL shows that a diverse ensemble of smaller models outperforms a single large judge (GPT-4) on six datasets while reducing costs by 7× and reducing intra-family bias. The use of "diverse models" as a panel to overcome single-model limitations closely mirrors No-Harm-VLLM's 5-juror architecture.

---

**"Prometheus: Inducing Fine-grained Evaluation Capability in Language Models"**
- Authors: Seungone Kim, Jamin Shin, Yejin Cho, et al.
- Year: 2024 | Venue: ICLR 2024
- arXiv: 2310.08491
- Relevance: Introduces an open-source evaluator LLM trained to score long-form text against user-supplied rubrics, achieving GPT-4-level performance when given reference materials. No-Harm-VLLM's harm dimension prompts function as precisely such rubrics; Prometheus's architecture validates the approach of giving structured evaluation criteria to LLM judges.

---

**"G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment"**
- Authors: Yang Liu, Dan Iter, Yichong Xu, et al.
- Year: 2023 | Venue: EMNLP 2023
- arXiv: 2303.16634
- Relevance: Establishes that chain-of-thought prompting combined with form-filling paradigms produces evaluators with substantially stronger human correlation than prior methods. The structured scoring approach — providing explicit criteria and having models reason through each dimension — directly informs how No-Harm-VLLM prompts its jurors.

---

**"ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate"**
- Authors: Chi-Min Chan, Weize Chen, Yusheng Su, et al.
- Year: 2023 | Venue: ICLR 2024
- arXiv: 2308.07201
- Relevance: Proposes multi-agent debate between evaluator LLMs as a refinement mechanism, showing that group deliberation produces higher-quality evaluation than individual scoring. While No-Harm-VLLM uses independent (non-deliberating) jurors, this work provides a useful comparison point and motivation for multi-model evaluation.

---

**"LLM-Eval: Unified Multi-Dimensional Automatic Evaluation for Open-Domain Conversations"**
- Authors: Yen-Ting Lin, Yun-Nung Chen
- Year: 2023 | Venue: NLP4ConvAI Workshop (ACL 2023)
- arXiv: 2305.13711
- Relevance: Introduces a single-prompt evaluation method measuring multiple quality dimensions simultaneously, showing that multi-dimensional evaluation schemas outperform single-criterion scoring. Directly validates No-Harm-VLLM's approach of having jurors score across 7 distinct harm axes.

---

### 1.2 Judge Reliability, Bias, and Inter-Rater Agreement

**"Judging the Judges: Evaluating Alignment and Vulnerabilities in LLMs-as-Judges"**
- Authors: Aman Singh Thakur, Kartik Choudhary, Venkat Srinik Ramayapally, et al.
- Year: 2025 | Venue: arXiv preprint
- arXiv: 2406.12624
- Relevance: Evaluates 13 judge models against 9 exam-taker models, finding only the largest models achieve reasonable human alignment and identifying leniency bias and prompt sensitivity as key vulnerabilities. This directly motivates No-Harm-VLLM's use of reliability-weighted aggregation: not all jurors are equally trustworthy.

---

**"Evaluative Fingerprints: Stable and Systematic Differences in LLM Evaluator Behavior"**
- Authors: Wajid Nasser
- Year: 2026 | Venue: arXiv preprint
- arXiv: 2601.05114
- Relevance: Discovers a "reliability paradox" — LLM judges are individually self-consistent but show near-zero inter-judge agreement (Krippendorff's α = 0.042), with each model having distinct "evaluative dispositions." This finding **strongly justifies using reliability-weighted median** rather than simple averaging, exactly as implemented in No-Harm-VLLM.

---

**"LLMs instead of Human Judges? A Large Scale Empirical Study across 20 NLP Evaluation Tasks" (JUDGE-BENCH)**
- Authors: Anna Bavaresco, Raffaella Bernardi, Leonardo Bertolazzi, et al.
- Year: 2025 | Venue: ACL 2025
- arXiv: 2406.18403
- Relevance: Evaluates 11 LLMs as human-judge replacements across 20 datasets, finding "considerable variability depending on task type" and recommending careful validation against human judgments. The finding that no single LLM reliably replaces human judgment across all tasks directly motivates using an ensemble of diverse jurors.

---

**"Black-box Uncertainty Quantification Method for LLM-as-a-Judge"**
- Authors: Nico Wagner, Michael Desmond, Rahul Nair, et al.
- Year: 2024 | Venue: arXiv preprint
- arXiv: 2410.11594
- Relevance: Proposes quantifying uncertainty in LLM evaluation outputs via token probability confusion matrices, finding strong correlation between uncertainty scores and evaluation accuracy. Provides a technical foundation for No-Harm-VLLM's reliability tracking — unreliable juror responses can be detected and down-weighted.

---

**"AutoBench: Automating LLM Evaluation through Reciprocal Peer Assessment"**
- Authors: Dario Loi, Elena Maria Muià, Federico Siciliano, et al.
- Year: 2025 | Venue: arXiv preprint
- arXiv: 2510.22593
- Relevance: Demonstrates an iterative weighting mechanism for identifying reliable judges within a multi-LLM ensemble, achieving 78% correlation with MMLU-Pro. The reliability-weighted aggregation concept maps closely to No-Harm-VLLM's ReliabilityTracker component.

---

**"Autorubric: A Unified Framework for Rubric-Based LLM Evaluation"**
- Authors: Delip Rao, Chris Callison-Burch
- Year: 2026 | Venue: arXiv preprint
- arXiv: 2603.00077
- Relevance: Consolidates rubric-based evaluation techniques including position-bias mitigation, verbosity-bias correction, few-shot calibration, and multi-judge ensembling with reliability metrics (Cohen's κ). This is essentially a blueprint for the technical choices made in No-Harm-VLLM's evaluation pipeline.

---

**"Majority Rules: LLM Ensemble is a Winning Approach for Content Categorization"**
- Authors: Ariel Kamen, Yakov Kamen
- Year: 2025 | Venue: arXiv preprint
- arXiv: 2511.15714
- Relevance: Shows that ensembling 10 LLMs via collective decision-making yields up to 65% F1 improvement over the strongest single model in taxonomy-based categorization, closely analogous to No-Harm-VLLM's 7-dimension harm classification task.

---

## Section 2 — Medical AI Safety Evaluation

### 2.1 Clinical Safety Benchmarks

**"First, do NOHARM: towards clinically safe large language models"**
- Authors: David Wu et al. (50+ co-authors)
- Year: 2025 | Venue: arXiv preprint
- arXiv: 2512.01241
- Relevance: **Closest semantic parallel to No-Harm-VLLM.** Introduces NOHARM, a 100-case benchmark with 12,747 expert annotations across 31 models and 10 specialties. Finds up to 22.2% severe harm potential, with 76.6% from errors of omission, and only moderate correlation (r = 0.61–0.64) between safety and standard benchmarks. Uses physician annotation rather than LLM juries — the scalability gap that No-Harm-VLLM fills.

---

**"Large language models provide unsafe answers to patient-posed medical questions"**
- Authors: Rachel L. Draelos, Samina Afreen, Barbara Blasko, et al.
- Year: 2025 | Venue: arXiv preprint
- arXiv: 2507.18905
- Relevance: Physician-led evaluation of Claude, Gemini, GPT-4o, and Llama3-70B on 222 patient medical questions finds unsafe responses in 5%–13% and problematic responses in 22%–43% depending on model. Demonstrates the real-world safety stakes that No-Harm-VLLM's harm scoring system is designed to quantify at scale.

---

**"Beyond Benchmarks: Dynamic, Automatic and Systematic Red-Teaming Agents for Trustworthy Medical Language Models"**
- Authors: Jiazhen Pan, Bailiang Jian, Paul Hager, et al.
- Year: 2025 | Venue: arXiv preprint
- arXiv: 2508.00923
- Relevance: Red-teams 15 medical LLMs across safety dimensions including robustness, privacy, bias, and hallucination. Finds 94% of previously correct answers fail dynamic robustness tests and 86% show privacy breaches, revealing that standard benchmark accuracy dramatically overstates safety. Directly motivates multi-dimensional harm evaluation beyond accuracy-only benchmarks.

---

**"CARES: Comprehensive Evaluation of Safety and Adversarial Robustness in Medical LLMs"**
- Authors: Sijia Chen, Xiaomin Li, Mengxue Zhang, et al.
- Year: 2025 | Venue: arXiv preprint
- arXiv: 2505.11413
- Relevance: Presents a benchmark with 18,000+ prompts spanning 8 medical safety principles, 4 harm levels, and 4 prompting styles. The 8-principle safety taxonomy and multi-level harm scoring approach are structurally analogous to No-Harm-VLLM's 7-dimension framework, offering a directly comparable safety categorization methodology.

---

**"Towards Safe AI Clinicians: A Comprehensive Study on Large Language Model Jailbreaking in Healthcare"**
- Authors: Hang Zhang, Qian Lou, Yanshan Wang
- Year: 2025 | Venue: arXiv preprint
- arXiv: 2501.18632
- Relevance: Systematically evaluates vulnerabilities of 7 LLMs to jailbreaking in medical contexts, proposing an automated domain-adapted evaluation pipeline. The healthcare-specific safety evaluation pipeline design is directly relevant to No-Harm-VLLM's automated scoring architecture.

---

**"Beyond Accuracy: Risk-Sensitive Evaluation of Hallucinated Medical Advice"**
- Authors: Savan Doshi
- Year: 2026 | Venue: arXiv preprint
- arXiv: 2602.07319
- Relevance: Proposes evaluating hallucinations not just for factual accuracy but for clinical risk through markers like treatment directives, contraindications, and dangerous medication references. Demonstrates that models appearing similar on accuracy metrics have substantially different risk profiles — a core motivation for No-Harm-VLLM's harm-dimension scoring.

---

**"DocLens: Multi-aspect Fine-grained Evaluation for Medical Text Generation"**
- Authors: Yiqing Xie, Sheng Zhang, Hao Cheng, et al.
- Year: 2023 | Venue: ACL 2024
- arXiv: 2311.09581
- Relevance: Introduces LLM-based multi-aspect evaluation of medical text (completeness, conciseness, attribution) that achieves substantially higher agreement with medical experts than conventional metrics. Validates the multi-dimensional, LLM-assisted evaluation approach for clinical content specifically.

---

**"A Multi-agent Large Language Model Framework to Automatically Assess Performance of a Clinical AI Triage Tool"**
- Authors: Adam E. Flanders, Yifan Peng, Luciano Prevedello, et al.
- Year: 2025 | Venue: arXiv preprint
- arXiv: 2510.26498
- Relevance: Evaluates ensemble vs. single LLM assessment of a clinical AI tool across ~30,000 CT scans. Finds that "an ensemble of medium to large sized open-source LLMs provides a more consistent and reliable method" than any single judge — the medical domain analogue of PoLL and a direct validation of the No-Harm-VLLM jury approach in clinical contexts.

---

### 2.2 Medical LLM Capabilities and Benchmark Context

**"Large Language Models Encode Clinical Knowledge" (Med-PaLM)**
- Authors: Karan Singhal, Shekoofeh Azizi, Tao Tu, et al. (Google)
- Year: 2022 | Venue: Nature (2023)
- arXiv: 2212.13138
- Relevance: Introduces MultiMedQA combining MedQA, MedMCQA, PubMedQA, and HealthSearchQA, establishing Med-PaLM as the first LLM to achieve passing scores on USMLE-style questions. Contextualizes the performance landscape on which No-Harm-VLLM evaluates harm.

---

**"Towards Expert-Level Medical Question Answering with Large Language Models" (Med-PaLM 2)**
- Authors: Karan Singhal, Tao Tu, Juraj Gottweis, et al. (Google)
- Year: 2023 | Venue: arXiv preprint (Nature, 2025)
- arXiv: 2305.09617
- Relevance: Med-PaLM 2 achieves 86.5% on MedQA and physician-preferred clinical answers on 8/9 evaluation axes, raising the question of whether high benchmark performance correlates with low harm potential — a gap that No-Harm-VLLM is designed to study.

---

**"Capabilities of GPT-4 on Medical Challenge Problems"**
- Authors: Harsha Nori, Nicholas King, Scott Mayer McKinney, et al. (Microsoft)
- Year: 2023 | Venue: arXiv preprint
- arXiv: 2303.13375
- Relevance: Shows GPT-4 exceeds USMLE passing score by 20+ points without specialized medical training, with superior calibration. Establishes GPT-4-class models as viable medical evaluators — directly applicable to using frontier LLMs as No-Harm-VLLM jurors.

---

## Section 3 — Harm Taxonomies for AI

**"DecodingTrust: A Comprehensive Assessment of Trustworthiness in GPT Models"**
- Authors: Boxin Wang, Weixin Chen, Hengzhi Pei, et al. (UIUC / UChicago)
- Year: 2023 | Venue: NeurIPS 2023 (Outstanding Paper)
- arXiv: 2306.11698
- Relevance: Evaluates GPT models across 8 trustworthiness dimensions: toxicity, stereotype bias, adversarial robustness, OOD robustness, privacy, ethics, and fairness. The multi-dimensional decomposition of trustworthiness into orthogonal harm axes is structurally identical to No-Harm-VLLM's 7-dimension approach, though with different dimension labels tailored to the medical domain.

---

**"DarkPatterns-LLM: A Multi-Layer Benchmark for Detecting Manipulative and Harmful AI Behavior"**
- Authors: Sadia Asif, Israel Antonio Rosales Laguan, Haris Khan, et al.
- Year: 2025 | Venue: arXiv preprint
- arXiv: 2512.22470
- Relevance: **Independently arrived at a near-identical 7-dimension harm taxonomy** — Legal/Power, Psychological, Emotional, Physical, Autonomy, Economic, and Societal — nearly identical in structure to No-Harm-VLLM's 7 dimensions. Tests GPT-4 and Claude on detecting autonomy-undermining patterns, finding 65.2%–89.7% detection rates.

---

**"LLM Harms: A Taxonomy and Discussion"**
- Authors: Kevin Chen, Saleh Afroogh, Abhejay Murali, et al.
- Year: 2025 | Venue: arXiv preprint
- arXiv: 2512.05929
- Relevance: Proposes a five-category harm taxonomy (pre-development, direct output, misuse and malicious application, downstream application) with accountability and transparency framing. Complements No-Harm-VLLM's response-level scoring by situating output harms within a broader development lifecycle taxonomy.

---

**"Introducing v0.5 of the AI Safety Benchmark from MLCommons"**
- Authors: Bertie Vidgen et al. (MLCommons AI Safety Working Group)
- Year: 2024 | Venue: arXiv preprint
- arXiv: 2404.12241
- Relevance: Establishes a 13-category hazard taxonomy with 43,090 test items covering 3 user personas (typical, malicious, vulnerable). The vulnerable user persona category overlaps significantly with No-Harm-VLLM's concern for patient safety in medical QA contexts.

---

**"WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs"**
- Authors: Seungju Han, Kavel Rao, Allyson Ettinger, et al.
- Year: 2024 | Venue: NeurIPS 2024
- arXiv: 2406.18495
- Relevance: Produces WildGuard, a lightweight moderation system covering 13 risk categories trained on 92,000 labeled examples. Demonstrates that a specialized, lighter model can perform comparably to GPT-4 for safety classification — relevant to whether a fine-tuned juror model could replace general LLMs in No-Harm-VLLM.

---

**"Constitutional AI: Harmlessness from AI Feedback"**
- Authors: Yuntao Bai et al. (Anthropic)
- Year: 2022 | Venue: arXiv preprint
- arXiv: 2212.08073
- Relevance: Establishes the paradigm of using AI itself to identify and label harmful outputs according to a set of principles (a "constitution"), then training models to avoid those harms. No-Harm-VLLM can be seen as an operational instantiation of this concept: a jury of LLMs applying harm principles as evaluation rubrics.

---

**"Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned"**
- Authors: Deep Ganguli, Liane Lovitt, Jackson Kernion, et al. (Anthropic)
- Year: 2022 | Venue: NeurIPS 2022 Workshop
- arXiv: 2209.07858
- Relevance: Documents systematic red-teaming across 2.7B–52B parameter models, releasing ~39,000 red-team attacks and categorizing harmful outputs. The taxonomy of harm types and scaling behaviors of safety inform both the juror model selection and expected harm rate baselines in No-Harm-VLLM.

---

**"XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models"**
- Authors: Paul Röttger, Hannah Rose Kirk, Bertie Vidgen, et al.
- Year: 2024 | Venue: NAACL 2024
- arXiv: 2308.01263
- Relevance: Highlights the over-refusal problem: safety-optimized models refuse benign medical queries that superficially resemble unsafe ones. Directly relevant to calibrating No-Harm-VLLM's jurors to avoid false-positive harm scoring on legitimate medical information requests.

---

## Section 4 — Jury / Ensemble Aggregation Methods

**"A Judge-Aware Ranking Framework for Evaluating Large Language Models without Ground Truth"**
- Authors: Mingyuan Xu, Xinzi Tan, Jiawei Wu, Doudou Zhou
- Year: 2026 | Venue: arXiv preprint
- arXiv: 2601.21817
- Relevance: Extends Bradley-Terry-Luce modeling to account for varying judge reliability, providing identifiability and consistency proofs for multi-judge LLM evaluation. Directly formalizes the statistical foundations underlying No-Harm-VLLM's reliability-weighted median aggregation strategy.

---

**"Scoring, Reasoning, and Selecting the Best! Ensembling Large Language Models via a Peer-Review Process"**
- Authors: Zhijun Chen, Zeyu Ji, Qianren Mao, et al.
- Year: 2025 | Venue: arXiv preprint
- arXiv: 2512.23213
- Relevance: Proposes LLM-PeerReview, which uses multiple LLMs to evaluate each other's responses and aggregates scores through both averaging and graphical-model inference. Outperforms baselines by 6.9–7.3%, demonstrating the practical value of multi-model ensemble scoring over single-judge approaches.

---

**"Calibrating Large Language Models with Sample Consistency"**
- Authors: Qing Lyu, Kumar Shridhar, Chaitanya Malaviya, et al.
- Year: 2024 | Venue: arXiv preprint
- arXiv: 2402.13904
- Relevance: Shows that sampling multiple outputs and measuring consistency across samples improves LLM calibration substantially. In the context of No-Harm-VLLM, consistency across 5 independent jurors serves an analogous calibration function — highly consistent scores across jurors indicate reliable harm assessments.

---

**"Am I More Pointwise or Pairwise? Revealing Position Bias in Rubric-Based LLM-as-a-Judge"**
- Authors: Yuzheng Xu, Tosho Hirasawa, Tadashi Kozuno, Yoshitaka Ushiku
- Year: 2026 | Venue: arXiv preprint
- arXiv: 2602.02219
- Relevance: Demonstrates systematic position bias even in rubric-based evaluation and proposes balanced permutation strategies. Directly applicable to prompt design for No-Harm-VLLM's harm-scoring prompts to ensure each dimension is evaluated without positional artifacts.

---

**"Evaluating Large Language Models Against Human Annotators in Latent Content Analysis"**
- Authors: Bojic L., Zagovora O., Zelenkauskaite A., et al.
- Year: 2025 | Venue: arXiv preprint
- arXiv: 2501.02532
- Relevance: Compares 7 LLM variants against 33 human annotators using Krippendorff's alpha for inter-rater reliability, finding LLMs show higher consistency than humans in structured analytical tasks. Provides methodological grounding for measuring inter-juror agreement in No-Harm-VLLM.

---

## Section 5 — Medical QA Benchmarks

### 5.1 Foundational Datasets

**"What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams" (MedQA)**
- Authors: Di Jin, Eileen Pan, Nassim Oufattole, et al. (MIT)
- Year: 2020 | Venue: Applied Sciences 2021
- arXiv: 2009.13081
- Relevance: Introduces MedQA — 12,723 English USMLE-format questions. The primary benchmark used in No-Harm-VLLM's evaluations.

---

**"MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering"**
- Authors: Ankit Pal, Logesh Kumar Umapathi, Malaikannan Sankarasubbu
- Year: 2022 | Venue: CHIL 2022
- arXiv: 2203.14371
- Relevance: Introduces MedMCQA — 194,000+ questions from Indian postgraduate medical entrance exams (AIIMS, NEET PG), spanning 2,400 medical topics. The second primary benchmark in No-Harm-VLLM's dataset suite.

---

**"PubMedQA: A Dataset for Biomedical Research Question Answering"**
- Authors: Qiao Jin, Bhuwan Dhingra, Zhengping Liu, et al.
- Year: 2019 | Venue: EMNLP 2019
- arXiv: 1909.06146
- Relevance: Introduces PubMedQA — 1,000 expert-annotated yes/no/maybe research questions from PubMed abstracts. The third dataset in No-Harm-VLLM's test suite, representing biomedical reasoning from scientific literature.

---

### 5.2 State-of-the-Art Performance

**"CURE: Confidence-driven Unified Reasoning Ensemble Framework"**
- Authors: Ziad Elshaer, Essam A. Rashed
- Year: 2025 | Venue: arXiv preprint
- arXiv: 2510.14353
- Relevance: Achieves 95.0% on PubMedQA and 78.0% on MedMCQA using a multi-model confidence-driven ensemble without fine-tuning. Evaluates directly on two of No-Harm-VLLM's three benchmark datasets and uses an ensemble methodology — providing both SOTA performance baselines and a methodological parallel.

---

**"Evaluating LLMs in Medicine: A Call for Rigor, Transparency"**
- Authors: Mahmoud Alwakeel, Aditya Nagori, et al.
- Year: 2025 | Venue: arXiv preprint
- arXiv: 2507.08916
- Relevance: Reviews MedQA, MedMCQA, and PubMedQA evaluations, finding most existing datasets "lack clinical realism, transparency, and robust validation processes." This critique directly motivates No-Harm-VLLM: high benchmark accuracy does not guarantee safe real-world responses, hence the need for harm-dimension scoring.

---

**"The Validity Gap in Health AI Evaluation"**
- Authors: Alvin Rajkomar, Pavan Sudarshan, Angela Lai, Lily Peng (Google)
- Year: 2026 | Venue: arXiv preprint
- arXiv: 2603.18294
- Relevance: Analyzes 18,707 consumer health queries, finding that suicide/self-harm queries comprise less than 0.7% of benchmark corpora and chronic disease management only 5.5% — revealing a critical mismatch between what benchmarks test and what patients actually ask. No-Harm-VLLM's focus on MedQA/MedMCQA/PubMedQA should be interpreted against this validity gap.

---

**"Selective Chain-of-Thought in Medical Question Answering"**
- Authors: Zaifu Zhan, Min Zeng, Shuang Zhou, et al.
- Year: 2026 | Venue: arXiv preprint
- arXiv: 2602.20130
- Relevance: Tests on MedQA-USMLE, MedMCQA, and PubMedQA with selective CoT, achieving 13–45% inference time reduction while maintaining accuracy. Relevant to No-Harm-VLLM's concerns about juror inference efficiency — selective reasoning strategies could reduce the cost of running 5 jurors per response.

---

## Summary Table

| Paper | Year | arXiv | Primary Relevance |
|-------|------|-------|-------------------|
| PoLL — Replacing Judges with Juries | 2024 | 2404.18796 | **Jury-of-LLMs evaluation paradigm** |
| NOHARM — Clinically Safe LLMs | 2025 | 2512.01241 | **Medical harm evaluation benchmark** |
| DarkPatterns-LLM | 2025 | 2512.22470 | **Near-identical 7-dimension harm taxonomy** |
| Evaluative Fingerprints | 2026 | 2601.05114 | **Justifies reliability-weighted median** |
| DecodingTrust | 2023 | 2306.11698 | Multi-dimensional harm axes (NeurIPS Outstanding) |
| MT-Bench / LLM-as-Judge | 2023 | 2306.05685 | LLM evaluation foundations |
| Prometheus | 2024 | 2310.08491 | Rubric-based LLM judge |
| G-Eval | 2023 | 2303.16634 | Multi-dimensional LLM scoring |
| Multi-agent clinical eval | 2025 | 2510.26498 | Ensemble LLMs for clinical AI |
| CARES | 2025 | 2505.11413 | Medical safety multi-dimension benchmark |
| DAS Red-Teaming | 2025 | 2508.00923 | Medical LLM safety gaps |
| AutoBench | 2025 | 2510.22593 | Reliability weighting for judge ensembles |
| Constitutional AI | 2022 | 2212.08073 | AI-evaluated harm principles |
| Judge-Aware Ranking | 2026 | 2601.21817 | Statistical foundations for reliability weighting |
| Med-PaLM 2 | 2023 | 2305.09617 | SOTA on MedQA/MedMCQA/PubMedQA |
| MedQA | 2020 | 2009.13081 | Primary benchmark dataset |
| MedMCQA | 2022 | 2203.14371 | Primary benchmark dataset |
| PubMedQA | 2019 | 1909.06146 | Primary benchmark dataset |
| CURE ensemble | 2025 | 2510.14353 | Current SOTA on PubMedQA, MedMCQA |
| The Validity Gap | 2026 | 2603.18294 | Benchmark limitations for patient safety |

---

## Positioning: Where No-Harm-VLLM Sits

### What No-Harm-VLLM uniquely combines

No existing system combines all five of the following:

1. **Automated, scalable evaluation** — vs. NOHARM's expensive physician annotations
2. **5 diverse LLM jurors** — vs. CARES/JUDGE-BENCH single-scorer approaches
3. **Reliability-weighted median aggregation** — addressing the "evaluative fingerprints" problem (α = 0.042)
4. **7 orthogonal harm dimensions** — broader than most existing benchmarks
5. **Applied specifically to medical QA benchmark responses** — not general-domain safety

### Four research threads it bridges

| Thread | Key prior work | No-Harm-VLLM contribution |
|--------|---------------|--------------------------|
| Single-judge → jury evaluation | PoLL (2404.18796), AutoBench (2510.22593) | Applies jury design to harm scoring with reliability weighting |
| Accuracy → harm-aware evaluation | NOHARM (2512.01241), The Validity Gap (2603.18294) | Automated multi-dimensional harm measurement layer |
| Multi-dimensional harm taxonomy | DarkPatterns-LLM (2512.22470), DecodingTrust (2306.11698) | Medical domain-specific 7-dimension taxonomy |
| Domain-specific safety evaluation | CARES (2505.11413), DocLens (2311.09581) | LLM-jury scalability on medical QA responses |

### The gap No-Harm-VLLM fills

High benchmark accuracy on MedQA/MedMCQA/PubMedQA is a poor proxy for clinical safety (validated by NOHARM, "The Validity Gap", "Beyond Accuracy"). No-Harm-VLLM provides the missing **automated, multi-dimensional, jury-based harm measurement layer** — scalable to thousands of samples without physician annotation cost.
