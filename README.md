# EduGanda Extension: Position Bias Fix + Merge Ratio Ablation

## TL;DR

Reproduced the EduGanda-Gemma-3-1B pipeline using open-source data.
Fixed the known MCQ position bias (52pp spread → Xpp) through balanced dataset curation.
Ablated LEARNER/GRPO merge ratios to characterise the accuracy/fluency trade-off.

## Background

[Fab AI blog post — link here]. EduGanda-Gemma-3-1B (Crane AI Labs, 2026) is a
1B parameter model for foundational literacy education in Uganda, trained in Luganda
via CPT → SFT → GRPO → model merge. The blog acknowledges a position bias in the
benchmark results (B: 93%, D: 41%) caused by an imbalanced training dataset.

## What I changed

1. **Position-balanced training data:** The original training set over-represents
   answer position B. This project oversamples underrepresented positions (C, D)
   to match the most frequent position, without discarding any data.

2. **Merge ratio ablation:** Tested linear 80/20, 70/30, 60/40, and DARE-TIES 70/30
   (LEARNER/GRPO) to characterise the trade-off between pedagogy accuracy and
   linguistic fluency.

## Results

<!-- Paste your summary table from 06_evaluate_all.py here -->

| Model | PCK Acc | Spread | Notes |
|-------|---------|--------|-------|
| ganda-gemma-1b (base) | | | starting point |
| LEARNER (SFT, balanced) | | | Contribution #1 |
| Merged 80/20 | | | |
| Merged 70/30 | | | paper ratio |
| Merged 60/40 | | | |
| DARE-TIES 70/30 | | | |
| EduGanda reference | | | target |

## Key findings

<!-- Fill in after running experiments -->

- Did position balancing reduce spread without hurting overall accuracy?
- Did more GRPO weight help or hurt pedagogy accuracy?
- Which merge method (linear vs DARE-TIES) performed better?

## Reproduce

```bash
# On an A100 GPU instance (RunPod / Modal)
pip install -r requirements.txt
huggingface-cli login

python scripts/01_explore_data.py   # data exploration + contamination check
python scripts/02_baseline_eval.py  # baseline numbers before any training
python scripts/03_sft_learner.py    # SFT with position-balanced data
python scripts/04_grpo.py           # GRPO (skip if unstable — see notes below)
python scripts/05_merge.py          # 4 merge configurations
python scripts/06_evaluate_all.py   # full evaluation sweep + summary table
```

**If GRPO fails or diverges:** A LEARNER-only model with fixed position bias is
already a valid contribution. Skip `04_grpo.py` and `05_merge.py` and report
LEARNER vs reference directly.

## Data

| Asset | HuggingFace ID |
|-------|----------------|
| Base model (CPT) | `CraneAILabs/ganda-gemma-1b` |
| FLN training data | `CraneAILabs/luganda-fln-training-data` |
| Bilingual exercises | `CraneAILabs/luganda-bilingual-literacy-exercises` |
| Reward model | `CraneAILabs/luganda-reward-model` |
| LLPK benchmark | `CraneAILabs/pedagogy-luganda-replaced` |
| Reference model | `CraneAILabs/EduGanda-Gemma-3-1B` |

Training data available: ~8.3k items (1.37k FLN + 6.94k exercises).
The reference model reports 17,561 FLN items — some may not have been released.
Results may differ from published numbers; all models are evaluated under the same
setup for a fair comparison.

## Limitations

<!-- Fill in honestly after running — this is more credible than only reporting wins -->

- Training data gap vs reference (8.3k vs 17.5k items)
- GRPO stability on 1B models with unfamiliar reward models
- Evaluation uses prefix-level contamination check only

## Compute

| Stage | A100 time | Cost (~$1.99/hr) |
|-------|-----------|------------------|
| Setup + exploration | 0.5 hr | $1.00 |
| Baseline evaluation | 0.5 hr | $1.00 |
| SFT LEARNER | 1 hr | $1.99 |
| GRPO-600 | 2–3 hr | $5.97 |
| Merge (4 variants) | 0.25 hr | $0.50 |
| Full eval sweep | 1.5 hr | $2.99 |
| **Total** | **~6–7 hr** | **~$13** |
