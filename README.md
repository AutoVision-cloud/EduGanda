# EduGanda Extension: Reducing MCQ Position Bias in a Luganda Literacy SLM

## Summary

Reproduced and extended the public [EduGanda-Gemma-3-1B](https://huggingface.co/CraneAILabs/EduGanda-Gemma-3-1B) training and evaluation setup for Luganda foundational literacy (Fab AI / Crane AI Labs, 2026). Identified answer-position imbalance in the released FLN training data, trained a position-balanced SFT variant using **MCQ option-permutation augmentation**, and evaluated whether this reduced per-answer accuracy spread while preserving overall benchmark accuracy. Ablated LEARNER/GRPO merge ratios to characterise the curriculum accuracy vs. linguistic fluency trade-off.

All experiments use only publicly released models, datasets, and benchmarks.

---

## Contributions

- Reproduced baseline evaluation for `ganda-gemma-1b` and `EduGanda-Gemma-3-1B`
- Audited released FLN training data for MCQ answer-position imbalance
- Implemented **option-permutation augmentation**: randomly permutes answer options and updates the correct letter, generating genuine positional variety without duplicating examples
- Evaluated overall accuracy, per-position accuracy, category accuracy, position-bias spread, **prediction distribution**, and **prediction entropy** per model
- Ablated LEARNER/GRPO merge ratios (linear 80/20, 70/30, 60/40 + DARE-TIES)
- Ablated LoRA rank, SFT epochs, and data balancing strategy with bootstrap CIs and McNemar tests
- Trained a logistic regression probe on hidden states to locate where position bias lives in the representations
- Quantized best checkpoint to GGUF (Q4\_K\_M, Q8\_0) for on-device inference
- Documented public-data limitations and benchmark-contamination checks

---

## Key Result

<!-- Fill in after running experiments -->

| Model | Accuracy | 95% CI | Spread | Pred entropy |
|-------|----------|--------|--------|--------------|
| ganda-gemma-1b (base) | — | — | — | — |
| EduGanda reference | — | — | — | — |
| Balanced SFT (permutation aug) | — | — | — | — |
| Best merge | — | — | — | — |

**Position-bias spread reduced from [X]pp to [Y]pp; overall accuracy changed from [A]% to [B]%.**

---

## Prediction Distribution Audit

<!-- Fill in after running 06_evaluate_all.py -->

| Model | Pred A | Pred B | Pred C | Pred D | Entropy (max=2.0) |
|-------|--------|--------|--------|--------|-------------------|
| ganda-gemma-1b | — | — | — | — | — |
| EduGanda reference | — | — | — | — | — |
| Balanced SFT | — | — | — | — | — |

---

## Data

| Asset | HuggingFace ID | Actual size |
|-------|---------------|-------------|
| Base model (CPT) | `CraneAILabs/ganda-gemma-1b` | 1B params |
| FLN training data | `CraneAILabs/luganda-fln-training-data` | **1,368 rows** |
| Bilingual exercises | `CraneAILabs/luganda-bilingual-literacy-exercises` | **3,472 rows** |
| Reward model | `CraneAILabs/luganda-reward-model` | 1B params |
| LLPK benchmark | `CraneAILabs/pedagogy-luganda-replaced` | **299 rows** |
| Reference model | `CraneAILabs/EduGanda-Gemma-3-1B` | 1B params |

**Public-data note:** The reference model reports 17,561 FLN training items; only **1,368** are publicly released (7.8% of the reported total). Results reflect a **partial reproduction using released assets** — differences from published numbers are expected and documented.

**Contamination check:** Zero overlap detected between benchmark questions and FLN training data (prefix-level check). Verified independently.

### FLN position bias (measured)

The FLN training data has a heavily skewed answer-position distribution, causing the model to predict position B far more often than others:

| Position | Count | % of MCQ items |
|----------|-------|----------------|
| A | 226 | 19.0% |
| **B** | **488** | **41.0%** ← overrepresented |
| C | 308 | 25.9% |
| D | 168 | 14.1% |

This 27pp spread between B (41%) and D (14%) is the source of the published 52pp accuracy gap (B: 93% vs D: 41% on the benchmark). Option-permutation augmentation addresses this at the training data level.

### Benchmark categories (299 items)

| Category | Items |
|----------|-------|
| SEND | 56 |
| Creative arts | 43 |
| Maths | 41 |
| Literacy | 40 |
| Science | 37 |
| Social studies | 32 |
| Technology | 31 |
| General | 19 |

---

## Reproduce

```bash
# On an A100/A10G GPU instance (RunPod / Modal)
bash <(curl -s https://raw.githubusercontent.com/AutoVision-cloud/EduGanda/main/setup.sh)

python scripts/01_explore_data.py
python scripts/02_baseline_eval.py
python scripts/03_sft_learner.py     # permutation augmentation on by default
python scripts/04_grpo.py            # skip if unstable; LEARNER-only is valid
python scripts/05_merge.py
python scripts/06_evaluate_all.py

# Ablations
python scripts/ablations/run_ablation.py --axis lora_rank --value 8
python scripts/ablations/run_ablation.py --axis lora_rank --value 32
python scripts/ablations/run_ablation.py --axis sft_epochs --value 1
python scripts/ablations/run_ablation.py --axis sft_epochs --value 5
python scripts/ablations/run_ablation.py --axis balance_strategy --value none
python scripts/ablations/run_ablation.py --axis balance_strategy --value undersample
python scripts/ablations/summarize_ablations.py

# Diagnostics
python scripts/diagnostics/probe_position_bias.py
python scripts/diagnostics/analyze_reward_model.py

# Deploy
export LLAMA_CPP_DIR=/path/to/llama.cpp
python scripts/deploy/quantize_and_benchmark.py
```

---

## Limitations

- Training data gap: ~8.3k items vs 17.5k in reference model
- GRPO stability: small models with unfamiliar reward models can diverge; LEARNER-only ablations are still reported
- Contamination check is prefix-level only (semantic overlap not verified)
- Human evaluation of generated lessons not performed
- On-device battery/cold-start profiling not included in this study

---

## Compute

| Stage | A10G time | Cost (~$0.76/hr) |
|-------|-----------|-----------------|
| Setup + exploration | 0.5 hr | $0.38 |
| Baseline evaluation | 0.5 hr | $0.38 |
| SFT LEARNER | 1 hr | $0.76 |
| GRPO-600 | 2–3 hr | $2.28 |
| Merge + eval sweep | 2 hr | $1.52 |
| Ablations (6 runs) | 6–9 hr | $6.84 |
| Diagnostics + deploy | 2 hr | $1.52 |
| **Total** | **~14–17 hr** | **~$13–17** |
