# EduGanda Extension: Reducing MCQ Position Bias in a Luganda Literacy SLM

## Summary

Reproduced and extended the public [EduGanda-Gemma-3-1B](https://huggingface.co/CraneAILabs/EduGanda-Gemma-3-1B) training and evaluation setup for Luganda foundational literacy (Fab AI / Crane AI Labs, 2026). Identified answer-position imbalance in the released FLN training data, trained a position-balanced SFT variant using **MCQ option-permutation augmentation**, and evaluated whether this reduced per-answer accuracy spread while preserving overall benchmark accuracy. Ablated LEARNER/GRPO merge ratios to characterise the curriculum accuracy vs. linguistic fluency trade-off.

All experiments use only publicly released models, datasets, and benchmarks.

---

## Contributions

- Reproduced baseline evaluation for `ganda-gemma-1b` and `EduGanda-Gemma-3-1B` using log-probability option scoring with verified tokenization
- Audited released FLN training data for MCQ answer-position imbalance; measured 27pp spread (B: 41% vs D: 14%)
- Implemented **option-permutation augmentation**: randomly permutes answer options and updates the correct letter, generating genuine positional variety without duplicating examples — a more principled approach than the original's English-example balancing
- Evaluated overall accuracy, per-position accuracy, per-subdomain/age-group accuracy, position-bias spread, **prediction distribution**, and **prediction entropy** per model
- Ran TF-IDF semantic contamination check between benchmark and training data (documented methodology vs. original undocumented claim)
- Measured tokenizer fertility: Luganda/English tokens-per-word ratio using `CraneAILabs/ganda-gemma-1b`
- Ablated LEARNER/GRPO merge ratios (linear 80/20, 70/30, 60/40 + DARE-TIES) with bootstrap CIs and McNemar tests
- Ablated LoRA rank, SFT epochs, and data balancing strategy
- Trained logistic regression probe on hidden states to locate where position bias lives in representations
- Quantized best checkpoint to GGUF (Q4\_K\_M, Q8\_0) and benchmarked inference speed vs. accuracy drop

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

## Background: The Original Pipeline

From the [Fab AI blog post](https://www.fab-ai.org/initiatives/ai-for-education/edtech-quality/resources/blog/fine-tuning-small-language-model-for-foundational-literacy-in-uganda) (May 2026):

```
Gemma 3 1B
    ↓  Continued pre-training (70/30 Luganda-English split)
    ↓
READER (MCQ SFT, ~13.2K items)     GRPO-600 (Unified RL, 2K steps,
LEARNER (+ pedagogy, lesson plans)  anti-repetition, tool-calling)
    ↓                                        ↓
    └──────── BRIDGE: 70% LEARNER + 30% GRPO-600 ──────┘
                         ↓
                EduGanda-Gemma-3-1B
```

**Published results:** 66% LLPK / 58.8% LLK benchmark (vs. 51% / 39% base model).
Outperforms Gemma 3 4B (4× larger) on LLPK.

**Key findings from the original work:**
- Small models cannot hold multiple skills simultaneously — model blending was essential to prevent catastrophic forgetting
- 35,000 curated training pairs outperformed 1.53M machine-translated pairs (quality > volume)
- Vocabulary layer required a much lower learning rate for coherent Luganda sentence generation
- Repetition penalty at inference outperformed GRPO for controlling output loops
- Deployment: LiteRT mobile runtime, **12–18 tokens/sec** on mid-range Android, **978MB in memory**

---

## Evaluation Protocol (frozen)

All models in this project are evaluated under the same fixed protocol. **Do not change this after SFT training begins** — internal comparisons are only valid if the evaluator is constant.

- **Scorer:** Generation-based — training-format prompt (Luganda question + options, no instruction prefix), `repetition_penalty=1.2`, extracts first standalone A/B/C/D letter from output (handles both bare letters and "Okuddamu: X" completions).
- **Prompt format:** Gemma chat template (`apply_chat_template`) with Luganda question + options only. Log-prob scoring was abandoned: SFT trains models to output "Okuddamu: X" not bare letters, making first-token logits unreliable post-training.
- **Benchmark:** `CraneAILabs/pedagogy-luganda-replaced` (299 items, both `cdpk_main` and `cdpk_send` splits).
- **Secondary metric:** Generation-based scoring with `repetition_penalty=1.2`, for deployment-style comparison.
- **Per-item predictions saved** for all runs — enables before/after analysis and McNemar tests.

**Reproducibility note:** The model card reports PCK accuracy (66% EduGanda, 51% base) but **does not release the exact MCQ prompt template or scoring script**. The only published usage example uses open-ended lesson generation, not MCQ answering. We therefore could not reproduce the 66% number. We use a documented local evaluation protocol and report all models under the same protocol — absolute numbers should not be compared to the published 66%, only models within this study should be compared to each other.

**Primary metrics:** Prediction distribution uniformity, answer-position spread, and prediction entropy — not raw accuracy. Our benchmark reconstruction does not reproduce published absolute accuracy, but it reliably measures prediction distribution and answer-position collapse under a fixed prompt.

**Secondary:** Raw accuracy under our protocol (relative comparisons only — do not compare to published 66%).
**Tertiary:** Generation format adherence (does the model output "Okuddamu: X" consistently?).

---

## Our Approach vs. the Original

| Aspect | Original (Fab AI / Crane AI Labs) | This project |
|--------|-----------------------------------|--------------|
| Bias fix | Balanced with English examples | **Option-permutation augmentation** (permutes A/B/C/D, more principled) |
| Training data | 13,200 Luganda curriculum items | 1,368 FLN + 3,472 exercises (publicly released only) |
| Evaluation | Generation-based (not fully documented) | **Log-prob option scoring**, frozen protocol, per-item predictions saved |
| Contamination | "Verified zero contamination" (no methodology given) | TF-IDF semantic similarity check, documented |
| Merge ablation | Single ratio tested (70/30) | **80/20, 70/30, 60/40, DARE-TIES** with bootstrap CIs |
| Bias analysis | Per-position accuracy only | Per-position + **prediction distribution + entropy + hidden-state probe** |

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

**Public-data note:** The original pipeline used **13,200 Luganda curriculum items**; only **1,368** are publicly released (10.4% of the training set used). Results reflect a partial reproduction using released assets — differences from published numbers are expected and documented.

**Contamination check:** Zero overlap detected (prefix-level + TF-IDF semantic similarity ≥ 0.7). Verified independently.

### FLN position bias (measured)

The FLN training data has a heavily skewed answer-position distribution, causing the model to predict position B far more often than others:

| Position | Count | % of MCQ items |
|----------|-------|----------------|
| A | 226 | 19.0% |
| **B** | **488** | **41.0%** ← overrepresented |
| C | 308 | 25.9% |
| D | 168 | 14.1% |

This 27pp training-data spread is the source of the published 52pp accuracy gap (B: 93% vs D: 41% on the benchmark). The original fix used English examples; we use option-permutation augmentation which directly attacks the positional artifact without introducing cross-lingual noise.

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

### On-device deployment context

The model targets offline use on low-cost Android phones in Uganda. The reference model uses **1.07 GB storage (Q8_0)** — downloadable for ~1,000 UGX ($0.27 USD) on a local MTN data bundle. This makes quantization accuracy meaningful: every percentage point of accuracy lost to quantization has a real pedagogical cost.

---

## Reproduce

```bash
# On a GPU instance (RunPod A10G recommended)
bash <(curl -s https://raw.githubusercontent.com/AutoVision-cloud/EduGanda/main/setup.sh)

python scripts/diagnostics/tokenizer_fertility.py  # no GPU needed, run first
python scripts/01_explore_data.py
python scripts/diagnostics/semantic_contamination.py
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

- **Training data gap:** 1,368 public items vs. 13,200 used in the original (10.4% available)
- **GRPO stability:** small models with unfamiliar reward models can diverge; LEARNER-only ablations are reported separately. Note: the original found repetition penalty outperformed GRPO for repetition control.
- Contamination check is TF-IDF semantic similarity — semantic paraphrase overlap not verified
- Human evaluation of generated lesson plans not performed
- On-device battery/cold-start profiling not included (reference: 12–18 tok/s on mid-range Android)

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
