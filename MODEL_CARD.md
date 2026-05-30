# Model Card: EduGanda Extension

## Model Description

Fine-tuned extension of CraneAILabs/ganda-gemma-1b for foundational literacy
education in Uganda. Built on top of the EduGanda-Gemma-3-1B pipeline with two
contributions: (1) position-balanced training data to fix known MCQ position bias,
(2) merge ratio ablation to characterise the LEARNER/GRPO accuracy–fluency trade-off.

- **Base model:** CraneAILabs/ganda-gemma-1b (Gemma-3-1B after Luganda CPT)
- **Language:** Luganda (primary), English (instructions only)
- **Domain:** Foundational literacy, early childhood pedagogy
- **Intended use:** On-device educational assistant for P1–P3 learners in Uganda

---

## Training Data

| Dataset | HuggingFace ID | Rows used |
|---------|---------------|-----------|
| FLN training data | CraneAILabs/luganda-fln-training-data | ~1.37k |
| Bilingual exercises | CraneAILabs/luganda-bilingual-literacy-exercises | ~6.94k |

**Note:** The reference model (EduGanda-Gemma-3-1B) reports 17,561 FLN items. The
publicly available data totals ~8.3k items. Results may differ from the reference;
all models are evaluated under the same setup for comparability.

**Data contamination check:** No overlap detected between benchmark questions
(pedagogy-luganda-replaced) and training data (prefix-level check).

---

## Evaluation Results

Evaluated on CraneAILabs/pedagogy-luganda-replaced (299 bilingual MCQ items).
Prompt: English instruction + Luganda question + Luganda answer options.
Scoring: logit-based (softmax over A/B/C/D token probabilities).

<!-- Fill in after running 06_evaluate_all.py and summarize_ablations.py -->

| Model | Accuracy | 95% CI | Position Bias Spread |
|-------|----------|--------|----------------------|
| ganda-gemma-1b (base) | — | — | — |
| LEARNER (SFT, balanced) | — | — | — |
| Best ablation | — | — | — |
| EduGanda reference | — | — | — |

### Ablation findings
<!-- Fill in after running summarize_ablations.py -->

### Position bias probe
<!-- Fill in after running probe_position_bias.py -->
Probe CV accuracy on hidden states (chance = 25%):
- Base model: —
- LEARNER: —
- Best merged model: —

---

## Quantized Versions

<!-- Fill in after running quantize_and_benchmark.py -->

| Format | Accuracy | Speed (tok/s) | Size |
|--------|----------|---------------|------|
| bfloat16 (full) | — | — | ~2GB |
| Q8_0 GGUF | — | — | ~1GB |
| Q4_K_M GGUF | — | — | ~500MB |

---

## Known Limitations

- Training data gap: ~8.3k items vs 17.5k in reference model
- GRPO stability: small models with unfamiliar reward models can diverge
- Evaluation prompt sensitivity: absolute numbers depend on prompt template
- Contamination check is prefix-level only

---

## Reproduce

```bash
pip install -r requirements.txt
huggingface-cli login

# Core pipeline
python scripts/01_explore_data.py
python scripts/02_baseline_eval.py
python scripts/03_sft_learner.py
python scripts/04_grpo.py          # skip if unstable
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
