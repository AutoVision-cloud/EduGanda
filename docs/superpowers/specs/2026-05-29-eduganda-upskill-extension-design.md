# EduGanda Technical Upskill Extension — Design Spec

**Date:** 2026-05-29
**Status:** Approved
**Goal:** Extend the existing EduGanda replication pipeline with three technical tracks to maximize upskilling across research, engineering, and data-centric ML — all on a single GPU.

---

## Context

The existing pipeline (scripts 01–06) covers: data exploration, baseline evaluation, SFT with position-balanced data, GRPO, model merging (linear + DARE-TIES), and full evaluation sweep. This extension builds on top of that without replacing it.

The control experiment is a complete run of scripts 01–06 with the default hyperparameters. All ablations and diagnostics are measured relative to this control.

---

## Overall Structure

```
EduGanda/
├── scripts/
│   ├── 01_explore_data.py          ← existing
│   ├── 02_baseline_eval.py         ← existing
│   ├── 03_sft_learner.py           ← existing (+ W&B + diagnostic callback)
│   ├── 04_grpo.py                  ← existing (+ W&B + diagnostic callback)
│   ├── 05_merge.py                 ← existing
│   ├── 06_evaluate_all.py          ← existing (+ bootstrap CI + McNemar)
│   ├── ablations/
│   │   ├── run_ablation.py         ← parameterised single ablation run
│   │   └── summarize_ablations.py  ← aggregates all results into table + plots
│   ├── diagnostics/
│   │   ├── probe_position_bias.py  ← linear probe on hidden states
│   │   └── analyze_reward_model.py ← reward score distribution before/after GRPO
│   └── deploy/
│       └── quantize_and_benchmark.py ← GGUF quantization + inference benchmark
├── configs/                        ← existing (mergekit YAMLs)
├── results/
│   ├── baseline_results.json       ← existing
│   ├── full_results.json           ← existing
│   ├── ablations/
│   │   └── <axis>/<value>/results.json
│   └── diagnostics/
│       ├── training_curves.jsonl
│       ├── position_bias_probe.json
│       └── reward_analysis.json
├── docs/
│   └── superpowers/specs/          ← this file
├── MODEL_CARD.md
├── README.md                       ← existing (fill results table after experiments)
└── requirements.txt                ← existing (+ wandb, llama-cpp-python)
```

Execution order:
1. Scripts 01–06 (control run)
2. `scripts/ablations/run_ablation.py` (repeated per axis/value)
3. `scripts/diagnostics/probe_position_bias.py`
4. `scripts/diagnostics/analyze_reward_model.py`
5. `scripts/ablations/summarize_ablations.py`
6. `scripts/deploy/quantize_and_benchmark.py`

---

## Track B — Rigorous Ablation Study

### Ablation Axes

| Axis | Values | Control |
|------|--------|---------|
| Data balancing strategy | none, oversample, undersample-to-min | oversample |
| LoRA rank | 8, 16, 32 | 16 |
| SFT training duration | 1 epoch, 3 epochs, 5 epochs | 3 epochs |
| Merge ratio | 80/20, 70/30, 60/40, DARE-TIES | 70/30 (already in plan) |
| GRPO steps | 200, 600, 1000 | 600 |

Each run is a single axis variation against the control. All other hyperparameters are held fixed.

### Implementation: `run_ablation.py`

A parameterised script that accepts CLI arguments (`--axis`, `--value`) and runs the relevant training stage(s), saving results to `results/ablations/<axis>/<value>/results.json`. Reuses training and evaluation functions from the existing scripts — this requires extracting the core logic from `03_sft_learner.py`, `04_grpo.py`, and `06_evaluate_all.py` into importable functions (e.g. `train_sft(config)`, `train_grpo(config)`, `evaluate(model, tokenizer, benchmark)`). The existing scripts become thin CLI wrappers around these functions.

### Evaluation Rigor

Every model comparison in `06_evaluate_all.py` and `summarize_ablations.py` includes:

- **Bootstrap confidence intervals** (1000 resamples) on overall accuracy and per-position accuracy
- **McNemar's test** when comparing two models head-to-head on the same benchmark items (tests whether disagreements are statistically significant)
- **Calibration curve** — binned confidence vs accuracy plot (requires extracting token logits for A/B/C/D instead of greedy decoding)

### Output

`summarize_ablations.py` produces:
- A markdown table with accuracy ± CI for each ablation value
- Per-axis plots (accuracy vs value, spread vs value)
- A `results/ablations/summary.json` for the model card and README

---

## Track A — Diagnostics & Interpretability

### 1. Training Callbacks

A custom `DiagnosticCallback` (subclass of `transformers.TrainerCallback`) added to both `SFTTrainer` and `GRPOTrainer`. Logs per step to `results/diagnostics/training_curves.jsonl`:

- Gradient norm (global)
- Learning rate
- Loss
- (GRPO only) Reward mean, std, min, max
- (GRPO only) KL divergence from reference policy

Warning thresholds: if KL > 0.5 or reward std < 0.01 for 50 consecutive steps, print a warning (reward hacking or reward collapse signal).

### 2. Position Bias Probing (`probe_position_bias.py`)

For each of: base model, LEARNER, best merged model:

1. Extract last-layer hidden states for all 299 benchmark items (at the final token of the user prompt)
2. Train a logistic regression classifier (scikit-learn, 5-fold CV) to predict `correct_answer` (A/B/C/D) from the hidden state
3. Report CV accuracy — chance is 25%

If accuracy > 35% before SFT and drops toward chance after, position bias was encoded in representations and training removed it. If it stays high, the model learned position cues that survived training. This is a publishable finding either way.

### 3. Reward Model Analysis (`analyze_reward_model.py`)

Two runs:
- Score the full FLN training set with the reward model (before GRPO)
- Score 300 model outputs from the LEARNER model (before GRPO) and the GRPO model (after)

Produces:
- Score distribution histogram (pre vs post GRPO)
- Breakdown by format (MCQ vs non-MCQ) and position (A/B/C/D)
- Surface pattern check: does GRPO output have suspiciously uniform answer letters?

---

## Track C — Production ML

### 1. W&B Experiment Tracking

Add `wandb.init()` and `report_to="wandb"` to `SFTConfig` / `GRPOConfig` in the existing scripts. Each ablation run logs as a separate W&B run under the same project (`eduganda-extension`). Hyperparameters auto-logged. Falls back gracefully with `WANDB_MODE=disabled` if not configured.

Addition to `requirements.txt`: `wandb>=0.17.0`

### 2. Quantization + Inference Benchmark (`quantize_and_benchmark.py`)

Steps:
1. Identify best checkpoint from `results/ablations/summary.json` (highest accuracy)
2. Export to GGUF via `llama.cpp` convert script: Q4_K_M and Q8_0 quantization levels
3. Run inference benchmark on the 299 benchmark items at each quantization level:
   - Tokens/sec (throughput)
   - Peak memory (MB)
   - Benchmark accuracy (does quantization hurt?)
4. Save to `results/deploy/quantization_benchmark.json`

This directly addresses the deployment context: the model is intended for on-device use in Uganda.

Addition to `requirements.txt`: `llama-cpp-python>=0.2.0`

### 3. Model Card (`MODEL_CARD.md`)

Structured sections:
- **Intended use:** foundational literacy education in Luganda, on-device inference
- **Training data:** sources, sizes, known gaps vs reference (8.3k vs 17.5k items)
- **Evaluation:** LLPK benchmark results table with bootstrap CIs; position bias spread before/after fix
- **Quantization:** accuracy and throughput at Q4_K_M and Q8_0
- **Known limitations:** data gap, GRPO stability, evaluation prompt sensitivity
- **Reproduce:** pip install + script execution order

---

## Key Constraints

- Single GPU (A100 on RunPod/Modal) — no distributed training
- All experiments share the same evaluation function and prompt template for comparability
- GRPO ablations only run if the control GRPO run succeeded; otherwise LEARNER-only ablations are reported with a note
- No new datasets introduced — all data sourced from the existing CraneAILabs HuggingFace repos

---

## Success Criteria

| Dimension | Signal |
|-----------|--------|
| Research | Position bias probe produces interpretable finding; ablation table shows clear trends with CIs |
| Engineering | W&B dashboard shows all runs; quantization benchmark completed; model card published |
| Portfolio | Public repo with README results table, model card, and plots; CV bullet updated |
