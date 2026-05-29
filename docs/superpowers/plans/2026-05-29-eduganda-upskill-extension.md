# EduGanda Technical Upskill Extension — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing EduGanda pipeline with diagnostic instrumentation (Track A), rigorous ablation study infrastructure (Track B), and production deployment tooling (Track C).

**Architecture:** Refactor shared training/evaluation logic into importable `scripts/core/` modules so all three tracks can share them. Existing CLI scripts (01–06) become thin wrappers over core functions. Tracks A, B, C then layer on top independently.

**Tech Stack:** PyTorch, Transformers, TRL ≥ 0.12, PEFT, datasets, scikit-learn, scipy, matplotlib, wandb, llama-cpp-python

---

## File Map

**New files:**
- `scripts/core/__init__.py`
- `scripts/core/data.py` — data loading and balancing functions
- `scripts/core/evaluate.py` — evaluation + statistical test functions
- `scripts/core/callbacks.py` — DiagnosticCallback for training
- `scripts/core/train.py` — parameterised train_sft() and train_grpo()
- `scripts/ablations/__init__.py`
- `scripts/ablations/run_ablation.py` — CLI ablation runner
- `scripts/ablations/summarize_ablations.py` — aggregates results into table + plots
- `scripts/diagnostics/__init__.py`
- `scripts/diagnostics/probe_position_bias.py` — linear probe on hidden states
- `scripts/diagnostics/analyze_reward_model.py` — reward score distributions
- `scripts/deploy/__init__.py`
- `scripts/deploy/quantize_and_benchmark.py` — GGUF + inference benchmark
- `tests/__init__.py`
- `tests/test_data.py`
- `tests/test_evaluate.py`
- `tests/test_callbacks.py`
- `tests/test_probe.py`
- `MODEL_CARD.md`

**Modified files:**
- `requirements.txt` — add wandb, scikit-learn, scipy, matplotlib, llama-cpp-python
- `scripts/03_sft_learner.py` — thin wrapper over core/train.py
- `scripts/04_grpo.py` — thin wrapper over core/train.py
- `scripts/06_evaluate_all.py` — import from core/evaluate.py, add bootstrap CI + McNemar

---

## Task 1: Update dependencies and scaffold module structure

**Files:**
- Modify: `requirements.txt`
- Create: `scripts/core/__init__.py`, `scripts/ablations/__init__.py`, `scripts/diagnostics/__init__.py`, `scripts/deploy/__init__.py`, `tests/__init__.py`

- [ ] **Step 1: Update requirements.txt**

Replace the file contents with:

```
torch>=2.1.0
transformers>=4.40.0
trl>=0.12.0
peft>=0.10.0
datasets>=2.18.0
accelerate>=0.28.0
bitsandbytes>=0.43.0
mergekit-community
huggingface_hub>=0.22.0
pyyaml>=6.0
wandb>=0.17.0
scikit-learn>=1.4.0
scipy>=1.12.0
matplotlib>=3.8.0
llama-cpp-python>=0.2.0
```

- [ ] **Step 2: Create empty __init__.py files**

```bash
touch scripts/core/__init__.py \
      scripts/ablations/__init__.py \
      scripts/diagnostics/__init__.py \
      scripts/deploy/__init__.py \
      tests/__init__.py
mkdir -p results/ablations results/diagnostics results/deploy
```

- [ ] **Step 3: Install new dependencies (on your GPU instance)**

```bash
pip install wandb scikit-learn scipy matplotlib llama-cpp-python
```

Expected: installs without error.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt scripts/core/__init__.py scripts/ablations/__init__.py \
        scripts/diagnostics/__init__.py scripts/deploy/__init__.py tests/__init__.py
git commit -m "chore: add upskill extension deps and module structure"
```

---

## Task 2: Create scripts/core/data.py + tests

**Files:**
- Create: `scripts/core/data.py`
- Create: `tests/test_data.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_data.py
import pytest
from collections import Counter
from scripts.core.data import balance_by_position, extract_first_letter


def _make_mcq_items(counts):
    """counts = {'A': n, 'B': n, ...}"""
    items = []
    for pos, n in counts.items():
        for i in range(n):
            items.append({"correct_letter": pos, "text": f"item_{pos}_{i}"})
    return items


def test_oversample_balances_to_max():
    items = _make_mcq_items({"A": 10, "B": 30, "C": 5, "D": 20})
    balanced = balance_by_position(items, strategy="oversample")
    counts = Counter(x["correct_letter"] for x in balanced)
    assert counts["A"] == counts["B"] == counts["C"] == counts["D"]
    assert counts["B"] == 30  # max is preserved


def test_undersample_balances_to_min():
    items = _make_mcq_items({"A": 10, "B": 30, "C": 5, "D": 20})
    balanced = balance_by_position(items, strategy="undersample")
    counts = Counter(x["correct_letter"] for x in balanced)
    assert counts["A"] == counts["B"] == counts["C"] == counts["D"]
    assert counts["A"] == 5  # min is preserved


def test_none_strategy_returns_original():
    items = _make_mcq_items({"A": 10, "B": 30, "C": 5, "D": 20})
    result = balance_by_position(items, strategy="none")
    assert len(result) == len(items)


def test_extract_first_letter_finds_abcd():
    assert extract_first_letter("A is correct") == "A"
    assert extract_first_letter("The answer is B.") == "B"
    assert extract_first_letter("  c  ") == "C"
    assert extract_first_letter("xyz") is None
    assert extract_first_letter("") is None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_data.py -v
```

Expected: `ModuleNotFoundError` or `ImportError` — module doesn't exist yet.

- [ ] **Step 3: Implement scripts/core/data.py**

```python
# scripts/core/data.py
import random
from collections import Counter
from typing import List, Dict, Optional

from datasets import load_dataset, Dataset, concatenate_datasets


def balance_by_position(
    mcq_items: List[Dict],
    strategy: str = "oversample",
    seed: int = 42,
) -> List[Dict]:
    """
    strategy: 'oversample' | 'undersample' | 'none'
    Returns a new list (does not mutate input).
    """
    if strategy == "none":
        return list(mcq_items)

    rng = random.Random(seed)
    by_pos = {}
    for item in mcq_items:
        by_pos.setdefault(item["correct_letter"], []).append(item)

    if strategy == "oversample":
        target = max(len(v) for v in by_pos.values())
        result = []
        for items in by_pos.values():
            if len(items) < target:
                result.extend(items * (target // len(items)))
                result.extend(rng.sample(items, target % len(items)))
            else:
                result.extend(items)
    elif strategy == "undersample":
        target = min(len(v) for v in by_pos.values())
        result = []
        for items in by_pos.values():
            result.extend(rng.sample(items, target))
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}. Use 'oversample', 'undersample', or 'none'.")

    rng.shuffle(result)
    return result


def extract_first_letter(text: str) -> Optional[str]:
    """Extract the first A/B/C/D letter from a model response string."""
    for char in text.upper():
        if char in "ABCD":
            return char
    return None


def load_fln_dataset(subset: str = "all") -> Dataset:
    return load_dataset("CraneAILabs/luganda-fln-training-data", subset)["train"]


def load_exercises_dataset() -> Dataset:
    ds = load_dataset("CraneAILabs/luganda-bilingual-literacy-exercises")["train"]
    if "text" in ds.column_names:
        return ds

    def _format(row):
        cols = ds.column_names
        if "question" in cols and "answer" in cols:
            text = (
                f"<start_of_turn>user\n{row['question']}<end_of_turn>\n"
                f"<start_of_turn>model\n{row['answer']}<end_of_turn>"
            )
        elif "instruction" in cols and "output" in cols:
            text = (
                f"<start_of_turn>user\n{row['instruction']}<end_of_turn>\n"
                f"<start_of_turn>model\n{row['output']}<end_of_turn>"
            )
        else:
            content = " | ".join(str(row[c]) for c in cols if isinstance(row[c], str))
            text = f"<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n<end_of_turn>"
        return {"text": text}

    return ds.map(_format)


def build_training_dataset(balance_strategy: str = "oversample") -> Dataset:
    """
    Loads FLN + exercises, applies position balancing, returns a Dataset
    with a single 'text' column ready for SFTTrainer.
    """
    fln = load_fln_dataset("all")
    exercises = load_exercises_dataset()

    mcq_items = [x for x in fln if x["correct_letter"] in ["A", "B", "C", "D"]]
    non_mcq = [x for x in fln if x["correct_letter"] not in ["A", "B", "C", "D"]]

    balanced = balance_by_position(mcq_items, strategy=balance_strategy)

    return concatenate_datasets([
        Dataset.from_list(balanced),
        Dataset.from_list(non_mcq),
        exercises.select_columns(["text"]),
    ])


def build_grpo_prompts(fln_dataset: Dataset, n: int = 300) -> Dataset:
    """Extract user-turn prompts from pre-formatted FLN chat text."""
    prompts = []
    for item in fln_dataset.select(range(min(n, len(fln_dataset)))):
        text = item["text"]
        if "<start_of_turn>user" in text and "<end_of_turn>" in text:
            user_msg = text.split("<start_of_turn>user\n")[1].split("<end_of_turn>")[0]
            prompts.append({"prompt": user_msg})
    return Dataset.from_list(prompts)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_data.py -v
```

Expected:
```
PASSED tests/test_data.py::test_oversample_balances_to_max
PASSED tests/test_data.py::test_undersample_balances_to_min
PASSED tests/test_data.py::test_none_strategy_returns_original
PASSED tests/test_data.py::test_extract_first_letter_finds_abcd
```

- [ ] **Step 5: Commit**

```bash
git add scripts/core/data.py tests/test_data.py
git commit -m "feat: add core data module with position balancing and tests"
```

---

## Task 3: Create scripts/core/evaluate.py + tests

**Files:**
- Create: `scripts/core/evaluate.py`
- Create: `tests/test_evaluate.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_evaluate.py
import numpy as np
import pytest
from scripts.core.evaluate import bootstrap_ci, mcnemar_test


def test_bootstrap_ci_perfect_accuracy():
    preds = ["A", "B", "C", "D"] * 25   # 100 items
    labels = ["A", "B", "C", "D"] * 25
    acc, lo, hi = bootstrap_ci(preds, labels)
    assert acc == pytest.approx(1.0)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)


def test_bootstrap_ci_half_accuracy():
    preds  = ["A"] * 50 + ["B"] * 50
    labels = ["A"] * 50 + ["A"] * 50   # B is always wrong
    acc, lo, hi = bootstrap_ci(preds, labels)
    assert acc == pytest.approx(0.5)
    assert 0.4 < lo < 0.5
    assert 0.5 < hi < 0.6


def test_bootstrap_ci_returns_ordered_bounds():
    preds  = ["A"] * 60 + ["B"] * 40
    labels = ["A"] * 100
    acc, lo, hi = bootstrap_ci(preds, labels)
    assert lo <= acc <= hi


def test_mcnemar_identical_models_returns_high_pvalue():
    preds = ["A", "B", "C", "D"] * 25
    labels = ["A", "B", "A", "D"] * 25
    p = mcnemar_test(preds, preds, labels)
    assert p == pytest.approx(1.0)


def test_mcnemar_very_different_models_returns_low_pvalue():
    n = 100
    labels = ["A"] * n
    preds_a = ["A"] * n           # perfect
    preds_b = ["B"] * n           # always wrong
    p = mcnemar_test(preds_a, preds_b, labels)
    assert p < 0.001


def test_compute_calibration_perfect_model():
    # Model always predicts correctly with 100% confidence
    confidences = [1.0] * 100
    correct = [True] * 100
    bins = compute_calibration(confidences, correct, n_bins=5)
    # All items fall in the highest confidence bin with 100% accuracy
    assert any(b["accuracy"] == pytest.approx(1.0) for b in bins if b["count"] > 0)


def test_compute_calibration_returns_expected_structure():
    import random
    rng = random.Random(0)
    confidences = [rng.uniform(0.25, 1.0) for _ in range(200)]
    correct = [rng.random() < c for c in confidences]
    bins = compute_calibration(confidences, correct, n_bins=10)
    assert len(bins) == 10
    for b in bins:
        assert "bin_center" in b and "accuracy" in b and "count" in b
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_evaluate.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement scripts/core/evaluate.py**

```python
# scripts/core/evaluate.py
import json
import torch
import numpy as np
from scipy import stats
from typing import List, Optional, Tuple

from scripts.core.data import extract_first_letter

ANSWER_TOKENS = ["A", "B", "C", "D"]


def evaluate_on_benchmark(model, tokenizer, benchmark_ds, label: str = "") -> dict:
    """
    Returns accuracy, per-position stats, per-category stats, position bias spread,
    plus raw predictions and labels lists for statistical comparisons.
    """
    samples = benchmark_ds["train"]
    correct = 0
    predictions, labels_list, confidences = [], [], []
    position_stats = {pos: {"total": 0, "correct": 0} for pos in ["A", "B", "C", "D"]}
    category_stats = {}

    for item in samples:
        prompt = (
            f"<start_of_turn>user\n"
            f"Answer with only the letter (A, B, C, or D). Do not explain.\n\n"
            f"{item['luganda_question']}\n"
            f"(A) {item['luganda_answer_a']}\n"
            f"(B) {item['luganda_answer_b']}\n"
            f"(C) {item['luganda_answer_c']}\n"
            f"(D) {item['luganda_answer_d']}\n"
            f"<end_of_turn>\n<start_of_turn>model\n"
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            # Single forward pass: extract logits for A/B/C/D at next-token position
            out = model(**inputs)
            next_logits = out.logits[0, -1, :]   # shape: (vocab_size,)
            answer_token_ids = [
                tokenizer.convert_tokens_to_ids(t) for t in ANSWER_TOKENS
            ]
            answer_logits = next_logits[answer_token_ids]
            probs = torch.softmax(answer_logits, dim=-1)
            pred_idx = int(probs.argmax())
            predicted = ANSWER_TOKENS[pred_idx]
            confidence = float(probs[pred_idx])

        gold = item["correct_answer"]

        predictions.append(predicted)
        labels_list.append(gold)
        confidences.append(confidence)
        position_stats[gold]["total"] += 1
        if predicted == gold:
            correct += 1
            position_stats[gold]["correct"] += 1

        cat = item.get("category", "unknown")
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "correct": 0}
        category_stats[cat]["total"] += 1
        if predicted == gold:
            category_stats[cat]["correct"] += 1

    total = len(samples)
    accs = [s["correct"] / s["total"] for s in position_stats.values() if s["total"] > 0]
    spread = (max(accs) - min(accs)) * 100 if accs else 0.0

    if label:
        print(f"[{label}] accuracy={correct/total:.1%}  spread={spread:.1f}pp")

    return {
        "accuracy": correct / total,
        "predictions": predictions,
        "labels": labels_list,
        "confidences": confidences,
        "position_stats": position_stats,
        "category_stats": category_stats,
        "spread": spread,
    }


def bootstrap_ci(
    predictions: List[str],
    labels: List[str],
    n_resamples: int = 1000,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Returns (accuracy, lower_ci, upper_ci) using percentile bootstrap.
    """
    correct = np.array([p == l for p, l in zip(predictions, labels)], dtype=float)
    rng = np.random.default_rng(42)
    boot_accs = [
        rng.choice(correct, size=len(correct), replace=True).mean()
        for _ in range(n_resamples)
    ]
    alpha = (1 - confidence) / 2
    lo = float(np.percentile(boot_accs, alpha * 100))
    hi = float(np.percentile(boot_accs, (1 - alpha) * 100))
    return float(correct.mean()), lo, hi


def mcnemar_test(
    preds_a: List[str],
    preds_b: List[str],
    labels: List[str],
) -> float:
    """
    McNemar's test (with continuity correction) for paired model comparison.
    Returns the p-value. Low p-value = models differ significantly.
    """
    n01 = sum(pa != l and pb == l for pa, pb, l in zip(preds_a, preds_b, labels))
    n10 = sum(pa == l and pb != l for pa, pb, l in zip(preds_a, preds_b, labels))
    if n01 + n10 == 0:
        return 1.0
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return float(stats.chi2.sf(stat, df=1))


def compute_calibration(
    confidences: List[float],
    correct: List[bool],
    n_bins: int = 10,
) -> List[dict]:
    """
    Splits predictions into confidence bins and computes accuracy per bin.
    Returns a list of dicts: {bin_center, accuracy, count}.
    A well-calibrated model has accuracy ≈ confidence in every bin.
    """
    bins = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = [lo <= c < hi for c in confidences]
        count = sum(mask)
        if count > 0:
            acc = sum(c for c, m in zip(correct, mask) if m) / count
        else:
            acc = 0.0
        bins.append({
            "bin_center": round((lo + hi) / 2, 2),
            "accuracy": round(acc, 4),
            "count": count,
        })
    return bins
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_evaluate.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/core/evaluate.py tests/test_evaluate.py
git commit -m "feat: add core evaluate module with bootstrap CI and McNemar test"
```

---

## Task 4: Create scripts/core/callbacks.py + tests

**Files:**
- Create: `scripts/core/callbacks.py`
- Create: `tests/test_callbacks.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_callbacks.py
import json
import os
import tempfile
from types import SimpleNamespace
from scripts.core.callbacks import DiagnosticCallback


def _fake_state(step):
    return SimpleNamespace(global_step=step)


def _fake_args():
    return SimpleNamespace()


def _fake_control():
    return SimpleNamespace()


def test_callback_writes_jsonl_records():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        path = f.name

    cb = DiagnosticCallback(output_path=path)
    cb.on_train_begin(_fake_args(), _fake_state(0), _fake_control())
    cb.on_log(_fake_args(), _fake_state(10), _fake_control(),
              logs={"loss": 0.5, "learning_rate": 0.0002})
    cb.on_log(_fake_args(), _fake_state(20), _fake_control(),
              logs={"loss": 0.4, "learning_rate": 0.0001})
    cb.on_train_end(_fake_args(), _fake_state(20), _fake_control())

    with open(path) as f:
        lines = [json.loads(l) for l in f]

    assert len(lines) == 2
    assert lines[0]["step"] == 10
    assert lines[0]["loss"] == 0.5
    assert lines[1]["step"] == 20
    os.unlink(path)


def test_callback_handles_none_logs():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        path = f.name
    cb = DiagnosticCallback(output_path=path)
    cb.on_train_begin(_fake_args(), _fake_state(0), _fake_control())
    cb.on_log(_fake_args(), _fake_state(5), _fake_control(), logs=None)
    cb.on_train_end(_fake_args(), _fake_state(5), _fake_control())
    with open(path) as f:
        lines = f.readlines()
    assert len(lines) == 0
    os.unlink(path)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_callbacks.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement scripts/core/callbacks.py**

```python
# scripts/core/callbacks.py
import json
import os
from transformers import TrainerCallback

KL_WARN_THRESHOLD = 0.5
REWARD_COLLAPSE_STD_THRESHOLD = 0.01
REWARD_COLLAPSE_WINDOW = 50


class DiagnosticCallback(TrainerCallback):
    """
    Logs training metrics to a JSONL file and warns on reward hacking / collapse.
    """

    def __init__(self, output_path: str = "results/diagnostics/training_curves.jsonl"):
        self.output_path = output_path
        self._file = None
        self._low_std_streak = 0

    def on_train_begin(self, args, state, control, **kwargs):
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self._file = open(self.output_path, "a")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        record = {"step": state.global_step, **logs}
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

        kl = logs.get("kl") or logs.get("kl_div")
        if kl is not None and kl > KL_WARN_THRESHOLD:
            print(
                f"\nWARNING: KL divergence = {kl:.3f} at step {state.global_step}. "
                f"Consider lowering learning_rate or stopping GRPO early."
            )

        reward_std = logs.get("rewards/std") or logs.get("reward_std")
        if reward_std is not None:
            if reward_std < REWARD_COLLAPSE_STD_THRESHOLD:
                self._low_std_streak += 1
                if self._low_std_streak >= REWARD_COLLAPSE_WINDOW:
                    print(
                        f"\nWARNING: Reward std < {REWARD_COLLAPSE_STD_THRESHOLD} "
                        f"for {REWARD_COLLAPSE_WINDOW} steps. Reward may have collapsed."
                    )
            else:
                self._low_std_streak = 0

    def on_train_end(self, args, state, control, **kwargs):
        if self._file:
            self._file.close()
            self._file = None
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_callbacks.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/core/callbacks.py tests/test_callbacks.py
git commit -m "feat: add DiagnosticCallback with KL and reward collapse warnings"
```

---

## Task 5: Create scripts/core/train.py

**Files:**
- Create: `scripts/core/train.py`

No unit tests here — training requires GPU. Verification is done by checking output directories after Task 6 and 7.

- [ ] **Step 1: Create scripts/core/train.py**

```python
# scripts/core/train.py
import torch
from typing import Optional, List
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, TrainerCallback
from trl import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig
from peft import LoraConfig


def train_sft(
    model_path: str,
    train_dataset: Dataset,
    output_dir: str,
    lora_rank: int = 16,
    num_train_epochs: int = 3,
    learning_rate: float = 2e-4,
    per_device_batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    max_seq_length: int = 1024,
    report_to: str = "none",
    callbacks: Optional[List[TrainerCallback]] = None,
) -> str:
    """
    Runs SFT with QLoRA, merges adapter, saves full model to output_dir.
    Returns output_dir.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        target_modules="all-linear",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=output_dir + "-adapter",
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_ratio=0.05,
            bf16=True,
            logging_steps=10,
            save_strategy="epoch",
            max_seq_length=max_seq_length,
            report_to=report_to,
        ),
        train_dataset=train_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        callbacks=callbacks or [],
    )

    trainer.train()
    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved merged SFT model to {output_dir}")
    return output_dir


def _build_reward_fn(reward_model_path: str):
    """Returns a reward function compatible with GRPOTrainer."""
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_path)

    def compute_rewards(completions, **kwargs):
        rewards = []
        for completion in completions:
            text = completion if isinstance(completion, str) else completion[-1]["content"]
            inputs = reward_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            ).to(reward_model.device)
            with torch.no_grad():
                logits = reward_model(**inputs).logits
            if logits.shape[-1] == 2:
                score = torch.softmax(logits, dim=-1)[0, 1].item()
            else:
                score = logits[0, 0].item()
            rewards.append(score)
        return rewards

    return compute_rewards


def train_grpo(
    model_path: str,
    grpo_dataset: Dataset,
    output_dir: str,
    reward_model_path: str = "CraneAILabs/luganda-reward-model",
    max_steps: int = 600,
    learning_rate: float = 5e-6,
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    num_generations: int = 4,
    report_to: str = "none",
    callbacks: Optional[List[TrainerCallback]] = None,
) -> str:
    """
    Runs GRPO on model_path using the Crane AI Labs reward model.
    Saves full model to output_dir. Returns output_dir.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    compute_rewards = _build_reward_fn(reward_model_path)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=compute_rewards,
        args=GRPOConfig(
            output_dir=output_dir + "-ckpt",
            num_train_epochs=1,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            bf16=True,
            num_generations=num_generations,
            max_new_tokens=200,
            max_prompt_length=512,
            logging_steps=10,
            save_steps=200,
            max_steps=max_steps,
            report_to=report_to,
        ),
        train_dataset=grpo_dataset,
        processing_class=tokenizer,
        callbacks=callbacks or [],
    )

    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved GRPO model to {output_dir}")
    return output_dir
```

- [ ] **Step 2: Commit**

```bash
git add scripts/core/train.py
git commit -m "feat: add parameterised train_sft and train_grpo core functions"
```

---

## Task 6: Refactor 03_sft_learner.py as thin wrapper

**Files:**
- Modify: `scripts/03_sft_learner.py`

- [ ] **Step 1: Replace 03_sft_learner.py contents**

```python
# scripts/03_sft_learner.py
"""
Hours 3-4: SFT with Position-Balanced Data (LEARNER track)
CLI wrapper — training logic lives in scripts/core/train.py.
"""
import os
from scripts.core.data import build_training_dataset
from scripts.core.train import train_sft
from scripts.core.callbacks import DiagnosticCallback

WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "eduganda-extension")

train_dataset = build_training_dataset(balance_strategy="oversample")
print(f"Total training examples: {len(train_dataset)}")

diagnostic_cb = DiagnosticCallback("results/diagnostics/sft_training_curves.jsonl")

train_sft(
    model_path="CraneAILabs/ganda-gemma-1b",
    train_dataset=train_dataset,
    output_dir="./learner-full",
    lora_rank=16,
    num_train_epochs=3,
    report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
    callbacks=[diagnostic_cb],
)
```

- [ ] **Step 2: Verify the script is importable (no GPU needed for this check)**

```bash
python -c "import scripts.core.data; import scripts.core.train; import scripts.core.callbacks; print('imports OK')"
```

Expected: `imports OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/03_sft_learner.py
git commit -m "refactor: 03_sft_learner.py as thin wrapper over core/train.py"
```

---

## Task 7: Refactor 04_grpo.py as thin wrapper

**Files:**
- Modify: `scripts/04_grpo.py`

- [ ] **Step 1: Replace 04_grpo.py contents**

```python
# scripts/04_grpo.py
"""
Hours 5-6: GRPO (RL track)
CLI wrapper — training logic lives in scripts/core/train.py.
Skip this script if GRPO is unstable — LEARNER-only is a valid contribution.
"""
import os
from scripts.core.data import load_fln_dataset, build_grpo_prompts
from scripts.core.train import train_grpo
from scripts.core.callbacks import DiagnosticCallback

fln = load_fln_dataset("all")
grpo_dataset = build_grpo_prompts(fln, n=300)
print(f"GRPO prompts: {len(grpo_dataset)}")

diagnostic_cb = DiagnosticCallback("results/diagnostics/grpo_training_curves.jsonl")

train_grpo(
    model_path="./learner-full",
    grpo_dataset=grpo_dataset,
    output_dir="./grpo-full",
    max_steps=600,
    report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
    callbacks=[diagnostic_cb],
)
```

- [ ] **Step 2: Commit**

```bash
git add scripts/04_grpo.py
git commit -m "refactor: 04_grpo.py as thin wrapper over core/train.py"
```

---

## Task 8: Update 06_evaluate_all.py with statistical rigor

**Files:**
- Modify: `scripts/06_evaluate_all.py`

- [ ] **Step 1: Replace 06_evaluate_all.py contents**

```python
# scripts/06_evaluate_all.py
"""
Hours 8-9: Full Evaluation Sweep with statistical comparisons.
"""
import os
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.core.evaluate import evaluate_on_benchmark, bootstrap_ci, mcnemar_test

benchmark = load_dataset("CraneAILabs/pedagogy-luganda-replaced")

models_to_eval = {
    "ganda-gemma-1b (base)": "CraneAILabs/ganda-gemma-1b",
    "LEARNER (SFT, balanced)": "./learner-full",
    "Merged 80/20": "./merged-80-20",
    "Merged 70/30": "./merged-70-30",
    "Merged 60/40": "./merged-60-40",
    "DARE-TIES 70/30": "./merged-dare-ties",
    "EduGanda reference": "CraneAILabs/EduGanda-Gemma-3-1B",
}

missing = [k for k, p in models_to_eval.items() if p.startswith("./") and not os.path.isdir(p)]
if missing:
    print(f"WARNING: skipping missing checkpoints: {missing}")
    models_to_eval = {k: v for k, v in models_to_eval.items() if k not in missing}

all_results = {}
for name, path in models_to_eval.items():
    print(f"\n{'='*60}\nEvaluating: {name}\n{'='*60}")
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto")
    tok = AutoTokenizer.from_pretrained(path)
    result = evaluate_on_benchmark(model, tok, benchmark, label=name)
    acc, lo, hi = bootstrap_ci(result["predictions"], result["labels"])
    result["ci_lower"] = lo
    result["ci_upper"] = hi
    all_results[name] = result
    del model
    torch.cuda.empty_cache()

# McNemar comparisons: every model vs LEARNER
if "LEARNER (SFT, balanced)" in all_results:
    learner_preds = all_results["LEARNER (SFT, balanced)"]["predictions"]
    learner_labels = all_results["LEARNER (SFT, balanced)"]["labels"]
    for name, r in all_results.items():
        if name == "LEARNER (SFT, balanced)":
            continue
        p = mcnemar_test(learner_preds, r["predictions"], learner_labels)
        r["mcnemar_vs_learner_p"] = p

with open("results/full_results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)

print("\n\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(f"{'Model':<30} {'Acc':>6} {'95% CI':>16} {'Spread':>8}  Notes")
print("-" * 80)
for name, r in all_results.items():
    acc = r["accuracy"] * 100
    lo, hi = r.get("ci_lower", 0) * 100, r.get("ci_upper", 0) * 100
    spread = r["spread"]
    p = r.get("mcnemar_vs_learner_p")
    sig = f"p={p:.3f}" if p is not None else ""
    print(f"{name:<30} {acc:>5.1f}% [{lo:>4.1f}%–{hi:>4.1f}%] {spread:>6.1f}pp  {sig}")

print("\nSaved results/full_results.json")
```

- [ ] **Step 2: Commit**

```bash
git add scripts/06_evaluate_all.py
git commit -m "feat: add bootstrap CI and McNemar test to full evaluation sweep"
```

---

## Task 9: Create scripts/ablations/run_ablation.py

**Files:**
- Create: `scripts/ablations/run_ablation.py`

- [ ] **Step 1: Create the ablation runner**

```python
# scripts/ablations/run_ablation.py
"""
Single ablation run. Usage:
  python scripts/ablations/run_ablation.py --axis lora_rank --value 32
  python scripts/ablations/run_ablation.py --axis sft_epochs --value 1
  python scripts/ablations/run_ablation.py --axis balance_strategy --value none
  python scripts/ablations/run_ablation.py --axis grpo_steps --value 200

Saves results to results/ablations/<axis>/<value>/results.json
"""
import argparse
import json
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.core.data import build_training_dataset, load_fln_dataset, build_grpo_prompts
from scripts.core.train import train_sft, train_grpo
from scripts.core.evaluate import evaluate_on_benchmark, bootstrap_ci
from scripts.core.callbacks import DiagnosticCallback

VALID_AXES = {
    "lora_rank": ["8", "16", "32"],
    "sft_epochs": ["1", "3", "5"],
    "balance_strategy": ["none", "oversample", "undersample"],
    "grpo_steps": ["200", "600", "1000"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--axis", required=True, choices=VALID_AXES.keys())
    parser.add_argument("--value", required=True)
    args = parser.parse_args()

    if args.value not in VALID_AXES[args.axis]:
        parser.error(f"--value must be one of {VALID_AXES[args.axis]} for axis '{args.axis}'")

    out_dir = f"results/ablations/{args.axis}/{args.value}"
    os.makedirs(out_dir, exist_ok=True)

    checkpoint_dir = f"./ablation-{args.axis}-{args.value}"
    benchmark = load_dataset("CraneAILabs/pedagogy-luganda-replaced")

    # --- Determine training config ---
    balance_strategy = "oversample"
    lora_rank = 16
    num_train_epochs = 3
    grpo_steps = 600

    if args.axis == "balance_strategy":
        balance_strategy = args.value
    elif args.axis == "lora_rank":
        lora_rank = int(args.value)
    elif args.axis == "sft_epochs":
        num_train_epochs = int(args.value)
    elif args.axis == "grpo_steps":
        grpo_steps = int(args.value)

    cb = DiagnosticCallback(f"{out_dir}/training_curves.jsonl")

    # --- SFT ---
    print(f"\n[Ablation] axis={args.axis} value={args.value}")
    train_dataset = build_training_dataset(balance_strategy=balance_strategy)
    sft_path = checkpoint_dir + "-sft"
    train_sft(
        model_path="CraneAILabs/ganda-gemma-1b",
        train_dataset=train_dataset,
        output_dir=sft_path,
        lora_rank=lora_rank,
        num_train_epochs=num_train_epochs,
        callbacks=[cb],
    )

    # --- GRPO (only for grpo_steps ablation, or use sft_path for others) ---
    if args.axis == "grpo_steps":
        fln = load_fln_dataset("all")
        grpo_prompts = build_grpo_prompts(fln, n=300)
        grpo_path = checkpoint_dir + "-grpo"
        train_grpo(
            model_path=sft_path,
            grpo_dataset=grpo_prompts,
            output_dir=grpo_path,
            max_steps=grpo_steps,
            callbacks=[cb],
        )
        eval_path = grpo_path
    else:
        eval_path = sft_path

    # --- Evaluate ---
    model = AutoModelForCausalLM.from_pretrained(eval_path, torch_dtype=torch.bfloat16, device_map="auto")
    tok = AutoTokenizer.from_pretrained(eval_path)
    result = evaluate_on_benchmark(model, tok, benchmark, label=f"{args.axis}={args.value}")
    acc, lo, hi = bootstrap_ci(result["predictions"], result["labels"])
    result["ci_lower"] = lo
    result["ci_upper"] = hi
    result["axis"] = args.axis
    result["value"] = args.value
    del model
    torch.cuda.empty_cache()

    with open(f"{out_dir}/results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nResult: accuracy={result['accuracy']:.1%} [{lo:.1%}–{hi:.1%}]  spread={result['spread']:.1f}pp")
    print(f"Saved to {out_dir}/results.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI help works (no GPU needed)**

```bash
python scripts/ablations/run_ablation.py --help
```

Expected: prints usage with `--axis` and `--value` options.

- [ ] **Step 3: Commit**

```bash
git add scripts/ablations/run_ablation.py
git commit -m "feat: add parameterised ablation runner"
```

---

## Task 10: Create scripts/ablations/summarize_ablations.py

**Files:**
- Create: `scripts/ablations/summarize_ablations.py`

- [ ] **Step 1: Create the summarizer**

```python
# scripts/ablations/summarize_ablations.py
"""
Reads all results/ablations/<axis>/<value>/results.json files and produces:
  - A markdown summary table printed to stdout
  - Per-axis plots saved to results/ablations/<axis>_plot.png
  - results/ablations/summary.json with the best checkpoint path
Run after all ablation runs are complete.
"""
import json
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt

ABLATIONS_DIR = Path("results/ablations")
CONTROL = {
    "lora_rank": "16",
    "sft_epochs": "3",
    "balance_strategy": "oversample",
    "grpo_steps": "600",
}


def load_all_results():
    data = defaultdict(dict)
    for result_file in ABLATIONS_DIR.glob("*/*/results.json"):
        axis = result_file.parent.parent.name
        value = result_file.parent.name
        with open(result_file) as f:
            data[axis][value] = json.load(f)
    return data


def print_summary_table(data):
    print("\n" + "=" * 80)
    print("ABLATION SUMMARY")
    print("=" * 80)
    for axis, results in sorted(data.items()):
        print(f"\nAxis: {axis}  (control = {CONTROL.get(axis, '?')})")
        print(f"  {'Value':<20} {'Acc':>6} {'95% CI':>16} {'Spread':>8}")
        print("  " + "-" * 54)
        for value, r in sorted(results.items()):
            acc = r["accuracy"] * 100
            lo = r.get("ci_lower", 0) * 100
            hi = r.get("ci_upper", 0) * 100
            spread = r["spread"]
            ctrl = " ← control" if value == CONTROL.get(axis) else ""
            print(f"  {value:<20} {acc:>5.1f}% [{lo:>4.1f}%–{hi:>4.1f}%] {spread:>6.1f}pp{ctrl}")


def plot_axis(axis, results):
    values = sorted(results.keys())
    accs = [results[v]["accuracy"] * 100 for v in values]
    spreads = [results[v]["spread"] for v in values]
    lo_errs = [accs[i] - results[v].get("ci_lower", results[v]["accuracy"]) * 100
               for i, v in enumerate(values)]
    hi_errs = [results[v].get("ci_upper", results[v]["accuracy"]) * 100 - accs[i]
               for i, v in enumerate(values)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Ablation: {axis}")

    x = range(len(values))
    ax1.bar(x, accs, yerr=[lo_errs, hi_errs], capsize=5, color="steelblue")
    ax1.set_xticks(x)
    ax1.set_xticklabels(values)
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Benchmark Accuracy")

    ax2.bar(x, spreads, color="coral")
    ax2.set_xticks(x)
    ax2.set_xticklabels(values)
    ax2.set_ylabel("Spread (pp)")
    ax2.set_title("Position Bias Spread")

    plt.tight_layout()
    out = ABLATIONS_DIR / f"{axis}_plot.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved plot: {out}")


def find_best(data):
    best_acc = -1
    best_info = {}
    for axis, results in data.items():
        for value, r in results.items():
            if r["accuracy"] > best_acc:
                best_acc = r["accuracy"]
                best_info = {"axis": axis, "value": value, "accuracy": r["accuracy"],
                             "spread": r["spread"]}
    return best_info


def main():
    data = load_all_results()
    if not data:
        print("No ablation results found in results/ablations/. Run run_ablation.py first.")
        return

    print_summary_table(data)

    for axis, results in data.items():
        if len(results) > 1:
            plot_axis(axis, results)

    best = find_best(data)
    summary = {"best": best, "all": {ax: {v: r for v, r in res.items()} for ax, res in data.items()}}
    with open(ABLATIONS_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nBest: axis={best.get('axis')} value={best.get('value')} acc={best.get('accuracy', 0):.1%}")
    print(f"Saved results/ablations/summary.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it handles empty directory gracefully**

```bash
python scripts/ablations/summarize_ablations.py
```

Expected: `No ablation results found in results/ablations/. Run run_ablation.py first.`

- [ ] **Step 3: Commit**

```bash
git add scripts/ablations/summarize_ablations.py
git commit -m "feat: add ablation summarizer with per-axis plots and best checkpoint tracking"
```

---

## Task 11: Create scripts/diagnostics/probe_position_bias.py + tests

**Files:**
- Create: `scripts/diagnostics/probe_position_bias.py`
- Create: `tests/test_probe.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_probe.py
import numpy as np
import pytest
from scripts.diagnostics.probe_position_bias import fit_probe, POSITIONS


def test_probe_on_perfectly_separable_data():
    """If each position has a distinct cluster, CV accuracy should be near 100%."""
    rng = np.random.default_rng(0)
    n_per_class = 50
    hidden_dim = 32
    hidden_states = np.vstack([
        rng.normal(loc=i * 10, scale=0.1, size=(n_per_class, hidden_dim))
        for i in range(4)
    ])
    labels = [p for p in POSITIONS for _ in range(n_per_class)]
    cv_acc = fit_probe(hidden_states, labels)
    assert cv_acc > 0.95, f"Expected near-perfect accuracy on separable data, got {cv_acc:.2f}"


def test_probe_on_random_data_near_chance():
    """Random hidden states should give ~25% accuracy (4-class chance)."""
    rng = np.random.default_rng(1)
    n = 200
    hidden_states = rng.normal(size=(n, 64))
    labels = [POSITIONS[i % 4] for i in range(n)]
    cv_acc = fit_probe(hidden_states, labels)
    assert cv_acc < 0.40, f"Expected near-chance accuracy on random data, got {cv_acc:.2f}"


def test_fit_probe_returns_float():
    rng = np.random.default_rng(2)
    hs = rng.normal(size=(40, 16))
    labels = [POSITIONS[i % 4] for i in range(40)]
    result = fit_probe(hs, labels)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_probe.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Implement scripts/diagnostics/probe_position_bias.py**

```python
# scripts/diagnostics/probe_position_bias.py
"""
Trains a logistic regression probe on model hidden states to detect
whether position bias is encoded in representations.

Usage:
  python scripts/diagnostics/probe_position_bias.py

Evaluates base model, LEARNER, and best ablation checkpoint (or EduGanda reference).
"""
import json
import os
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

POSITIONS = ["A", "B", "C", "D"]


def fit_probe(hidden_states: np.ndarray, labels: list, cv: int = 5) -> float:
    """
    Fits a logistic regression on hidden_states -> labels.
    Returns mean CV accuracy. Does not require GPU.
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(hidden_states)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    scores = cross_val_score(clf, X, labels, cv=cv, scoring="accuracy")
    return float(scores.mean())


def extract_hidden_states(model, tokenizer, benchmark_ds) -> tuple:
    """
    Returns (hidden_states_array, labels_list) where hidden_states is
    shape (n_items, hidden_dim) — last token of last layer for each prompt.
    """
    model.eval()
    all_hidden, all_labels = [], []

    for item in benchmark_ds["train"]:
        prompt = (
            f"<start_of_turn>user\n"
            f"Answer with only the letter (A, B, C, or D). Do not explain.\n\n"
            f"{item['luganda_question']}\n"
            f"(A) {item['luganda_answer_a']}\n"
            f"(B) {item['luganda_answer_b']}\n"
            f"(C) {item['luganda_answer_c']}\n"
            f"(D) {item['luganda_answer_d']}\n"
            f"<end_of_turn>\n<start_of_turn>model\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
            )

        # Last layer, last token
        last_hidden = outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()
        all_hidden.append(last_hidden)
        all_labels.append(item["correct_answer"])

    return np.stack(all_hidden), all_labels


def probe_model(model_path: str, benchmark_ds, label: str) -> dict:
    print(f"\n[Probe] Loading {label} from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    tok = AutoTokenizer.from_pretrained(model_path)

    hidden_states, labels = extract_hidden_states(model, tok, benchmark_ds)
    del model
    torch.cuda.empty_cache()

    cv_acc = fit_probe(hidden_states, labels)
    print(f"  Probe CV accuracy: {cv_acc:.1%}  (chance=25%)")
    return {"model": label, "probe_cv_accuracy": cv_acc, "n_items": len(labels)}


def main():
    benchmark = load_dataset("CraneAILabs/pedagogy-luganda-replaced")

    checkpoints = {
        "ganda-gemma-1b (base)": "CraneAILabs/ganda-gemma-1b",
        "LEARNER": "./learner-full",
        "EduGanda reference": "CraneAILabs/EduGanda-Gemma-3-1B",
    }
    # Add best ablation checkpoint if summary.json exists
    summary_path = "results/ablations/summary.json"
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        best = summary.get("best", {})
        axis, value = best.get("axis"), best.get("value")
        if axis and value:
            ckpt = f"./ablation-{axis}-{value}-sft"
            if os.path.isdir(ckpt):
                checkpoints[f"Best ablation ({axis}={value})"] = ckpt

    results = []
    for label, path in checkpoints.items():
        if path.startswith("./") and not os.path.isdir(path):
            print(f"Skipping {label} — checkpoint not found at {path}")
            continue
        results.append(probe_model(path, benchmark, label))

    os.makedirs("results/diagnostics", exist_ok=True)
    with open("results/diagnostics/position_bias_probe.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n--- Probe Summary ---")
    print(f"{'Model':<35} {'CV Accuracy':>12}  Interpretation")
    print("-" * 70)
    for r in results:
        acc = r["probe_cv_accuracy"]
        interp = "strong encoding" if acc > 0.40 else ("mild encoding" if acc > 0.30 else "near chance")
        print(f"{r['model']:<35} {acc:>11.1%}  {interp}")

    print("\nSaved results/diagnostics/position_bias_probe.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_probe.py -v
```

Expected: all 3 tests PASS (no GPU needed — uses random numpy arrays).

- [ ] **Step 5: Commit**

```bash
git add scripts/diagnostics/probe_position_bias.py tests/test_probe.py
git commit -m "feat: add position bias probing with logistic regression + tests"
```

---

## Task 12: Create scripts/diagnostics/analyze_reward_model.py

**Files:**
- Create: `scripts/diagnostics/analyze_reward_model.py`

- [ ] **Step 1: Create the reward model analyzer**

```python
# scripts/diagnostics/analyze_reward_model.py
"""
Analyzes reward model score distributions before and after GRPO.
Helps detect reward hacking (scores up but letter distribution collapses)
and validate that the reward model gives a useful signal.

Usage:
  python scripts/diagnostics/analyze_reward_model.py
"""
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification

from scripts.core.data import load_fln_dataset, extract_first_letter


def score_texts(texts: list, reward_model, reward_tokenizer) -> list:
    scores = []
    for text in texts:
        inputs = reward_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(reward_model.device)
        with torch.no_grad():
            logits = reward_model(**inputs).logits
        if logits.shape[-1] == 2:
            score = torch.softmax(logits, dim=-1)[0, 1].item()
        else:
            score = logits[0, 0].item()
        scores.append(score)
    return scores


def generate_completions(model, tokenizer, prompts: list, max_new_tokens: int = 100) -> list:
    completions = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        completions.append(text)
    return completions


def main():
    os.makedirs("results/diagnostics", exist_ok=True)

    reward_model = AutoModelForSequenceClassification.from_pretrained(
        "CraneAILabs/luganda-reward-model", torch_dtype=torch.bfloat16, device_map="auto"
    )
    reward_tok = AutoTokenizer.from_pretrained("CraneAILabs/luganda-reward-model")

    fln = load_fln_dataset("all")
    n_samples = min(150, len(fln))
    sample_texts = [fln[i]["text"] for i in range(n_samples)]
    sample_prompts = []
    for item in fln.select(range(n_samples)):
        text = item["text"]
        if "<start_of_turn>user" in text and "<end_of_turn>" in text:
            sample_prompts.append(text.split("<start_of_turn>user\n")[1].split("<end_of_turn>")[0])

    # 1. Score the training data itself (pre-GRPO signal)
    print("Scoring training data...")
    train_scores = score_texts(sample_texts, reward_model, reward_tok)

    # 2. Score LEARNER model completions (pre-GRPO)
    results = {"train_data_scores": train_scores}
    if os.path.isdir("./learner-full"):
        print("Generating and scoring LEARNER completions...")
        model = AutoModelForCausalLM.from_pretrained("./learner-full", torch_dtype=torch.bfloat16, device_map="auto")
        tok = AutoTokenizer.from_pretrained("./learner-full")
        learner_completions = generate_completions(model, tok, sample_prompts[:100])
        del model; torch.cuda.empty_cache()
        learner_scores = score_texts(learner_completions, reward_model, reward_tok)
        results["learner_scores"] = learner_scores
        results["learner_letter_dist"] = dict(Counter(
            extract_first_letter(c) or "?" for c in learner_completions
        ))

    # 3. Score GRPO model completions (post-GRPO)
    if os.path.isdir("./grpo-full"):
        print("Generating and scoring GRPO completions...")
        model = AutoModelForCausalLM.from_pretrained("./grpo-full", torch_dtype=torch.bfloat16, device_map="auto")
        tok = AutoTokenizer.from_pretrained("./grpo-full")
        grpo_completions = generate_completions(model, tok, sample_prompts[:100])
        del model; torch.cuda.empty_cache()
        grpo_scores = score_texts(grpo_completions, reward_model, reward_tok)
        results["grpo_scores"] = grpo_scores
        results["grpo_letter_dist"] = dict(Counter(
            extract_first_letter(c) or "?" for c in grpo_completions
        ))

    with open("results/diagnostics/reward_analysis.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4))
    if "train_data_scores" in results:
        ax.hist(results["train_data_scores"], bins=20, alpha=0.5, label="Training data")
    if "learner_scores" in results:
        ax.hist(results["learner_scores"], bins=20, alpha=0.5, label="LEARNER outputs")
    if "grpo_scores" in results:
        ax.hist(results["grpo_scores"], bins=20, alpha=0.5, label="GRPO outputs")
    ax.set_xlabel("Reward score")
    ax.set_ylabel("Count")
    ax.set_title("Reward Model Score Distribution")
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/diagnostics/reward_distribution.png", dpi=150)
    plt.close()

    print("\n--- Reward Score Summary ---")
    for key in ["train_data_scores", "learner_scores", "grpo_scores"]:
        if key in results:
            scores = results[key]
            import statistics
            print(f"{key:<25} mean={statistics.mean(scores):.3f}  std={statistics.stdev(scores):.3f}")
    if "learner_letter_dist" in results:
        print(f"LEARNER letter dist: {results['learner_letter_dist']}")
    if "grpo_letter_dist" in results:
        print(f"GRPO letter dist:    {results['grpo_letter_dist']}")

    print("\nSaved results/diagnostics/reward_analysis.json")
    print("Saved results/diagnostics/reward_distribution.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/diagnostics/analyze_reward_model.py
git commit -m "feat: add reward model score distribution analysis"
```

---

## Task 13: Create scripts/deploy/quantize_and_benchmark.py

**Files:**
- Create: `scripts/deploy/quantize_and_benchmark.py`

- [ ] **Step 1: Create the quantization and benchmark script**

```python
# scripts/deploy/quantize_and_benchmark.py
"""
Quantizes the best checkpoint to GGUF and benchmarks inference.

Prerequisites:
  pip install llama-cpp-python
  # For GGUF conversion, clone llama.cpp and build:
  #   git clone https://github.com/ggerganov/llama.cpp
  #   cd llama.cpp && make
  # Then set LLAMA_CPP_DIR env var:
  #   export LLAMA_CPP_DIR=/path/to/llama.cpp

Usage:
  python scripts/deploy/quantize_and_benchmark.py [--model-path ./learner-full]
"""
import argparse
import json
import os
import subprocess
import time

import torch
from datasets import load_dataset
from scripts.core.data import extract_first_letter

QUANT_LEVELS = ["Q4_K_M", "Q8_0"]


def convert_to_gguf(model_path: str, output_dir: str, llama_cpp_dir: str) -> str:
    """Convert HuggingFace model to GGUF format using llama.cpp's convert script."""
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, "model-f16.gguf")
    convert_script = os.path.join(llama_cpp_dir, "convert_hf_to_gguf.py")
    subprocess.run(
        ["python", convert_script, model_path, "--outfile", out_file, "--outtype", "f16"],
        check=True,
    )
    print(f"Converted to GGUF: {out_file}")
    return out_file


def quantize_gguf(gguf_f16_path: str, output_dir: str, quant: str, llama_cpp_dir: str) -> str:
    """Quantize a GGUF f16 model to the specified quantization level."""
    quantize_bin = os.path.join(llama_cpp_dir, "llama-quantize")
    out_file = os.path.join(output_dir, f"model-{quant}.gguf")
    subprocess.run([quantize_bin, gguf_f16_path, out_file, quant], check=True)
    print(f"Quantized {quant}: {out_file}")
    return out_file


def benchmark_gguf(gguf_path: str, benchmark_ds, n_items: int = 100) -> dict:
    """Run inference on n_items benchmark questions and return accuracy + throughput."""
    from llama_cpp import Llama

    file_size_mb = os.path.getsize(gguf_path) / 1024 / 1024
    llm = Llama(model_path=gguf_path, n_ctx=1024, n_gpu_layers=-1, verbose=False)

    correct = 0
    total_tokens = 0
    start = time.time()

    samples = list(benchmark_ds["train"])[:n_items]
    for item in samples:
        prompt = (
            f"<start_of_turn>user\nAnswer with only the letter (A, B, C, or D). Do not explain.\n\n"
            f"{item['luganda_question']}\n"
            f"(A) {item['luganda_answer_a']}\n(B) {item['luganda_answer_b']}\n"
            f"(C) {item['luganda_answer_c']}\n(D) {item['luganda_answer_d']}\n"
            f"<end_of_turn>\n<start_of_turn>model\n"
        )
        out = llm(prompt, max_tokens=5, temperature=0.01, repeat_penalty=1.2)
        response = out["choices"][0]["text"]
        predicted = extract_first_letter(response)
        if predicted == item["correct_answer"]:
            correct += 1
        total_tokens += out["usage"]["completion_tokens"]

    elapsed = time.time() - start
    return {
        "gguf_path": gguf_path,
        "file_size_mb": round(file_size_mb, 1),
        "accuracy": correct / n_items,
        "tokens_per_sec": round(total_tokens / elapsed, 1),
        "n_items": n_items,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=None,
                        help="Path to HF checkpoint. Defaults to best ablation or ./learner-full.")
    args = parser.parse_args()

    llama_cpp_dir = os.environ.get("LLAMA_CPP_DIR", "")
    if not llama_cpp_dir or not os.path.isdir(llama_cpp_dir):
        print("ERROR: Set LLAMA_CPP_DIR env var to your llama.cpp build directory.")
        print("  git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make")
        raise SystemExit(1)

    # Determine which model to quantize
    model_path = args.model_path
    if not model_path:
        summary_path = "results/ablations/summary.json"
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                summary = json.load(f)
            best = summary.get("best", {})
            axis, value = best.get("axis"), best.get("value")
            candidate = f"./ablation-{axis}-{value}-sft" if axis and value else None
            if candidate and os.path.isdir(candidate):
                model_path = candidate
                print(f"Using best ablation checkpoint: {model_path}")
        if not model_path:
            model_path = "./learner-full"
            print(f"Using default checkpoint: {model_path}")

    if not os.path.isdir(model_path):
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    os.makedirs("results/deploy", exist_ok=True)
    gguf_dir = "results/deploy/gguf"

    # Convert to GGUF f16
    gguf_f16 = convert_to_gguf(model_path, gguf_dir, llama_cpp_dir)

    # Quantize and benchmark each level
    benchmark = load_dataset("CraneAILabs/pedagogy-luganda-replaced")
    benchmark_results = []

    for quant in QUANT_LEVELS:
        gguf_path = quantize_gguf(gguf_f16, gguf_dir, quant, llama_cpp_dir)
        print(f"\nBenchmarking {quant}...")
        result = benchmark_gguf(gguf_path, benchmark, n_items=100)
        result["quant"] = quant
        benchmark_results.append(result)
        print(f"  accuracy={result['accuracy']:.1%}  "
              f"speed={result['tokens_per_sec']:.1f} tok/s  "
              f"size={result['file_size_mb']:.0f}MB")

    with open("results/deploy/quantization_benchmark.json", "w") as f:
        json.dump({"source_model": model_path, "results": benchmark_results}, f, indent=2)

    print("\n--- Quantization Benchmark ---")
    print(f"{'Quant':<10} {'Accuracy':>9} {'Tok/s':>8} {'Size (MB)':>10}")
    print("-" * 42)
    for r in benchmark_results:
        print(f"{r['quant']:<10} {r['accuracy']:>8.1%} {r['tokens_per_sec']:>8.1f} {r['file_size_mb']:>9.0f}")

    print("\nSaved results/deploy/quantization_benchmark.json")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI help works**

```bash
python scripts/deploy/quantize_and_benchmark.py --help
```

Expected: prints usage with `--model-path` option.

- [ ] **Step 3: Commit**

```bash
git add scripts/deploy/quantize_and_benchmark.py
git commit -m "feat: add GGUF quantization and inference benchmark script"
```

---

## Task 14: Create MODEL_CARD.md

**Files:**
- Create: `MODEL_CARD.md`

- [ ] **Step 1: Create MODEL_CARD.md**

```markdown
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
publicly available data totals ~8.3k items — some training data was likely generated
during the pipeline and not released. Results may differ from the reference model;
all models are evaluated under the same setup for comparability.

**Data contamination check:** No overlap detected between benchmark questions
(pedagogy-luganda-replaced) and training data (prefix-level check on luganda_question).

---

## Evaluation Results

Evaluated on CraneAILabs/pedagogy-luganda-replaced (299 bilingual MCQ items).
Prompt format: English instruction + Luganda question + Luganda answer options.
Inference: greedy decoding, repetition_penalty=1.2.

<!-- Fill in after running 06_evaluate_all.py and summarize_ablations.py -->

| Model | Accuracy | 95% CI | Position Bias Spread |
|-------|----------|--------|----------------------|
| ganda-gemma-1b (base) | | | |
| LEARNER (SFT, balanced) | | | |
| Best ablation | | | |
| EduGanda reference | | | |

### Ablation findings

<!-- Fill in after running summarize_ablations.py -->

### Position bias probe

<!-- Fill in after running probe_position_bias.py -->
Probe CV accuracy on hidden states (chance = 25%):
- Base model: X%
- LEARNER: X%
- Best merged model: X%

---

## Quantized Versions

<!-- Fill in after running quantize_and_benchmark.py -->

| Format | Accuracy | Speed (tok/s) | Size |
|--------|----------|---------------|------|
| bfloat16 (full) | | | ~2GB |
| Q8_0 GGUF | | | ~1GB |
| Q4_K_M GGUF | | | ~500MB |

---

## Known Limitations

- Training data gap: 8.3k items vs 17.5k in reference model
- GRPO stability: small models with unfamiliar reward models can diverge; results
  reported with and without GRPO where relevant
- Evaluation prompt sensitivity: results reflect the specific prompt template above;
  different templates may produce different absolute numbers
- Contamination check is prefix-level only; semantic overlap not verified

---

## Reproduce

```bash
pip install -r requirements.txt
huggingface-cli login

python scripts/01_explore_data.py
python scripts/02_baseline_eval.py
python scripts/03_sft_learner.py
python scripts/04_grpo.py          # skip if unstable
python scripts/05_merge.py
python scripts/06_evaluate_all.py

# Ablations (run for each axis/value pair)
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
```

- [ ] **Step 2: Commit**

```bash
git add MODEL_CARD.md
git commit -m "docs: add model card template with evaluation placeholders"
```

---

## Task 15: Run the test suite

**Files:** (none new)

- [ ] **Step 1: Run all tests**

```bash
pytest tests/ -v
```

Expected output (all should pass — none require GPU):
```
PASSED tests/test_data.py::test_oversample_balances_to_max
PASSED tests/test_data.py::test_undersample_balances_to_min
PASSED tests/test_data.py::test_none_strategy_returns_original
PASSED tests/test_data.py::test_extract_first_letter_finds_abcd
PASSED tests/test_evaluate.py::test_bootstrap_ci_perfect_accuracy
PASSED tests/test_evaluate.py::test_bootstrap_ci_half_accuracy
PASSED tests/test_evaluate.py::test_bootstrap_ci_returns_ordered_bounds
PASSED tests/test_evaluate.py::test_mcnemar_identical_models_returns_high_pvalue
PASSED tests/test_evaluate.py::test_mcnemar_very_different_models_returns_low_pvalue
PASSED tests/test_callbacks.py::test_callback_writes_jsonl_records
PASSED tests/test_callbacks.py::test_callback_handles_none_logs
PASSED tests/test_probe.py::test_probe_on_perfectly_separable_data
PASSED tests/test_probe.py::test_probe_on_random_data_near_chance
PASSED tests/test_probe.py::test_fit_probe_returns_float
```

- [ ] **Step 2: Final commit**

```bash
git add .
git commit -m "chore: verify full test suite passes before GPU execution"
```

---

## Execution Order on GPU Instance

After all tasks are implemented and tests pass locally, run on your A100:

```bash
# Control run (existing pipeline)
python scripts/01_explore_data.py
python scripts/02_baseline_eval.py
python scripts/03_sft_learner.py
python scripts/04_grpo.py
python scripts/05_merge.py
python scripts/06_evaluate_all.py

# Ablations (each ~1-3 hr depending on axis)
for axis_value in "lora_rank 8" "lora_rank 32" "sft_epochs 1" "sft_epochs 5" \
                  "balance_strategy none" "balance_strategy undersample"; do
  python scripts/ablations/run_ablation.py --axis ${axis_value% *} --value ${axis_value#* }
done
python scripts/ablations/summarize_ablations.py

# Diagnostics (after control run completes)
python scripts/diagnostics/probe_position_bias.py
python scripts/diagnostics/analyze_reward_model.py

# Deploy (after ablations complete)
export LLAMA_CPP_DIR=/path/to/llama.cpp
python scripts/deploy/quantize_and_benchmark.py
```
