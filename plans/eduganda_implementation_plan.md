# EduGanda Replication + Extension Plan (v2)

**Goal:** Reproduce and extend the EduGanda-Gemma-3-1B pipeline using only public resources.
Two concrete contributions: (1) fix the known position bias, (2) ablate merge ratios.

---

## What is already open-sourced

| Asset | HuggingFace ID | Size | Notes |
|-------|---------------|------|-------|
| Base model | `google/gemma-3-1b-it` | 1B | Starting architecture |
| After Luganda CPT | `CraneAILabs/ganda-gemma-1b` | 1B | **Your starting point** — skip CPT |
| FLN training data | `CraneAILabs/luganda-fln-training-data` | 1.37k rows | 5 subsets, pre-formatted |
| Bilingual exercises | `CraneAILabs/luganda-bilingual-literacy-exercises` | 6.94k rows | Exercises across FLN domains |
| Reward model | `CraneAILabs/luganda-reward-model` | 1B | Classifier for GRPO |
| LLPK benchmark | `CraneAILabs/pedagogy-luganda-replaced` | 299 rows | Bilingual MCQ + metadata |
| LLK benchmark | `CraneAILabs/pedagogy-luganda-reviewed` | 299 rows | Reviewed version |
| Reference model | `CraneAILabs/EduGanda-Gemma-3-1B` | 1B | Your target to match/beat |

---

## Real data schemas (not placeholders)

### Training data (`luganda-fln-training-data`)

Already pre-formatted in Gemma chat template. No formatting function needed.

| Column | Type | Example |
|--------|------|---------|
| `text` | string | `<start_of_turn>user\nOmusomesa ayagala...(A)...(B)...(C)...(D)<end_of_turn>\n<start_of_turn>model\nOkuddamu: A ...<end_of_turn>` |
| `format` | string | `mcq`, `mcq_if`, content generation |
| `correct_letter` | string | `A`, `B`, `C`, `D` (or blank for non-MCQ) |
| `category` | string | `Literacy` + 5 others |
| `source` | string | e.g. `fln_synthetic_phonological_awareness` |

**5 subsets:**
- `fln_synthetic` — 590 rows (synthetic MCQs from Gemini 2.5 Flash)
- `fln_weak_sections` — 600 rows (targeted at model weak spots)
- `fln_shortform` — 123 rows (short-form exercises)
- `fln_content` — 55 rows (content generation examples)
- `all` — 1.37k rows (union)

### Benchmark (`pedagogy-luganda-replaced`)

| Column | Type | What it is |
|--------|------|------------|
| `luganda_question` | string | Question text in Luganda |
| `luganda_answer_a` through `_d` | string | Answer options A–D in Luganda |
| `english_question` | string | English translation of question |
| `english_answer_a` through `_d` | string | English answer options |
| `correct_answer` | string | `A`, `B`, `C`, or `D` |
| `category` | string | 8 categories (domains of pedagogy) |
| `subdomain` | string | 5 subdomains |
| `age_group` | string | 6 age groups |

---

## Fixes from v1 of this plan

**Fix 1: The training data is already chat-formatted.**
The `text` column already contains `<start_of_turn>user...<start_of_turn>model...`.
You pass `dataset_text_field="text"` to SFTTrainer. No `format_training_example` needed.

**Fix 2: mergekit requires full models, not LoRA adapters.**
After SFT and GRPO, you must merge the LoRA adapter back into the base model
via `model.merge_and_unload()` and save the full checkpoint. Only then can
mergekit do the linear interpolation.

**Fix 3: Embedding layer learning rate.**
The blog says a much lower learning rate for the vocabulary layer was essential.
Without it, the model learns individual Luganda words but cannot form sentences.
This requires parameter-group-level LR control, not a single global LR.

**Fix 4: Data contamination check.**
Verify that benchmark items from `pedagogy-luganda-replaced` are not present in
the training set. The model card says "verified zero data contamination" — you
should verify this yourself before claiming benchmark numbers.

**Fix 5: The available training data (1.37k + 6.94k = ~8.3k) is less than what
the model card reports (17,561 FLN items).**
Some items were likely generated during the pipeline or not released. Your
results may differ from the reference. This is fine — report it transparently.

---

## Saturday

### Hour 1: Setup + data exploration

```bash
# On your RunPod/Modal A100 instance
pip install torch transformers trl peft datasets accelerate bitsandbytes
pip install mergekit-community huggingface_hub
huggingface-cli login   # needed for gated Gemma model
```

```python
from datasets import load_dataset
import collections

# Explore training data
fln = load_dataset("CraneAILabs/luganda-fln-training-data", "all")
exercises = load_dataset("CraneAILabs/luganda-bilingual-literacy-exercises")

print(f"FLN train: {len(fln['train'])} rows")
print(f"Exercises: {len(exercises['train'])} rows")
print(f"\nFLN columns: {fln['train'].column_names}")
print(f"Exercise columns: {exercises['train'].column_names}")

# Look at actual examples
print("\n--- FLN example ---")
print(fln['train'][0]['text'][:500])
print(f"\nFormat: {fln['train'][0]['format']}")
print(f"Correct letter: {fln['train'][0]['correct_letter']}")

# Check position distribution (THIS IS THE BIAS)
mcq_items = [x for x in fln['train'] if x['correct_letter'] in ['A','B','C','D']]
position_dist = collections.Counter(x['correct_letter'] for x in mcq_items)
print(f"\n--- Position distribution (source of bias) ---")
for pos in ['A','B','C','D']:
    print(f"  {pos}: {position_dist.get(pos, 0)} items ({position_dist.get(pos,0)/len(mcq_items)*100:.1f}%)")

# Explore benchmark
benchmark = load_dataset("CraneAILabs/pedagogy-luganda-replaced")
print(f"\n--- Benchmark ---")
print(f"Columns: {benchmark['train'].column_names}")
print(f"Example question: {benchmark['train'][0]['luganda_question'][:200]}...")
print(f"Correct answer: {benchmark['train'][0]['correct_answer']}")

# DATA CONTAMINATION CHECK
# Extract benchmark question texts and check overlap with training data
benchmark_texts = set(row['luganda_question'][:100] for row in benchmark['train'])
train_texts = set(row['text'][:100] for row in fln['train'])
overlap = benchmark_texts & train_texts
print(f"\n--- Contamination check ---")
print(f"Overlap between benchmark and training: {len(overlap)} items")
```

### Hour 2: Baseline evaluation

Evaluate both the unmodified starting point and the reference model on the benchmark.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def evaluate_on_benchmark(model, tokenizer, benchmark_ds, n_samples=None):
    """
    Evaluate MCQ accuracy on the LLPK benchmark.
    Uses REAL column names from pedagogy-luganda-replaced.
    
    The blog post notes: "Both evaluated in Luganda but keeping the
    instructions in English to make sure all models understand the task equally."
    """
    samples = benchmark_ds['train']
    if n_samples:
        samples = samples.select(range(min(n_samples, len(samples))))
    
    correct = 0
    position_stats = {pos: {"total": 0, "correct": 0} for pos in ["A","B","C","D"]}
    category_stats = {}
    
    for item in samples:
        # Format: English instruction + Luganda question + Luganda options
        # (matches the blog: "instructions in English, evaluated in Luganda")
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
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=0.01,
                repetition_penalty=1.2,   # REQUIRED per model card
                do_sample=False
            )
        
        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        ).strip().upper()
        
        # Extract first letter that's A-D
        predicted = None
        for char in response:
            if char in "ABCD":
                predicted = char
                break
        
        gold = item['correct_answer']
        position_stats[gold]["total"] += 1
        if predicted == gold:
            correct += 1
            position_stats[gold]["correct"] += 1
        
        # Track by category
        cat = item.get('category', 'unknown')
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "correct": 0}
        category_stats[cat]["total"] += 1
        if predicted == gold:
            category_stats[cat]["correct"] += 1
    
    # Report
    total = len(samples)
    print(f"\nOverall: {correct}/{total} = {correct/total:.1%}")
    
    print("\nPer-position accuracy:")
    accs = []
    for pos in ["A","B","C","D"]:
        s = position_stats[pos]
        if s["total"] > 0:
            acc = s["correct"] / s["total"]
            accs.append(acc)
            print(f"  {pos}: {acc:.1%} ({s['correct']}/{s['total']})")
    
    spread = (max(accs) - min(accs)) * 100 if accs else 0
    print(f"  Position bias spread: {spread:.1f} percentage points")
    
    print("\nPer-category accuracy:")
    for cat, s in sorted(category_stats.items()):
        if s["total"] > 0:
            print(f"  {cat}: {s['correct']/s['total']:.1%} ({s['total']} items)")
    
    return {
        "accuracy": correct / total,
        "position_stats": position_stats,
        "category_stats": category_stats,
        "spread": spread
    }

# --- Run baselines ---

# 1. Your starting point (ganda-gemma-1b, no education SFT)
print("=" * 60)
print("BASELINE: ganda-gemma-1b (Luganda CPT only, no education SFT)")
print("=" * 60)
model_base = AutoModelForCausalLM.from_pretrained(
    "CraneAILabs/ganda-gemma-1b",
    torch_dtype=torch.bfloat16, device_map="auto"
)
tok_base = AutoTokenizer.from_pretrained("CraneAILabs/ganda-gemma-1b")
base_results = evaluate_on_benchmark(model_base, tok_base, benchmark)
del model_base; torch.cuda.empty_cache()

# 2. Reference model (EduGanda — your target)
print("\n" + "=" * 60)
print("REFERENCE: EduGanda-Gemma-3-1B (Crane AI Labs shipped model)")
print("=" * 60)
model_ref = AutoModelForCausalLM.from_pretrained(
    "CraneAILabs/EduGanda-Gemma-3-1B",
    torch_dtype=torch.bfloat16, device_map="auto"
)
tok_ref = AutoTokenizer.from_pretrained("CraneAILabs/EduGanda-Gemma-3-1B")
ref_results = evaluate_on_benchmark(model_ref, tok_ref, benchmark)
del model_ref; torch.cuda.empty_cache()

# Save baseline numbers — you will compare against these later
import json
with open("baseline_results.json", "w") as f:
    json.dump({"base": base_results, "reference": ref_results}, f, indent=2, default=str)
```

### Hours 3–4: SFT with position-balanced data (LEARNER track)

```python
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from datasets import load_dataset, Dataset, concatenate_datasets
from collections import Counter
import random, torch

# Load all training data
fln = load_dataset("CraneAILabs/luganda-fln-training-data", "all")
exercises = load_dataset("CraneAILabs/luganda-bilingual-literacy-exercises")

# --- YOUR CONTRIBUTION #1: Position-balance the MCQ items ---
train_data = fln['train']
mcq_items = [x for x in train_data if x['correct_letter'] in ['A','B','C','D']]
non_mcq_items = [x for x in train_data if x['correct_letter'] not in ['A','B','C','D']]

print(f"MCQ items: {len(mcq_items)}")
print(f"Non-MCQ items: {len(non_mcq_items)}")
print(f"Before balancing: {Counter(x['correct_letter'] for x in mcq_items)}")

# Group by position and oversample underrepresented positions
by_pos = {}
for item in mcq_items:
    pos = item['correct_letter']
    by_pos.setdefault(pos, []).append(item)

# Oversample to match the largest position (preserves all data)
max_count = max(len(v) for v in by_pos.values())
balanced = []
for pos, items in by_pos.items():
    if len(items) < max_count:
        # Oversample with repetition
        balanced.extend(items * (max_count // len(items)))
        balanced.extend(random.sample(items, max_count % len(items)))
    else:
        balanced.extend(items)

random.shuffle(balanced)
balanced_mcq = Dataset.from_list(balanced)
print(f"After balancing: {Counter(x['correct_letter'] for x in balanced_mcq)}")

# Combine: balanced MCQs + non-MCQ FLN items + bilingual exercises
all_train = concatenate_datasets([
    balanced_mcq,
    Dataset.from_list(non_mcq_items),
    exercises['train']     # check if this also has a 'text' column — if not, format it
])
print(f"Total training examples: {len(all_train)}")

# Load model with QLoRA
model = AutoModelForCausalLM.from_pretrained(
    "CraneAILabs/ganda-gemma-1b",
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("CraneAILabs/ganda-gemma-1b")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)

# --- EMBEDDING LAYER LR FIX ---
# The blog says: "using a much lower learning rate for the vocabulary layer
# was essential; without this, the model could learn individual words but
# struggled to build sentences."
#
# TRL's SFTConfig doesn't natively support per-layer LR. Two options:
#
# Option A (simple): Exclude embed_tokens from LoRA targets entirely.
#   This means embeddings stay frozen — a safe default for 1B models.
#   In LoraConfig, set: modules_to_save=[] and ensure "embed_tokens"
#   is NOT in target_modules.
#
# Option B (advanced): Use a custom optimizer with param groups.
#   For a weekend project, Option A is recommended.

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir="./learner-sft",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_seq_length=1024,
    ),
    train_dataset=all_train,
    peft_config=peft_config,
    dataset_text_field="text",   # data is already chat-formatted
)

trainer.train()

# CRITICAL: Merge LoRA back to full model for mergekit later
# mergekit needs full-weight checkpoints, not adapters
from peft import PeftModel
merged_model = trainer.model.merge_and_unload()
merged_model.save_pretrained("./learner-full")
tokenizer.save_pretrained("./learner-full")
print("Saved merged LEARNER model to ./learner-full")
```

**Immediately evaluate the LEARNER model on the benchmark** — check both accuracy and position bias spread.

### Hours 5–6: GRPO (RL track)

```python
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForSequenceClassification

# --- First: understand the reward model output ---
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "CraneAILabs/luganda-reward-model",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
reward_tokenizer = AutoTokenizer.from_pretrained("CraneAILabs/luganda-reward-model")

# Probe the model to understand its output
print(f"Num labels: {reward_model.config.num_labels}")
print(f"Id2label: {getattr(reward_model.config, 'id2label', 'not set')}")

# Test with a sample input
test_input = reward_tokenizer("Test input in Luganda", return_tensors="pt")
test_input = {k: v.to(reward_model.device) for k, v in test_input.items()}
with torch.no_grad():
    test_output = reward_model(**test_input)
print(f"Output shape: {test_output.logits.shape}")
print(f"Output values: {test_output.logits}")
# This tells you whether it's binary (2 logits) or regression (1 logit)

# --- Define reward function based on what you discover above ---
def compute_rewards(completions, **kwargs):
    """
    Adapt this based on the reward model output format.
    If binary classifier: use softmax and take positive class prob.
    If regression: use raw logit.
    """
    rewards = []
    for completion in completions:
        # Each completion is a list of message dicts or a string
        text = completion if isinstance(completion, str) else completion[-1]["content"]
        inputs = reward_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(reward_model.device)
        
        with torch.no_grad():
            logits = reward_model(**inputs).logits
        
        if logits.shape[-1] == 2:
            # Binary classifier: P(positive class)
            score = torch.softmax(logits, dim=-1)[0, 1].item()
        else:
            # Regression or single logit
            score = logits[0, 0].item()
        
        rewards.append(score)
    return rewards

# --- Prepare GRPO prompts ---
# Use a subset of FLN data as prompts (not the full formatted text,
# just the user turns as prompts for generation)
grpo_prompts = []
for item in fln['train'].select(range(min(300, len(fln['train'])))):
    # Extract just the user prompt from the pre-formatted text
    text = item['text']
    if '<start_of_turn>user' in text and '<end_of_turn>' in text:
        user_msg = text.split('<start_of_turn>user\n')[1].split('<end_of_turn>')[0]
        grpo_prompts.append({"prompt": user_msg})

grpo_dataset = Dataset.from_list(grpo_prompts)

# Load LEARNER checkpoint as policy
policy_model = AutoModelForCausalLM.from_pretrained(
    "./learner-full",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

grpo_trainer = GRPOTrainer(
    model=policy_model,
    reward_funcs=compute_rewards,
    args=GRPOConfig(
        output_dir="./grpo-output",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,          # much lower than SFT — RL is sensitive
        bf16=True,
        num_generations=4,           # 4 completions per prompt for group ranking
        max_new_tokens=200,
        max_prompt_length=512,
        logging_steps=10,
        save_steps=200,
        max_steps=600,               # match GRPO-600 from the paper name
    ),
    train_dataset=grpo_dataset,
    processing_class=tokenizer,
)

grpo_trainer.train()

# Save full model for mergekit
grpo_trainer.model.save_pretrained("./grpo-full")
tokenizer.save_pretrained("./grpo-full")
```

**If GRPO fails or is unstable** (common with small models and unfamiliar reward models),
skip to the merge step using just the LEARNER checkpoint at different LoRA ranks instead.
A LEARNER-only model with fixed position bias is already a valid contribution.

---

## Sunday

### Hour 7: Merge ablation

mergekit operates on full model directories (not LoRA adapters — that's why
we ran `merge_and_unload()` above).

```bash
pip install mergekit-community
```

Create three merge configs:

```python
import yaml, subprocess, os

configs = {
    "80-20": {"learner": 0.8, "grpo": 0.2},
    "70-30": {"learner": 0.7, "grpo": 0.3},   # original paper ratio
    "60-40": {"learner": 0.6, "grpo": 0.4},
}

for name, weights in configs.items():
    config = {
        "models": [
            {"model": os.path.abspath("./learner-full"),
             "parameters": {"weight": weights["learner"]}},
            {"model": os.path.abspath("./grpo-full"),
             "parameters": {"weight": weights["grpo"]}},
        ],
        "merge_method": "linear",
        "dtype": "bfloat16",
    }
    
    config_path = f"merge_{name}.yaml"
    output_path = f"./merged-{name}"
    
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    subprocess.run([
        "mergekit-yaml", config_path, output_path, "--cuda"
    ], check=True)
    
    print(f"Merged {name} → {output_path}")
```

Also try DARE-TIES (what the paper actually calls "DARE/TIES of experts"):

```python
dare_config = {
    "models": [
        {"model": os.path.abspath("./learner-full"),
         "parameters": {"weight": 0.7, "density": 0.5}},
        {"model": os.path.abspath("./grpo-full"),
         "parameters": {"weight": 0.3, "density": 0.5}},
    ],
    "merge_method": "dare_ties",
    "base_model": os.path.abspath("./learner-full"),
    "dtype": "bfloat16",
}

with open("merge_dare_ties.yaml", "w") as f:
    yaml.dump(dare_config, f)

subprocess.run([
    "mergekit-yaml", "merge_dare_ties.yaml", "./merged-dare-ties", "--cuda"
], check=True)
```

### Hours 8–9: Full evaluation sweep

```python
# Evaluate ALL checkpoints
models_to_eval = {
    "ganda-gemma-1b (base)": "CraneAILabs/ganda-gemma-1b",
    "LEARNER (SFT, balanced)": "./learner-full",
    "Merged 80/20": "./merged-80-20",
    "Merged 70/30": "./merged-70-30",
    "Merged 60/40": "./merged-60-40",
    "DARE-TIES 70/30": "./merged-dare-ties",
    "EduGanda reference": "CraneAILabs/EduGanda-Gemma-3-1B",
}

all_results = {}
for name, path in models_to_eval.items():
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"{'='*60}")
    
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tok = AutoTokenizer.from_pretrained(path)
    
    results = evaluate_on_benchmark(model, tok, benchmark)
    all_results[name] = results
    
    del model; torch.cuda.empty_cache()

# Save all results
with open("full_results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)

# Print summary table
print("\n\n" + "="*80)
print("SUMMARY TABLE (for README)")
print("="*80)
print(f"{'Model':<30} {'PCK':>6} {'Spread':>8} {'Notes'}")
print("-"*80)
for name, r in all_results.items():
    acc = r['accuracy']*100
    spread = r['spread']
    notes = ""
    if name == "EduGanda reference":
        notes = "← target"
    elif spread < 20:
        notes = "← bias reduced!"
    print(f"{name:<30} {acc:>5.1f}% {spread:>6.1f}pp {notes}")
```

### Hour 10: Write up + push

**README structure:**

```markdown
# EduGanda Extension: Position Bias Fix + Merge Ablation

## TL;DR
Reproduced the EduGanda-Gemma-3-1B pipeline using open-source data.
Fixed the known MCQ position bias (52pp spread → Xpp) through balanced
dataset curation. Ablated LEARNER/GRPO merge ratios to characterise the
accuracy/fluency trade-off.

## Background
[Link to Fab AI blog post. 2-sentence summary of what they did.]

## What I changed
1. **Position-balanced training data:** Original model has B:93%, D:41%
   accuracy (52pp spread). Oversampled underrepresented positions.
2. **Merge ratio ablation:** Tested 80/20, 70/30, 60/40, + DARE-TIES.

## Results
[Paste your summary table here]

## Key findings
[What did the merge ratio ablation show?
Did more GRPO help or hurt pedagogy accuracy?
Did position balancing reduce the spread without hurting overall accuracy?]

## Reproduce
[pip install, dataset IDs, training commands]

## Limitations
[What didn't work. Be honest — this is more credible than only showing wins.]
```

**GitHub repo structure:**

```
eduganda-extension/
├── README.md
├── requirements.txt
├── scripts/
│   ├── 01_explore_data.py
│   ├── 02_baseline_eval.py
│   ├── 03_sft_learner.py
│   ├── 04_grpo.py
│   ├── 05_merge.py
│   └── 06_evaluate_all.py
├── configs/
│   ├── merge_80_20.yaml
│   ├── merge_70_30.yaml
│   ├── merge_60_40.yaml
│   └── merge_dare_ties.yaml
├── results/
│   ├── baseline_results.json
│   └── full_results.json
└── blog/                     # optional TurboQuant post
    └── eduganda_extension.md
```

---

## What can go wrong (and what to do)

| Problem | Symptom | Fix |
|---------|---------|-----|
| Bilingual exercises dataset has different columns | `KeyError: 'text'` during training | Inspect with `print(exercises['train'].column_names)`, write a formatting function for that dataset only |
| GRPO training diverges | Loss spikes, reward collapses to constant | Lower LR to 1e-6, reduce `num_generations` to 2, or skip GRPO entirely — LEARNER-only with bias fix is still a valid contribution |
| Reward model outputs unexpected format | Reward values are all the same | Probe with test inputs first (see GRPO section). If the classifier doesn't work well, use a simple rule-based reward (e.g. penalise repetition, reward Luganda token ratio) |
| mergekit fails on LoRA checkpoints | Error about mismatched keys | You forgot `merge_and_unload()`. Go back and save full-weight models |
| Position bias fix improves spread but drops overall accuracy | e.g. spread goes from 52pp to 15pp but accuracy drops from 66% to 58% | This is a real result — report it as a trade-off finding. The original model was "cheating" on B-heavy questions |
| Results don't match published numbers | Your base model gets 45% not 51% | Different eval prompt template, different sampling. Report your evaluation setup clearly and compare all models under YOUR setup consistently |

---

## CV bullet

> Extended the EduGanda-Gemma-3-1B pipeline (Fab AI / Crane AI Labs, 2026) for
> on-device foundational literacy in Uganda: reproduced SFT → GRPO → model merge
> pipeline; fixed known MCQ position bias through balanced dataset curation
> (spread: 52pp → [X]pp); ablated merge ratios (linear + DARE-TIES) to characterise
> the curriculum accuracy vs. linguistic fluency trade-off. Open-sourced at [link].

**Skills demonstrated:** SFT, GRPO/RL, reward model usage, model merging (mergekit),
QLoRA, low-resource language ML, evaluation design, data-centric ML.

---

## Compute budget

| Stage | A100 time | Cost at $1.99/hr |
|-------|-----------|-------------------|
| Setup + exploration | 0.5 hr | $1.00 |
| Baseline evaluation | 0.5 hr | $1.00 |
| SFT LEARNER | 1 hr | $1.99 |
| GRPO-600 | 2–3 hr | $5.97 |
| Merge (4 variants) | 0.25 hr | $0.50 |
| Full eval sweep (7 models) | 1.5 hr | $2.99 |
| **Total** | **~6–7 hr** | **~$13** |
