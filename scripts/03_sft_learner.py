"""
Hours 3-4: SFT with Position-Balanced Data (LEARNER track)
Contribution #1: fixes known MCQ position bias through oversampling.
Saves a full merged checkpoint at ./learner-full for mergekit.
"""

import random
import torch
from collections import Counter
from datasets import load_dataset, Dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

# --- Load data ---
fln = load_dataset("CraneAILabs/luganda-fln-training-data", "all")
exercises = load_dataset("CraneAILabs/luganda-bilingual-literacy-exercises")

# --- Handle exercises dataset: may not have a 'text' column ---
# Run 01_explore_data.py first to confirm the exercises schema.
if 'text' not in exercises['train'].column_names:
    print("exercises dataset has no 'text' column — applying formatter.")
    print(f"Available columns: {exercises['train'].column_names}")

    def format_exercise(row):
        # Adapt this based on what 01_explore_data.py shows.
        # Common pattern: 'question'/'answer' or 'instruction'/'output'.
        cols = exercises['train'].column_names
        if 'question' in cols and 'answer' in cols:
            text = (
                f"<start_of_turn>user\n{row['question']}<end_of_turn>\n"
                f"<start_of_turn>model\n{row['answer']}<end_of_turn>"
            )
        elif 'instruction' in cols and 'output' in cols:
            text = (
                f"<start_of_turn>user\n{row['instruction']}<end_of_turn>\n"
                f"<start_of_turn>model\n{row['output']}<end_of_turn>"
            )
        else:
            # Fallback: join all string-valued columns as a single user message
            content = " | ".join(str(row[c]) for c in cols if isinstance(row[c], str))
            text = f"<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n<end_of_turn>"
        return {"text": text}

    exercises_formatted = exercises['train'].map(format_exercise)
else:
    exercises_formatted = exercises['train']

# --- Contribution #1: Position-balance the MCQ items ---
train_data = fln['train']
mcq_items = [x for x in train_data if x['correct_letter'] in ['A', 'B', 'C', 'D']]
non_mcq_items = [x for x in train_data if x['correct_letter'] not in ['A', 'B', 'C', 'D']]

print(f"MCQ items: {len(mcq_items)}")
print(f"Non-MCQ items: {len(non_mcq_items)}")
print(f"Before balancing: {Counter(x['correct_letter'] for x in mcq_items)}")

by_pos = {}
for item in mcq_items:
    by_pos.setdefault(item['correct_letter'], []).append(item)

max_count = max(len(v) for v in by_pos.values())
balanced = []
for pos, items in by_pos.items():
    if len(items) < max_count:
        balanced.extend(items * (max_count // len(items)))
        balanced.extend(random.sample(items, max_count % len(items)))
    else:
        balanced.extend(items)

random.shuffle(balanced)
balanced_mcq = Dataset.from_list(balanced)
print(f"After balancing: {Counter(x['correct_letter'] for x in balanced_mcq)}")

# Combine all training sources
all_train = concatenate_datasets([
    balanced_mcq,
    Dataset.from_list(non_mcq_items),
    exercises_formatted.select_columns(['text']),
])
print(f"Total training examples: {len(all_train)}")

# --- Load model with QLoRA ---
model = AutoModelForCausalLM.from_pretrained(
    "CraneAILabs/ganda-gemma-1b",
    torch_dtype=torch.bfloat16,
    load_in_4bit=True,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("CraneAILabs/ganda-gemma-1b")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
    # embed_tokens intentionally excluded so the embedding LR stays at zero
    # (Option A from the plan — safe default for 1B models)
)

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
    dataset_text_field="text",
)

trainer.train()

# Merge LoRA back into full weights — required for mergekit
merged_model = trainer.model.merge_and_unload()
merged_model.save_pretrained("./learner-full")
tokenizer.save_pretrained("./learner-full")
print("Saved merged LEARNER model to ./learner-full")
