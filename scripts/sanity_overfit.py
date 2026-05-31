"""
Sanity check 2: Can the model overfit 20 training examples?
If it can't reach high train accuracy after 20-50 epochs on 20 items,
the training pipeline (LoRA targets, loss mask, tokenizer) is broken.

Usage: python scripts/sanity_overfit.py
Expected: train accuracy should climb toward 80-100% by epoch 10-20.
"""
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from scripts.core.evaluate import evaluate_on_benchmark, _get_choice_token_ids, ANSWER_TOKENS

N_EXAMPLES = 20
N_EPOCHS = 30

print("Loading 20 MCQ examples from FLN training data...")
fln = load_dataset("CraneAILabs/luganda-fln-training-data", "all")["train"]
mcq_items = [x for x in fln if x["correct_letter"] in ANSWER_TOKENS][:N_EXAMPLES]
tiny_ds = Dataset.from_list(mcq_items)
print(f"  {len(tiny_ds)} items, columns: {tiny_ds.column_names}\n")

# Also build a tiny eval set matching benchmark format — evaluate on same 20 items
# by mapping them to benchmark-like structure
def mcq_to_bench_item(row):
    """Convert FLN training item to benchmark-style dict for evaluate_on_benchmark."""
    text = row["text"]
    # Extract question and options from pre-formatted chat text
    if "<start_of_turn>user\n" not in text:
        return None
    user_part = text.split("<start_of_turn>user\n")[1].split("<end_of_turn>")[0].strip()
    lines = [l.strip() for l in user_part.split("\n") if l.strip()]
    # Options are lines starting with (A), (B), (C), (D)
    opts, question_lines = {}, []
    for line in lines:
        if line.startswith("(A)"):
            opts["a"] = line[3:].strip()
        elif line.startswith("(B)"):
            opts["b"] = line[3:].strip()
        elif line.startswith("(C)"):
            opts["c"] = line[3:].strip()
        elif line.startswith("(D)"):
            opts["d"] = line[3:].strip()
        else:
            question_lines.append(line)
    return {
        "luganda_question": " ".join(question_lines),
        "luganda_answer_a": opts.get("a", ""),
        "luganda_answer_b": opts.get("b", ""),
        "luganda_answer_c": opts.get("c", ""),
        "luganda_answer_d": opts.get("d", ""),
        "correct_answer": row["correct_letter"],
        "category": row.get("category", "unknown"),
        "subdomain": "", "age_group": "",
    }

tiny_bench_items = [mcq_to_bench_item(r) for r in mcq_items]
tiny_bench_items = [x for x in tiny_bench_items if x is not None]
tiny_bench = {"train": Dataset.from_list(tiny_bench_items)}
print(f"Tiny eval set: {len(tiny_bench['train'])} items\n")

print("Loading ganda-gemma-1b with QLoRA...")
model = AutoModelForCausalLM.from_pretrained(
    "CraneAILabs/ganda-gemma-1b",
    torch_dtype=torch.bfloat16, load_in_4bit=True, device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("CraneAILabs/ganda-gemma-1b")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    r=16, lora_alpha=32, target_modules="all-linear",
    lora_dropout=0.05, task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir="./sanity-overfit",
        num_train_epochs=N_EPOCHS,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        max_seq_length=512,
        report_to="none",
    ),
    train_dataset=tiny_ds,
    peft_config=peft_config,
    dataset_text_field="text",
)

print(f"Training on {N_EXAMPLES} examples for {N_EPOCHS} epochs...\n")
trainer.train()

print("\nMerging LoRA and evaluating on training set...")
merged = trainer.model.merge_and_unload()
merged.eval()

result = evaluate_on_benchmark(merged, tokenizer, tiny_bench, label="overfit-check")
print(f"\n=== OVERFIT SANITY RESULT ===")
print(f"Train accuracy: {result['accuracy']:.1%} ({int(result['accuracy']*len(tiny_bench_items))}/{len(tiny_bench_items)} correct)")
print(f"Prediction dist: {result['prediction_distribution']}")
print(f"Spread: {result['spread']:.1f}pp")
print()
if result["accuracy"] >= 0.7:
    print("✓ PASS: model can overfit training data — pipeline is working.")
elif result["accuracy"] >= 0.4:
    print("⚠ MARGINAL: partial overfit — check loss curve and LoRA rank.")
else:
    print("✗ FAIL: model cannot overfit 20 examples — check loss mask, LoRA targets, tokenizer.")
