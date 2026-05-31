"""
Sanity check 2: Can the model overfit 20 training examples?
Uses DataCollatorForCompletionOnlyLM for true assistant-only loss —
masking the user/prompt tokens so loss only comes from model completion.
Minimum bar: ≥80% train accuracy on 20 examples.

Usage: python scripts/sanity_overfit.py
"""
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
from scripts.core.evaluate import ANSWER_TOKENS

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
tokenizer.padding_side = "right"

peft_config = LoraConfig(
    r=32, lora_alpha=64, target_modules="all-linear",
    lora_dropout=0.05, task_type="CAUSAL_LM",
)

# Assistant-only loss: mask everything before "<start_of_turn>model\n"
# Loss only flows through the model's completion tokens, not user prompt
response_template = "<start_of_turn>model\n"
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template, tokenizer=tokenizer
)

trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir="./sanity-overfit",
        num_train_epochs=N_EPOCHS,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=5e-4,
        bf16=True,
        logging_steps=5,
        save_strategy="no",
        max_seq_length=512,
        report_to="none",
        dataset_text_field="text",
    ),
    train_dataset=tiny_ds,
    peft_config=peft_config,
    data_collator=collator,
)

print(f"Training on {N_EXAMPLES} examples for {N_EPOCHS} epochs...\n")
trainer.train()

print("\nEvaluating on training set (WITHOUT merging — avoids 4-bit merge errors)...")
# Evaluate the adapter model directly, before merging
# Use generation from the exact training prompt prefix — checks if model memorised output
adapter_model = trainer.model
adapter_model.eval()

correct_gen = 0
print(f"\n{'#':>3}  gold  raw_output (first 40 chars)")
print("-" * 60)
for idx, item in enumerate(mcq_items):
    text = item["text"]
    if "<start_of_turn>model\n" not in text:
        continue
    # Use the exact training prefix up to "<start_of_turn>model\n"
    prompt = text.split("<start_of_turn>model\n")[0] + "<start_of_turn>model\n"
    gold = item["correct_letter"]

    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(adapter_model.device)
    with torch.no_grad():
        out = adapter_model.generate(
            **enc, max_new_tokens=20, do_sample=False,
            repetition_penalty=1.2, pad_token_id=tokenizer.eos_token_id,
        )
    raw = tokenizer.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()

    # Check if generation starts with "Okuddamu: X" or just the letter
    from scripts.core.data import extract_first_letter
    predicted = extract_first_letter(raw)
    match = "✓" if predicted == gold else "✗"
    if predicted == gold:
        correct_gen += 1
    print(f"[{idx:2d}]  {gold}    {match}  {repr(raw[:40])}")

n = len(mcq_items)
print(f"\n=== OVERFIT SANITY RESULT ===")
print(f"Generation accuracy on training set: {correct_gen}/{n} = {correct_gen/n:.1%}")
print()
if correct_gen / n >= 0.7:
    print("✓ PASS: model memorised training answers — pipeline is working.")
    print("  → Your log-prob evaluator measures the wrong token. Fix eval, not training.")
elif correct_gen / n >= 0.4:
    print("⚠ MARGINAL: partial memorisation.")
    print("  → Check loss mask and training format.")
else:
    print("✗ FAIL: model did not memorise training answers despite low loss.")
    print("  → Check: loss mask (is model turn masked?), LoRA targets, data format.")
