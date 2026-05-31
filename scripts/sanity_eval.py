"""
Sanity check: evaluate_on_benchmark must score ≥80% (forced) / ≥60% (free)
on the 20 training examples the model already memorised.

If forced format fails ≥80%, do NOT trust baseline evaluation yet.

Usage: python scripts/sanity_eval.py
"""
import sys
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from peft import LoraConfig
sys.path.insert(0, '/workspace/EduGanda')
from scripts.core.evaluate import evaluate_on_benchmark, ANSWER_TOKENS

N_EXAMPLES = 20
N_EPOCHS   = 30

print("Loading 20 FLN MCQ examples...")
fln = load_dataset("CraneAILabs/luganda-fln-training-data", "all")["train"]
mcq_items = [x for x in fln if x["correct_letter"] in ANSWER_TOKENS][:N_EXAMPLES]
tiny_ds = Dataset.from_list(mcq_items)

# Benchmark format A: reconstructed from parsed text (matches evaluate_on_benchmark prompt)
def to_bench_reconstructed(row):
    text = row["text"]
    if "<start_of_turn>user\n" not in text:
        return None
    user_part = text.split("<start_of_turn>user\n")[1].split("<end_of_turn>")[0].strip()
    lines = [l.strip() for l in user_part.split("\n") if l.strip()]
    opts, q_lines = {}, []
    for line in lines:
        if line.startswith("(A)"): opts["a"] = line[3:].strip()
        elif line.startswith("(B)"): opts["b"] = line[3:].strip()
        elif line.startswith("(C)"): opts["c"] = line[3:].strip()
        elif line.startswith("(D)"): opts["d"] = line[3:].strip()
        else: q_lines.append(line)
    if len(opts) < 4:
        return None
    return {
        "luganda_question": " ".join(q_lines),
        "luganda_answer_a": opts["a"], "luganda_answer_b": opts["b"],
        "luganda_answer_c": opts["c"], "luganda_answer_d": opts["d"],
        "correct_answer": row["correct_letter"],
        "category": "train", "subdomain": "", "age_group": "",
    }

tiny_bench = {"train": Dataset.from_list(
    [x for x in (to_bench_reconstructed(r) for r in mcq_items) if x]
)}
print(f"  {len(tiny_bench['train'])} reconstructed benchmark items\n")

# Train
print(f"Training on {N_EXAMPLES} examples for {N_EPOCHS} epochs (sanity overfit)...")
model = AutoModelForCausalLM.from_pretrained(
    "CraneAILabs/ganda-gemma-1b",
    torch_dtype=torch.bfloat16, load_in_4bit=True, device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("CraneAILabs/ganda-gemma-1b")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_config = LoraConfig(r=32, lora_alpha=64, target_modules="all-linear",
                          lora_dropout=0.05, task_type="CAUSAL_LM")
collator = DataCollatorForCompletionOnlyLM("<start_of_turn>model\n", tokenizer=tokenizer)
trainer = SFTTrainer(
    model=model,
    args=SFTConfig(
        output_dir="./sanity-eval-tmp", num_train_epochs=N_EPOCHS,
        per_device_train_batch_size=4, learning_rate=5e-4, bf16=True,
        logging_steps=99, save_strategy="no", max_seq_length=512,
        report_to="none", dataset_text_field="text",
    ),
    train_dataset=tiny_ds, peft_config=peft_config, data_collator=collator,
)
trainer.train()
adapter = trainer.model
adapter.eval()

print("\n=== EVALUATOR SANITY TEST ===")
print("Model memorised 20 examples. Evaluator must score ≥80% forced / ≥60% free.\n")

thresholds = {"free generation": 0.6, "forced format": 0.8}
all_pass = True

for mode, forced in [("free generation", False), ("forced format", True)]:
    r = evaluate_on_benchmark(
        adapter, tokenizer, tiny_bench,
        label=f"sanity ({mode})", forced_format=forced
    )
    threshold = thresholds[mode]
    passed = r["accuracy"] >= threshold
    all_pass = all_pass and passed
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"{status}  {mode}: acc={r['accuracy']:.1%}  invalid={r['invalid_parse_rate']:.1%}  "
          f"spread={r['spread']:.1f}pp  dist={r['prediction_distribution']}")

print()
if all_pass:
    print("✓ Evaluator confirmed working. Safe to run 02_baseline_eval.py.")
else:
    print("✗ Evaluator not passing. DO NOT run baseline yet. Debug further.")
    print("  Check: are outputs like 'Okuddamu: Phonoeme...' being parsed as invalid?")
    print("  Run evaluate_on_benchmark with label= and inspect sample outputs.")
