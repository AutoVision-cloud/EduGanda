"""
Sanity check: evaluator must get ~100% on the 20 training examples
the model already memorised (from sanity_overfit.py).

If this fails, the evaluator is wrong. Fix evaluator before running baseline.

Usage: python scripts/sanity_eval.py
Expected: ≥80% on forced format, ≥60% on free gen (model knows the answers).
"""
import sys
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
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

# Convert to benchmark format
def to_bench(row):
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
    return {
        "luganda_question": " ".join(q_lines),
        "luganda_answer_a": opts.get("a", ""), "luganda_answer_b": opts.get("b", ""),
        "luganda_answer_c": opts.get("c", ""), "luganda_answer_d": opts.get("d", ""),
        "correct_answer": row["correct_letter"],
        "category": "train", "subdomain": "", "age_group": "",
    }

tiny_bench_items = [x for x in (to_bench(r) for r in mcq_items) if x]
tiny_bench = {"train": Dataset.from_list(tiny_bench_items)}
print(f"  {len(tiny_bench_items)} benchmark-format items\n")

# Quick-train on 20 examples (re-use same overfit setup)
print("Training on 20 examples for sanity overfit...")
model = AutoModelForCausalLM.from_pretrained(
    "CraneAILabs/ganda-gemma-1b",
    torch_dtype=torch.bfloat16, load_in_4bit=True, device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("CraneAILabs/ganda-gemma-1b")
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_config = LoraConfig(r=32, lora_alpha=64, target_modules="all-linear",
                          lora_dropout=0.05, task_type="CAUSAL_LM")
collator = DataCollatorForCompletionOnlyLM("<start_of_turn>model\n", tokenizer=tokenizer)
trainer = SFTTrainer(
    model=model,
    args=SFTConfig(output_dir="./sanity-eval-tmp", num_train_epochs=N_EPOCHS,
                   per_device_train_batch_size=4, learning_rate=5e-4, bf16=True,
                   logging_steps=99, save_strategy="no", max_seq_length=512,
                   report_to="none", dataset_text_field="text"),
    train_dataset=tiny_ds, peft_config=peft_config, data_collator=collator,
)
trainer.train()
adapter_model = trainer.model
adapter_model.eval()

print("\n=== EVALUATOR SANITY TEST ===")
print("(model memorised these 20 examples — evaluator must get ~100%)\n")

for mode, forced in [("free generation", False), ("forced format", True)]:
    r = evaluate_on_benchmark(adapter_model, tokenizer, tiny_bench,
                               label=f"sanity ({mode})", forced_format=forced)
    print(f"{mode}: {r['accuracy']:.1%}  invalid={r['invalid_parse_rate']:.1%}  "
          f"dist={r['prediction_distribution']}")
    if r["accuracy"] < 0.6:
        print(f"  ✗ FAIL — evaluator broken for {mode}")
    else:
        print(f"  ✓ PASS")
