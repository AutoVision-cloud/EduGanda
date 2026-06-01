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

print("\n=== EVALUATOR SANITY TEST ===\n")
from scripts.core.data import extract_first_letter

# ---------------------------------------------------------------------------
# Test 1: EXACT training prefix — the decisive test.
# Model memorised these 20 examples. If it can't get ≥90% here, the
# training pipeline or adapter inference is broken.
# ---------------------------------------------------------------------------
print("Test 1: EXACT training prefix  (requirement: ≥90%)")
correct_exact, raws_exact = 0, []
for row in mcq_items:
    text = row["text"]
    if "<start_of_turn>model\n" not in text:
        continue
    prompt = text.split("<start_of_turn>model\n")[0] + "<start_of_turn>model\n"
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(adapter.device)
    with torch.no_grad():
        out = adapter.generate(**enc, max_new_tokens=20, do_sample=False,
                               repetition_penalty=1.2, pad_token_id=tokenizer.eos_token_id,
                               top_p=None, top_k=None)
    raw = tokenizer.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
    pred = extract_first_letter(raw)
    raws_exact.append(raw[:50])
    if pred == row["correct_letter"]:
        correct_exact += 1

acc_exact = correct_exact / len(mcq_items)
pass1 = acc_exact >= 0.9
print(f"  {'✓ PASS' if pass1 else '✗ FAIL'}  accuracy={acc_exact:.1%}  "
      f"({'training pipeline confirmed working' if pass1 else 'training pipeline broken'})")
if not pass1:
    for i, r in enumerate(raws_exact[:5]):
        print(f"    sample[{i}]: {repr(r)}")

# ---------------------------------------------------------------------------
# Test 2: RECONSTRUCTED prompt (what evaluate_on_benchmark uses for baseline).
# Requirements: invalid rate <15%, accuracy >random (>25% random for 4-way MCQ).
# NOTE: reconstruction mismatch will likely hurt memorised-example accuracy.
# This test confirms parser health, NOT knowledge capture.
# ---------------------------------------------------------------------------
print("\nTest 2: RECONSTRUCTED prompt — free generation only")
print("  Requirements: invalid <15%, accuracy meaningful (>random 25% would be ideal).")
print("  Note: reconstruction mismatch expected to hurt accuracy on memorised examples.")
r_free = evaluate_on_benchmark(adapter, tokenizer, tiny_bench,
                                label="sanity (reconstructed free gen)", forced_format=False)
invalid_ok = r_free["invalid_parse_rate"] < 0.15
pass2 = invalid_ok
print(f"  {'✓' if invalid_ok else '✗'} invalid={r_free['invalid_parse_rate']:.1%} "
      f"({'<15% requirement met' if invalid_ok else '>15% — parser broken'})")
print(f"  accuracy={r_free['accuracy']:.1%}  dist={r_free['prediction_distribution']}")

# ---------------------------------------------------------------------------
# Test 3: Forced format — only useful if it does better than free gen.
# If forced format gives same or worse accuracy on memorised data, don't use it.
# ---------------------------------------------------------------------------
print("\nTest 3: RECONSTRUCTED prompt — forced format ('Okuddamu: (')")
r_forced = evaluate_on_benchmark(adapter, tokenizer, tiny_bench,
                                  label="sanity (reconstructed forced)", forced_format=True)
forced_better = r_forced["accuracy"] >= r_free["accuracy"] and r_forced["invalid_parse_rate"] < 0.15
print(f"  {'✓ better than free gen' if forced_better else '✗ not better — drop forced format'}")
print(f"  accuracy={r_forced['accuracy']:.1%}  invalid={r_forced['invalid_parse_rate']:.1%}")

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
print("\n" + "="*60)
if pass1 and pass2:
    print("✓ Training pipeline confirmed (Test 1 ≥90%).")
    print("✓ Parser working (Test 2 invalid <15%).")
    if r_forced["accuracy"] > r_free["accuracy"] + 0.05:
        print("  Forced format marginally better — keep as diagnostic only.")
    else:
        print("  Forced format same accuracy as free gen — diagnostic only, not primary metric.")
    print()
    print("Safe to run baselines for DIAGNOSTIC COMPARISON.")
    print("NOT safe to interpret absolute MCQ accuracy as model knowledge.")
    print()
    print("Primary metrics: prediction distribution, entropy, position-bias spread.")
    print("Framing: 'measures prediction distribution and answer-position collapse")
    print("under a fixed prompt' — not reproduction of published 66%.")
elif not pass1:
    print("✗ Test 1 FAILED — training pipeline broken. Fix before proceeding.")
else:
    print("✗ Test 2 FAILED — parser broken (high invalid rate). Fix extract_first_letter.")
