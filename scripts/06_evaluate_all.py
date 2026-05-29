"""
Hours 8-9: Full Evaluation Sweep
Evaluates all checkpoints on the LLPK benchmark and prints a summary table.
Reuses the evaluate_on_benchmark function from 02_baseline_eval.py logic.
"""

import os
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def evaluate_on_benchmark(model, tokenizer, benchmark_ds, label=""):
    samples = benchmark_ds['train']
    correct = 0
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
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                temperature=0.01,
                repetition_penalty=1.2,
                do_sample=False,
            )

        response = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True,
        ).strip().upper()

        predicted = next((c for c in response if c in "ABCD"), None)
        gold = item['correct_answer']
        position_stats[gold]["total"] += 1
        if predicted == gold:
            correct += 1
            position_stats[gold]["correct"] += 1

        cat = item.get('category', 'unknown')
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "correct": 0}
        category_stats[cat]["total"] += 1
        if predicted == gold:
            category_stats[cat]["correct"] += 1

    total = len(samples)
    accs = [
        s["correct"] / s["total"]
        for s in position_stats.values() if s["total"] > 0
    ]
    spread = (max(accs) - min(accs)) * 100 if accs else 0
    accuracy = correct / total

    print(f"  [{label}] accuracy={accuracy:.1%}  spread={spread:.1f}pp")
    return {"accuracy": accuracy, "position_stats": position_stats,
            "category_stats": category_stats, "spread": spread}


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

# Skip entries whose local paths don't exist (e.g. if GRPO was skipped)
missing = [k for k, p in models_to_eval.items()
           if p.startswith("./") and not os.path.isdir(p)]
if missing:
    print(f"WARNING: skipping missing checkpoints: {missing}")
    models_to_eval = {k: v for k, v in models_to_eval.items() if k not in missing}

all_results = {}
for name, path in models_to_eval.items():
    print(f"\n{'='*60}\nEvaluating: {name}\n{'='*60}")
    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tok = AutoTokenizer.from_pretrained(path)
    all_results[name] = evaluate_on_benchmark(model, tok, benchmark, label=name)
    del model
    torch.cuda.empty_cache()

with open("results/full_results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)

# Summary table
print("\n\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(f"{'Model':<30} {'Acc':>6} {'Spread':>8}  Notes")
print("-" * 80)
for name, r in all_results.items():
    acc = r['accuracy'] * 100
    spread = r['spread']
    notes = ""
    if name == "EduGanda reference":
        notes = "← target"
    elif spread < 20:
        notes = "← bias reduced!"
    print(f"{name:<30} {acc:>5.1f}% {spread:>6.1f}pp  {notes}")

print("\nSaved results/full_results.json")
