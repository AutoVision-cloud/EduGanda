"""
Hour 2: Baseline Evaluation
Evaluates ganda-gemma-1b (CPT only) and EduGanda-Gemma-3-1B (reference) on the LLPK benchmark.
Run BEFORE any training to establish your comparison baseline.
"""

import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def evaluate_on_benchmark(model, tokenizer, benchmark_ds, n_samples=None, label=""):
    """
    Evaluate MCQ accuracy on the LLPK benchmark (pedagogy-luganda-replaced).
    Uses English instructions + Luganda questions, matching the blog's evaluation setup.
    """
    samples = benchmark_ds['train']
    if n_samples:
        samples = samples.select(range(min(n_samples, len(samples))))

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

        cat = item.get('category', 'unknown')
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "correct": 0}
        category_stats[cat]["total"] += 1
        if predicted == gold:
            category_stats[cat]["correct"] += 1

    total = len(samples)
    print(f"\n[{label}] Overall: {correct}/{total} = {correct/total:.1%}")

    print("  Per-position accuracy:")
    accs = []
    for pos in ["A", "B", "C", "D"]:
        s = position_stats[pos]
        if s["total"] > 0:
            acc = s["correct"] / s["total"]
            accs.append(acc)
            print(f"    {pos}: {acc:.1%} ({s['correct']}/{s['total']})")

    spread = (max(accs) - min(accs)) * 100 if accs else 0
    print(f"  Position bias spread: {spread:.1f} pp")

    print("  Per-category accuracy:")
    for cat, s in sorted(category_stats.items()):
        if s["total"] > 0:
            print(f"    {cat}: {s['correct']/s['total']:.1%} ({s['total']} items)")

    return {
        "accuracy": correct / total,
        "position_stats": position_stats,
        "category_stats": category_stats,
        "spread": spread,
    }


benchmark = load_dataset("CraneAILabs/pedagogy-luganda-replaced")

# 1. Starting point: ganda-gemma-1b (Luganda CPT, no education SFT)
print("=" * 60)
print("BASELINE: ganda-gemma-1b")
print("=" * 60)
model_base = AutoModelForCausalLM.from_pretrained(
    "CraneAILabs/ganda-gemma-1b",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tok_base = AutoTokenizer.from_pretrained("CraneAILabs/ganda-gemma-1b")
base_results = evaluate_on_benchmark(model_base, tok_base, benchmark, label="ganda-gemma-1b")
del model_base
torch.cuda.empty_cache()

# 2. Reference: EduGanda (target to match/beat)
print("\n" + "=" * 60)
print("REFERENCE: EduGanda-Gemma-3-1B")
print("=" * 60)
model_ref = AutoModelForCausalLM.from_pretrained(
    "CraneAILabs/EduGanda-Gemma-3-1B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tok_ref = AutoTokenizer.from_pretrained("CraneAILabs/EduGanda-Gemma-3-1B")
ref_results = evaluate_on_benchmark(model_ref, tok_ref, benchmark, label="EduGanda-Gemma-3-1B")
del model_ref
torch.cuda.empty_cache()

with open("results/baseline_results.json", "w") as f:
    json.dump({"base": base_results, "reference": ref_results}, f, indent=2, default=str)

print("\nSaved results/baseline_results.json")
