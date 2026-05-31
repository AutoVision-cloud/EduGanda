"""
Hour 2: Baseline Evaluation
Evaluates ganda-gemma-1b (CPT only) and EduGanda-Gemma-3-1B (reference) on the LLPK benchmark.
Run BEFORE any training to establish your comparison baseline.

All models evaluated under the same English-instruction, Luganda-question MCQ prompt.
Primary metric: log-probability option scoring — scores prompt+"A/B/C/D" via a full
forward pass and picks the highest log-probability. Avoids generation quirks and
parser sensitivity. Tokenization is verified before evaluation begins.
"""

import json
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.core.evaluate import evaluate_on_benchmark, evaluate_on_benchmark_generation, bootstrap_ci, check_tokenization

os.makedirs("results", exist_ok=True)

# --- Tokenization verification ---
# Confirm A/B/C/D each tokenize to exactly 1 token under the model tokenizer.
# This is a prerequisite for fair log-prob option scoring: if any answer label
# tokenizes differently (e.g. " A" vs "A"), score_choice still handles it
# correctly, but we log the evidence so the claim is documentable.
print("=" * 60)
print("TOKENIZATION VERIFICATION")
print("=" * 60)
_tok_check = AutoTokenizer.from_pretrained("CraneAILabs/ganda-gemma-1b")
check_tokenization(_tok_check)
print()
for letter in ["A", "B", "C", "D"]:
    n = len(_tok_check.encode(letter, add_special_tokens=False))
    status = "✓" if n == 1 else f"WARNING: {n} tokens"
    print(f"  '{letter}' → {n} token(s)  {status}")
del _tok_check
print()

benchmark = load_dataset("CraneAILabs/pedagogy-luganda-replaced")

# 1. Starting point: ganda-gemma-1b (Luganda CPT only, no education SFT)
print("=" * 60)
print("BASELINE: ganda-gemma-1b")
print("=" * 60)
model_base = AutoModelForCausalLM.from_pretrained(
    "CraneAILabs/ganda-gemma-1b",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)
model_base.eval()
tok_base = AutoTokenizer.from_pretrained("CraneAILabs/ganda-gemma-1b")
if tok_base.pad_token is None:
    tok_base.pad_token = tok_base.eos_token
base_results = evaluate_on_benchmark(model_base, tok_base, benchmark, label="ganda-gemma-1b (log-prob)")
base_results["ci_lower"], base_results["ci_upper"] = bootstrap_ci(
    base_results["predictions"], base_results["labels"]
)[1:]
# Also run generation-based scoring to match published methodology
base_gen = evaluate_on_benchmark_generation(model_base, tok_base, benchmark, label="ganda-gemma-1b (generation)")
base_results["generation_accuracy"] = base_gen["accuracy"]
del model_base
torch.cuda.empty_cache()

# 2. Reference: EduGanda-Gemma-3-1B (target to match/beat)
print("\n" + "=" * 60)
print("REFERENCE: EduGanda-Gemma-3-1B")
print("=" * 60)
model_ref = AutoModelForCausalLM.from_pretrained(
    "CraneAILabs/EduGanda-Gemma-3-1B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
)
model_ref.eval()
tok_ref = AutoTokenizer.from_pretrained("CraneAILabs/EduGanda-Gemma-3-1B")
if tok_ref.pad_token is None:
    tok_ref.pad_token = tok_ref.eos_token
ref_results = evaluate_on_benchmark(model_ref, tok_ref, benchmark, label="EduGanda (log-prob)")
ref_results["ci_lower"], ref_results["ci_upper"] = bootstrap_ci(
    ref_results["predictions"], ref_results["labels"]
)[1:]
ref_gen = evaluate_on_benchmark_generation(model_ref, tok_ref, benchmark, label="EduGanda (generation)")
ref_results["generation_accuracy"] = ref_gen["accuracy"]
del model_ref
torch.cuda.empty_cache()

# Summary
print("\n" + "=" * 60)
print("BASELINE SUMMARY")
print("=" * 60)
for name, r in [("ganda-gemma-1b (base)", base_results), ("EduGanda reference", ref_results)]:
    acc = r["accuracy"] * 100
    lo, hi = r.get("ci_lower", 0) * 100, r.get("ci_upper", 0) * 100
    spread = r["spread"]
    dist = r.get("prediction_distribution", {})
    entropy = r.get("prediction_entropy", 0)
    print(f"\n{name}")
    print(f"  Accuracy:  {acc:.1f}% [{lo:.1f}%–{hi:.1f}%]")
    print(f"  Spread:    {spread:.1f}pp")
    print(f"  Pred dist: A={dist.get('A',0):.1%} B={dist.get('B',0):.1%} "
          f"C={dist.get('C',0):.1%} D={dist.get('D',0):.1%}")
    print(f"  Entropy:   {entropy:.3f} bits (max 2.0)")
    gen_acc = r.get("generation_accuracy")
    if gen_acc is not None:
        print(f"  Generation accuracy: {gen_acc:.1%}  (published method, for comparison)")

with open("results/baseline_results.json", "w") as f:
    json.dump({"base": base_results, "reference": ref_results}, f, indent=2, default=str)

print("\nSaved results/baseline_results.json")
