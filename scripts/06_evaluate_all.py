"""
Hours 8-9: Full Evaluation Sweep with statistical comparisons.
"""
import os
import json
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.core.evaluate import evaluate_on_benchmark, bootstrap_ci, mcnemar_test, compute_calibration

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

missing = [k for k, p in models_to_eval.items() if p.startswith("./") and not os.path.isdir(p)]
if missing:
    print(f"WARNING: skipping missing checkpoints: {missing}")
    models_to_eval = {k: v for k, v in models_to_eval.items() if k not in missing}

all_results = {}
for name, path in models_to_eval.items():
    print(f"\n{'='*60}\nEvaluating: {name}\n{'='*60}")
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
    tok = AutoTokenizer.from_pretrained(path)
    result = evaluate_on_benchmark(model, tok, benchmark, label=name)
    acc, lo, hi = bootstrap_ci(result["predictions"], result["labels"])
    result["ci_lower"] = lo
    result["ci_upper"] = hi
    result["calibration"] = compute_calibration(
        result["confidences"],
        [p == l for p, l in zip(result["predictions"], result["labels"])],
    )
    all_results[name] = result
    del model
    torch.cuda.empty_cache()

# McNemar comparisons vs LEARNER
if "LEARNER (SFT, balanced)" in all_results:
    learner_preds = all_results["LEARNER (SFT, balanced)"]["predictions"]
    learner_labels = all_results["LEARNER (SFT, balanced)"]["labels"]
    for name, r in all_results.items():
        if name == "LEARNER (SFT, balanced)":
            continue
        r["mcnemar_vs_learner_p"] = mcnemar_test(learner_preds, r["predictions"], learner_labels)

with open("results/full_results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)

print("\n\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(f"{'Model':<30} {'Acc':>6} {'95% CI':>16} {'Spread':>8}  Notes")
print("-" * 80)
for name, r in all_results.items():
    acc = r["accuracy"] * 100
    lo, hi = r.get("ci_lower", 0) * 100, r.get("ci_upper", 0) * 100
    spread = r["spread"]
    p = r.get("mcnemar_vs_learner_p")
    sig = f"p={p:.3f}" if p is not None else ""
    print(f"{name:<30} {acc:>5.1f}% [{lo:>4.1f}%–{hi:>4.1f}%] {spread:>6.1f}pp  {sig}")

print("\nSaved results/full_results.json")

# Per-subdomain and per-age-group breakdown for the best model
best_name = max(all_results, key=lambda n: all_results[n]["accuracy"])
best = all_results[best_name]
print(f"\n--- Breakdown for best model: {best_name} ---")

print("\nPer-subdomain accuracy:")
for sub, s in sorted(best.get("subdomain_stats", {}).items()):
    if s["total"] > 0:
        print(f"  {sub:<30} {s['correct']/s['total']:.1%} ({s['total']} items)")

print("\nPer-age-group accuracy:")
for age, s in sorted(best.get("age_group_stats", {}).items()):
    if s["total"] > 0:
        print(f"  {age:<20} {s['correct']/s['total']:.1%} ({s['total']} items)")
