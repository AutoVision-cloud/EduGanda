"""
Baseline Evaluation — run BEFORE any training.

Evaluates ganda-gemma-1b (CPT only) and EduGanda-Gemma-3-1B (reference)
on the LLPK benchmark using generation-based MCQ scoring.

Eval protocol (frozen): training-format prompt (Luganda question + options,
no instruction prefix), repetition_penalty=1.2, extract letter from output.
"""

import json
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from scripts.core.evaluate import evaluate_on_benchmark, bootstrap_ci, check_tokenization

os.makedirs("results", exist_ok=True)


def eval_model(model_id, label, benchmark):
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa",
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("\n[free generation — deployment behaviour]")
    results = evaluate_on_benchmark(model, tok, benchmark,
                                    label=label, forced_format=False)
    results["ci_lower"], results["ci_upper"] = bootstrap_ci(
        results["predictions"], results["labels"])[1:]

    print("\n[forced format: 'Okuddamu: ' primer — MCQ accuracy]")
    forced = evaluate_on_benchmark(model, tok, benchmark,
                                   label=f"{label} (forced)", forced_format=True)
    forced["ci_lower"], forced["ci_upper"] = bootstrap_ci(
        forced["predictions"], forced["labels"])[1:]
    results["forced"] = forced

    del model
    torch.cuda.empty_cache()
    return results


# Tokenization check
print("=" * 60 + "\nTOKENIZATION VERIFICATION\n" + "=" * 60)
_tok = AutoTokenizer.from_pretrained("CraneAILabs/ganda-gemma-1b")
check_tokenization(_tok)
del _tok

benchmark = load_dataset("CraneAILabs/pedagogy-luganda-replaced")

base_results = eval_model("CraneAILabs/ganda-gemma-1b",
                          "BASELINE: ganda-gemma-1b", benchmark)
ref_results  = eval_model("CraneAILabs/EduGanda-Gemma-3-1B",
                          "REFERENCE: EduGanda-Gemma-3-1B", benchmark)

# Summary
print("\n" + "=" * 60 + "\nBASELINE SUMMARY\n" + "=" * 60)
for name, r in [("ganda-gemma-1b", base_results), ("EduGanda reference", ref_results)]:
    lo, hi = r.get("ci_lower", 0) * 100, r.get("ci_upper", 0) * 100
    dist = r.get("prediction_distribution", {})
    f = r.get("forced", {})
    print(f"\n{name}")
    print(f"  Free gen:   acc={r['accuracy']*100:.1f}% [{lo:.1f}%–{hi:.1f}%]  "
          f"spread={r['spread']:.1f}pp  invalid={r.get('invalid_parse_rate',0):.1%}")
    print(f"    dist: A={dist.get('A',0):.1%} B={dist.get('B',0):.1%} "
          f"C={dist.get('C',0):.1%} D={dist.get('D',0):.1%}  "
          f"entropy={r.get('prediction_entropy',0):.3f}")
    if f:
        fd = f.get("prediction_distribution", {})
        flo, fhi = f.get("ci_lower", 0)*100, f.get("ci_upper", 0)*100
        print(f"  Forced fmt: acc={f['accuracy']*100:.1f}% [{flo:.1f}%–{fhi:.1f}%]  "
              f"spread={f['spread']:.1f}pp  invalid={f.get('invalid_parse_rate',0):.1%}")
        print(f"    dist: A={fd.get('A',0):.1%} B={fd.get('B',0):.1%} "
              f"C={fd.get('C',0):.1%} D={fd.get('D',0):.1%}  "
              f"entropy={f.get('prediction_entropy',0):.3f}")

with open("results/baseline_results.json", "w") as f:
    json.dump({"base": base_results, "reference": ref_results}, f,
              indent=2, default=str)
print("\nSaved results/baseline_results.json")
