"""
Baseline Evaluation — run BEFORE any training.

Three dimensions per model:
  1. MCQ accuracy (log-prob scoring — frozen eval protocol)
  2. MCQ accuracy (generation — for comparison with published results)
  3. Open-ended generation quality (reward model scores + repetition rate)

Note: Published 66% PCK for EduGanda was not reproducible under this protocol.
See README → Evaluation Protocol for full explanation.
"""

import json
import os
import statistics
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from scripts.core.evaluate import (
    evaluate_on_benchmark,
    bootstrap_ci,
    check_tokenization,
)

os.makedirs("results", exist_ok=True)


# ---------------------------------------------------------------------------
# Generation quality helpers
# ---------------------------------------------------------------------------

def _score_texts(texts, reward_model, reward_tokenizer):
    scores = []
    for text in texts:
        inputs = reward_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(reward_model.device)
        with torch.no_grad():
            logits = reward_model(**inputs).logits
        score = (torch.softmax(logits, dim=-1)[0, 1] if logits.shape[-1] == 2
                 else logits[0, 0]).item()
        scores.append(score)
    return scores


def _repetition_rate(text: str, ngram: int = 5) -> float:
    tokens = text.split()
    if len(tokens) < ngram:
        return 0.0
    grams = [tuple(tokens[i:i + ngram]) for i in range(len(tokens) - ngram + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def _eval_generation_quality(model, tokenizer, reward_model, reward_tokenizer,
                              prompts, label):
    """Reward model scores + repetition rate, with and without repetition_penalty."""
    model.eval()
    results = {"label": label, "n_prompts": len(prompts)}

    for penalty, key in [(1.2, "with_penalty"), (1.0, "no_penalty")]:
        outputs, rep_rates, full_texts = [], [], []
        for prompt_text in prompts:
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=False, add_generation_prompt=True,
            )
            ids = tokenizer(formatted, return_tensors="pt",
                            add_special_tokens=False).input_ids.to(model.device)
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=150, do_sample=False,
                                     repetition_penalty=penalty,
                                     pad_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(out[0][ids.shape[1]:],
                                        skip_special_tokens=True).strip()
            outputs.append(response)
            rep_rates.append(_repetition_rate(response))
            full_texts.append(formatted + response)

        scores = _score_texts(full_texts, reward_model, reward_tokenizer)
        results[key] = {
            "mean_reward": round(statistics.mean(scores), 4),
            "std_reward": round(statistics.stdev(scores) if len(scores) > 1 else 0, 4),
            "mean_repetition_rate": round(statistics.mean(rep_rates), 4),
            "sample_outputs": outputs[:3],
        }
        print(f"  [{label} pen={penalty}]  reward={results[key]['mean_reward']:.3f}  "
              f"repetition={results[key]['mean_repetition_rate']:.3f}")
    return results


def _load_model(model_id):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16,
        device_map="auto", attn_implementation="sdpa",
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def eval_model(model_id, label, benchmark, reward_model, reward_tokenizer, gen_prompts):
    """Load, evaluate on all three dimensions, unload. Returns results dict."""
    print(f"\n{'='*60}\n{label}\n{'='*60}")
    model, tok = _load_model(model_id)

    print("\n--- MCQ Accuracy ---")
    results = evaluate_on_benchmark(model, tok, benchmark, label=label)
    results["ci_lower"], results["ci_upper"] = bootstrap_ci(
        results["predictions"], results["labels"])[1:]

    print("\n--- Open-ended Generation ---")
    results["generation_eval"] = _eval_generation_quality(
        model, tok, reward_model, reward_tokenizer, gen_prompts, label=label)

    del model
    torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

benchmark = load_dataset("CraneAILabs/pedagogy-luganda-replaced")

# Generation prompts from FLN content subset (non-MCQ items)
fln_content = load_dataset("CraneAILabs/luganda-fln-training-data", "fln_content")["train"]
gen_prompts = []
for row in fln_content:
    text = row.get("text", "")
    if "<start_of_turn>user\n" in text and "<end_of_turn>" in text:
        user_part = text.split("<start_of_turn>user\n")[1].split("<end_of_turn>")[0].strip()
        gen_prompts.append(user_part)
    if len(gen_prompts) >= 20:
        break
if len(gen_prompts) < 5:
    gen_prompts = [
        "Nkola ekikolwa eky'okusomesa abayizi ba P1 okumanya ennyingo.",
        "Nkola olukalala lw'ebikolwa okusomesa abayizi ba P2 okumanya okuwandiika.",
        "Nkola ekigendererwa eky'okusomesa abayizi ba P3 mu somo ly'endimi.",
        "Nkola ekikolwa eky'okusomesa abayizi okumanya ennyingo z'okusoma.",
        "Nkola olukalala lw'ebibuuzo okusomesa abayizi ba P1 okumanya ensimbi.",
    ]
print(f"Generation prompts: {len(gen_prompts)} items\n")

# Tokenization verification (run once, uses base model tokenizer)
print("=" * 60 + "\nTOKENIZATION VERIFICATION\n" + "=" * 60)
_tok = AutoTokenizer.from_pretrained("CraneAILabs/ganda-gemma-1b")
check_tokenization(_tok)
for letter in ["A", "B", "C", "D"]:
    n = len(_tok.encode(letter, add_special_tokens=False))
    print(f"  '{letter}' → {n} token(s)  {'✓' if n == 1 else f'WARNING: {n} tokens'}")
del _tok

# Reward model (shared, loaded once)
print("\nLoading reward model...")
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "CraneAILabs/luganda-reward-model",
    torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
)
reward_tokenizer = AutoTokenizer.from_pretrained("CraneAILabs/luganda-reward-model")
print(f"  {reward_model.config.num_labels} label(s), "
      f"id2label={getattr(reward_model.config, 'id2label', 'not set')}\n")

# ---------------------------------------------------------------------------
# Evaluate both models
# ---------------------------------------------------------------------------

base_results = eval_model(
    "CraneAILabs/ganda-gemma-1b", "BASELINE: ganda-gemma-1b",
    benchmark, reward_model, reward_tokenizer, gen_prompts,
)
ref_results = eval_model(
    "CraneAILabs/EduGanda-Gemma-3-1B", "REFERENCE: EduGanda-Gemma-3-1B",
    benchmark, reward_model, reward_tokenizer, gen_prompts,
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70 + "\nBASELINE SUMMARY\n" + "=" * 70)
for name, r in [("ganda-gemma-1b", base_results), ("EduGanda reference", ref_results)]:
    lo, hi = r.get("ci_lower", 0) * 100, r.get("ci_upper", 0) * 100
    dist = r.get("prediction_distribution", {})
    ge = r.get("generation_eval", {})
    print(f"\n{name}")
    print(f"  MCQ accuracy:  {r['accuracy']*100:.1f}% [{lo:.1f}%–{hi:.1f}%]  "
          f"spread={r['spread']:.1f}pp")
    print(f"  Pred dist: A={dist.get('A',0):.1%} B={dist.get('B',0):.1%} "
          f"C={dist.get('C',0):.1%} D={dist.get('D',0):.1%}  "
          f"entropy={r.get('prediction_entropy',0):.3f}")
    if ge:
        wp, np_ = ge.get("with_penalty", {}), ge.get("no_penalty", {})
        print(f"  Reward (pen=1.2): {wp.get('mean_reward',0):.3f}±{wp.get('std_reward',0):.3f}  "
              f"rep={wp.get('mean_repetition_rate',0):.3f}")
        print(f"  Reward (no pen):  {np_.get('mean_reward',0):.3f}  "
              f"rep={np_.get('mean_repetition_rate',0):.3f}")

with open("results/baseline_results.json", "w") as f:
    json.dump({"base": base_results, "reference": ref_results}, f,
              indent=2, default=str)
print("\nSaved results/baseline_results.json")
