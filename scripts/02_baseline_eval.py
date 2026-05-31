"""
Hour 2: Baseline Evaluation
Evaluates ganda-gemma-1b (CPT only) and EduGanda-Gemma-3-1B (reference).

Three evaluation dimensions:
  1. MCQ accuracy (LLPK benchmark, 299 items)
     - Primary: log-probability option scoring (principled, avoids parser sensitivity)
     - Secondary: generation scoring (matches published methodology)
  2. Open-ended generation quality (reward model scores on lesson plan prompts)
  3. Repetition rate (with vs without penalty — validates the blog's finding)
"""

import json
import os
import statistics
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
from scripts.core.evaluate import evaluate_on_benchmark, evaluate_on_benchmark_generation, bootstrap_ci, check_tokenization

os.makedirs("results", exist_ok=True)

# ---------------------------------------------------------------------------
# Helper: open-ended generation evaluation
# ---------------------------------------------------------------------------

def _score_texts(texts, reward_model, reward_tokenizer):
    scores = []
    for text in texts:
        inputs = reward_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(reward_model.device)
        with torch.no_grad():
            logits = reward_model(**inputs).logits
        if logits.shape[-1] == 2:
            score = torch.softmax(logits, dim=-1)[0, 1].item()
        else:
            score = logits[0, 0].item()
        scores.append(score)
    return scores


def _repetition_rate(text: str, ngram: int = 5) -> float:
    """Fraction of n-grams that are repeated. 0=no repetition, 1=fully repetitive."""
    tokens = text.split()
    if len(tokens) < ngram:
        return 0.0
    grams = [tuple(tokens[i:i+ngram]) for i in range(len(tokens) - ngram + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def evaluate_open_generation(model, tokenizer, reward_model, reward_tokenizer,
                              generation_prompts, label=""):
    """
    Generates open-ended responses and evaluates with the reward model.
    Tests both with and without repetition_penalty to document the failure mode.
    """
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {"label": label, "n_prompts": len(generation_prompts)}

    for penalty, key in [(1.2, "with_penalty"), (1.0, "no_penalty")]:
        outputs, rep_rates, full_texts = [], [], []
        for prompt_text in generation_prompts:
            # apply_chat_template adds BOS + proper special tokens
            formatted = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt_text}],
                tokenize=False, add_generation_prompt=True,
            )
            ids = tokenizer(formatted, return_tensors="pt",
                            add_special_tokens=False).input_ids.to(model.device)
            with torch.no_grad():
                out = model.generate(
                    ids,
                    max_new_tokens=150,
                    do_sample=False,
                    repetition_penalty=penalty,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response = tokenizer.decode(
                out[0][ids.shape[1]:], skip_special_tokens=True
            ).strip()
            outputs.append(response)
            rep_rates.append(_repetition_rate(response))
            # Reward model expects full conversation context
            full_texts.append(formatted + response)

        scores = _score_texts(full_texts, reward_model, reward_tokenizer)
        results[key] = {
            "mean_reward": round(statistics.mean(scores), 4),
            "std_reward": round(statistics.stdev(scores) if len(scores) > 1 else 0, 4),
            "mean_repetition_rate": round(statistics.mean(rep_rates), 4),
            "sample_outputs": outputs[:3],
        }
        print(f"  [{label} | rep_penalty={penalty}]  "
              f"reward={results[key]['mean_reward']:.3f}  "
              f"repetition={results[key]['mean_repetition_rate']:.3f}")

    return results


# ---------------------------------------------------------------------------
# Load shared resources
# ---------------------------------------------------------------------------

benchmark = load_dataset("CraneAILabs/pedagogy-luganda-replaced")

# Generation prompts: sampled from non-MCQ FLN items (content generation format)
# These ask for lesson plans, exercises, and teaching activities in Luganda.
fln = load_dataset("CraneAILabs/luganda-fln-training-data", "fln_content")["train"]
GEN_PROMPTS = []
for row in fln:
    text = row.get("text", "")
    if "<start_of_turn>user\n" in text and "<end_of_turn>" in text:
        user_part = text.split("<start_of_turn>user\n")[1].split("<end_of_turn>")[0].strip()
        GEN_PROMPTS.append(user_part)
    if len(GEN_PROMPTS) >= 20:
        break

# Fallback prompts if FLN content subset is too small
FALLBACK_PROMPTS = [
    "Nkola ekikolwa eky'okusomesa abayizi ba P1 okumanya ennyingo. Nkole etunula okukyusa amaanyi g'okusoma.",
    "Nkola olukalala lw'ebikolwa okusomesa abayizi ba P2 okumanya okuwandiika ebyayigirwa.",
    "Nkola ekigendererwa eky'okusomesa abayizi ba P3 mu somo ly'endimi okuhulikira n'okuddamu.",
    "Nkola ekikolwa eky'okusomesa abayizi okumanya ennyingo z'okusoma mu Luganda.",
    "Nkola olukalala lw'ebibuuzo okusomesa abayizi ba P1 okumanya ensimbi.",
]
if len(GEN_PROMPTS) < 5:
    GEN_PROMPTS = FALLBACK_PROMPTS

print(f"Using {len(GEN_PROMPTS)} generation prompts from FLN content subset.")

# Tokenization verification
print("\n" + "=" * 60)
print("TOKENIZATION VERIFICATION")
print("=" * 60)
_tok_check = AutoTokenizer.from_pretrained("CraneAILabs/ganda-gemma-1b")
check_tokenization(_tok_check)
for letter in ["A", "B", "C", "D"]:
    n = len(_tok_check.encode(letter, add_special_tokens=False))
    print(f"  '{letter}' → {n} token(s)  {'✓' if n == 1 else f'WARNING: {n} tokens'}")
del _tok_check
print()

# Load reward model (shared across both evals)
print("Loading reward model...")
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "CraneAILabs/luganda-reward-model",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
reward_tokenizer = AutoTokenizer.from_pretrained("CraneAILabs/luganda-reward-model")
print(f"  Reward model: {reward_model.config.num_labels} labels, "
      f"id2label={getattr(reward_model.config, 'id2label', 'not set')}\n")

# ---------------------------------------------------------------------------
# 1. ganda-gemma-1b (CPT only — no education SFT)
# ---------------------------------------------------------------------------
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

print("\n--- MCQ Accuracy ---")
base_results = evaluate_on_benchmark(
    model_base, tok_base, benchmark, label="ganda-gemma-1b (log-prob)")
base_results["ci_lower"], base_results["ci_upper"] = bootstrap_ci(
    base_results["predictions"], base_results["labels"])[1:]
base_gen = evaluate_on_benchmark_generation(
    model_base, tok_base, benchmark, label="ganda-gemma-1b (generation)")
base_results["generation_accuracy"] = base_gen["accuracy"]

print("\n--- Open-ended Generation ---")
base_gen_eval = evaluate_open_generation(
    model_base, tok_base, reward_model, reward_tokenizer, GEN_PROMPTS,
    label="ganda-gemma-1b")
base_results["generation_eval"] = base_gen_eval

del model_base
torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# 2. EduGanda-Gemma-3-1B (reference — target to match/beat)
# ---------------------------------------------------------------------------
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

print("\n--- MCQ Accuracy ---")
ref_results = evaluate_on_benchmark(
    model_ref, tok_ref, benchmark, label="EduGanda (log-prob)")
ref_results["ci_lower"], ref_results["ci_upper"] = bootstrap_ci(
    ref_results["predictions"], ref_results["labels"])[1:]
ref_gen = evaluate_on_benchmark_generation(
    model_ref, tok_ref, benchmark, label="EduGanda (generation)")
ref_results["generation_accuracy"] = ref_gen["accuracy"]

print("\n--- Open-ended Generation ---")
ref_gen_eval = evaluate_open_generation(
    model_ref, tok_ref, reward_model, reward_tokenizer, GEN_PROMPTS,
    label="EduGanda")
ref_results["generation_eval"] = ref_gen_eval

del model_ref
torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("BASELINE SUMMARY")
print("=" * 70)
for name, r in [("ganda-gemma-1b (base)", base_results),
                ("EduGanda reference", ref_results)]:
    acc_lp = r["accuracy"] * 100
    lo, hi = r.get("ci_lower", 0) * 100, r.get("ci_upper", 0) * 100
    acc_gen = r.get("generation_accuracy", 0) * 100
    spread = r["spread"]
    dist = r.get("prediction_distribution", {})
    entropy = r.get("prediction_entropy", 0)
    ge = r.get("generation_eval", {})

    print(f"\n{name}")
    print(f"  MCQ accuracy (log-prob):   {acc_lp:.1f}% [{lo:.1f}%–{hi:.1f}%]")
    print(f"  MCQ accuracy (generation): {acc_gen:.1f}%  ← published method")
    print(f"  Position bias spread:      {spread:.1f}pp")
    print(f"  Pred dist: A={dist.get('A',0):.1%} B={dist.get('B',0):.1%} "
          f"C={dist.get('C',0):.1%} D={dist.get('D',0):.1%}  entropy={entropy:.3f}")

    if ge:
        wp = ge.get("with_penalty", {})
        np_ = ge.get("no_penalty", {})
        print(f"  Reward score (penalty=1.2): {wp.get('mean_reward', 0):.3f} ± "
              f"{wp.get('std_reward', 0):.3f}")
        print(f"  Reward score (no penalty):  {np_.get('mean_reward', 0):.3f}  "
              f"repetition={np_.get('mean_repetition_rate', 0):.3f}")

with open("results/baseline_results.json", "w") as f:
    json.dump({"base": base_results, "reference": ref_results}, f,
              indent=2, default=str)

print("\nSaved results/baseline_results.json")
