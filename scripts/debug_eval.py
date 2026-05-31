"""
Diagnostic script — run on 20 benchmark items to verify eval pipeline.
Prints: prompt, gold, log-prob scores, predicted, raw generation.
Run before training to confirm evaluation is working correctly.
"""
import sys
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, '/workspace/EduGanda')
from scripts.core.evaluate import _get_choice_token_ids, ANSWER_TOKENS

MODEL = "CraneAILabs/EduGanda-Gemma-3-1B"  # use reference — should score ~66%
N = 20

print(f"Loading {MODEL}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa"
)
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model.eval()

bench = load_dataset("CraneAILabs/pedagogy-luganda-replaced")["train"]
print(f"Benchmark columns: {bench.column_names}\n")

# --- 1. Show what apply_chat_template actually produces ---
sample_content = "Test question?\n(A) opt1\n(B) opt2\n(C) opt3\n(D) opt4"
tmpl = tok.apply_chat_template(
    [{"role": "user", "content": sample_content}],
    tokenize=False, add_generation_prompt=True
)
print("=== CHAT TEMPLATE OUTPUT (first 200 chars) ===")
print(repr(tmpl[:200]))
print()

# --- 2. Check choice token IDs ---
SUFFIX = "<end_of_turn>\n<start_of_turn>model\n"
choice_ids = _get_choice_token_ids(tok, SUFFIX)
print("=== CHOICE TOKEN IDs ===")
for c, tid in choice_ids.items():
    decoded = tok.decode([tid]) if tid else "MULTI-TOKEN"
    print(f"  {c} → token {tid} → decoded: {repr(decoded)}")
print()

# --- 3. Per-item diagnostic ---
correct_lp, correct_gen = 0, 0
pred_dist_lp = {c: 0 for c in ANSWER_TOKENS}

print("=== PER-ITEM DIAGNOSTIC (first 20 items) ===")
for idx in range(N):
    item = bench[idx]
    gold = item["correct_answer"]
    content = (
        f"Answer with only the letter (A, B, C, or D). Do not explain.\n\n"
        f"{item['luganda_question']}\n"
        f"(A) {item['luganda_answer_a']}\n"
        f"(B) {item['luganda_answer_b']}\n"
        f"(C) {item['luganda_answer_c']}\n"
        f"(D) {item['luganda_answer_d']}"
    )
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True
    )

    # Log-prob scoring
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    with torch.no_grad():
        logits = model(ids).logits[0, -1, :]
    log_probs = torch.log_softmax(logits, dim=-1)
    scores = {c: log_probs[choice_ids[c]].item() for c in ANSWER_TOKENS if choice_ids[c]}
    predicted_lp = max(scores, key=scores.get)
    pred_dist_lp[predicted_lp] += 1
    if predicted_lp == gold:
        correct_lp += 1

    # Generation scoring
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=10, do_sample=False,
            repetition_penalty=1.2, pad_token_id=tok.eos_token_id
        )
    raw_gen = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    # Extract letter
    from scripts.core.data import extract_first_letter
    predicted_gen = extract_first_letter(raw_gen)
    if predicted_gen == gold:
        correct_gen += 1

    # Print
    score_str = "  ".join(f"{c}:{scores.get(c, float('nan')):.2f}" for c in ANSWER_TOKENS)
    match_lp = "✓" if predicted_lp == gold else "✗"
    match_gen = "✓" if predicted_gen == gold else "✗"
    print(f"[{idx:2d}] gold={gold}  lp={predicted_lp}{match_lp}  gen={predicted_gen or '?'}{match_gen}  "
          f"scores: {score_str}  raw: {repr(raw_gen[:40])}")

print(f"\nLog-prob accuracy: {correct_lp}/{N} = {correct_lp/N:.1%}")
print(f"Generation accuracy: {correct_gen}/{N} = {correct_gen/N:.1%}")
print(f"Prediction distribution (log-prob): {pred_dist_lp}")
print(f"\nExpected: ~66% for EduGanda reference (published)")
