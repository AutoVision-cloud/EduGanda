"""
Diagnostic: compare prompt formats to find which matches the published 66%.
The training data has NO English instruction — just the Luganda question + options.
Our "Answer with only the letter..." prefix is likely out-of-distribution.
"""
import sys
import torch
from collections import Counter
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, '/workspace/EduGanda')
from scripts.core.evaluate import _get_choice_token_ids, ANSWER_TOKENS
from scripts.core.data import extract_first_letter

MODEL = "CraneAILabs/EduGanda-Gemma-3-1B"
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
fln   = load_dataset("CraneAILabs/luganda-fln-training-data", "all")["train"]

# Show actual training data format
print("=== TRAINING DATA EXAMPLE ===")
mcq_example = next(r for r in fln if r["format"] == "mcq")
print(mcq_example["text"][:600])
print(f"\ncorrect_letter: {mcq_example['correct_letter']}")
print()

# Gold distribution
gold_dist = Counter(bench[i]["correct_answer"] for i in range(len(bench)))
print(f"Gold distribution (299 items): {dict(gold_dist)}\n")

SUFFIX = "<end_of_turn>\n<start_of_turn>model\n"
choice_ids = _get_choice_token_ids(tok, SUFFIX)

def lp_score(prompt_str):
    ids = tok(prompt_str, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    with torch.no_grad():
        logits = model(ids).logits[0, -1, :]
    lp = torch.log_softmax(logits, dim=-1)
    scores = {c: lp[choice_ids[c]].item() for c in ANSWER_TOKENS if choice_ids[c]}
    return max(scores, key=scores.get), scores

def gen_score(prompt_str):
    ids = tok(prompt_str, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=50, do_sample=False,
                              repetition_penalty=1.2, pad_token_id=tok.eos_token_id)
    raw = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    return extract_first_letter(raw), raw

def make_prompt(item, instruction=""):
    content = ""
    if instruction:
        content = instruction + "\n\n"
    content += (
        f"{item['luganda_question']}\n"
        f"(A) {item['luganda_answer_a']}\n"
        f"(B) {item['luganda_answer_b']}\n"
        f"(C) {item['luganda_answer_c']}\n"
        f"(D) {item['luganda_answer_d']}"
    )
    return tok.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True
    )

formats = {
    "with_instr":    "Answer with only the letter (A, B, C, or D). Do not explain.",
    "no_instr":      "",
    "en_short":      "Choose the correct answer: A, B, C, or D.",
}

correct_lp = {k: 0 for k in formats}
correct_gen = {k: 0 for k in formats}
dist_lp = {k: Counter() for k in formats}

print(f"{'#':>3}  gold  " + "  ".join(f"lp_{k[:8]}" for k in formats))
print("-"*80)

for idx in range(N):
    item = bench[idx]
    gold = item["correct_answer"]

    row = f"[{idx:2d}]  {gold}   "
    for k, instr in formats.items():
        prompt = make_prompt(item, instr)
        pred_lp, _ = lp_score(prompt)
        pred_gen, raw = gen_score(prompt)
        dist_lp[k][pred_lp] += 1
        if pred_lp == gold: correct_lp[k] += 1
        if pred_gen == gold: correct_gen[k] += 1
        m = "✓" if pred_lp == gold else "✗"
        row += f"  {pred_lp}{m}(gen:{pred_gen or '?'} {repr(raw[:15])})"
    print(row)

print(f"\n{'Format':<15} LP_acc  Gen_acc  Distribution")
for k in formats:
    print(f"  {k:<13} {correct_lp[k]}/{N}={correct_lp[k]/N:.0%}    "
          f"{correct_gen[k]}/{N}={correct_gen[k]/N:.0%}    {dict(dist_lp[k])}")
print(f"\nExpected EduGanda: ~66% (published)")
