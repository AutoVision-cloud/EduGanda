"""
Diagnostic script — compares two prompt strategies on 20 benchmark items.
Hypothesis: EduGanda was trained to output "Okuddamu: X" (Luganda for "Answer: X").
Priming the model turn with "Okuddamu: " should give much better accuracy.
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

# Gold distribution
gold_dist = Counter(bench[i]["correct_answer"] for i in range(len(bench)))
print(f"Gold distribution (299 items): {dict(gold_dist)}\n")

# Chat template check
sample = tok.apply_chat_template(
    [{"role": "user", "content": "test"}], tokenize=False, add_generation_prompt=True
)
print(f"Chat template: {repr(sample[:80])}\n")

# Token IDs for both suffix styles
SUFFIX_STD   = "<end_of_turn>\n<start_of_turn>model\n"
SUFFIX_OKUD  = "<end_of_turn>\n<start_of_turn>model\nOkuddamu: "
ids_std  = _get_choice_token_ids(tok, SUFFIX_STD)
ids_okud = _get_choice_token_ids(tok, SUFFIX_OKUD)
print("Token IDs (standard suffix):", {c: (v, tok.decode([v])) for c,v in ids_std.items() if v})
print("Token IDs (Okuddamu suffix):", {c: (v, tok.decode([v])) for c,v in ids_okud.items() if v})
print()

correct = {"lp_std": 0, "lp_okud": 0, "gen_std": 0, "gen_okud": 0}
dist    = {"lp_std": Counter(), "lp_okud": Counter()}

print(f"{'#':>3}  gold  lp_std  lp_okud  gen_std                 gen_okud")
print("-"*90)

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

    base_prompt = tok.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True
    )
    okud_prompt = base_prompt + "Okuddamu: "

    def score(prompt_str, tid_map):
        ids = tok(prompt_str, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        with torch.no_grad():
            logits = model(ids).logits[0, -1, :]
        lp = torch.log_softmax(logits, dim=-1)
        return {c: lp[tid_map[c]].item() for c in ANSWER_TOKENS if tid_map[c]}, ids

    def gen(ids):
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=50, do_sample=False,
                                  repetition_penalty=1.2, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

    sc_std,  ids_s = score(base_prompt, ids_std)
    sc_okud, ids_o = score(okud_prompt, ids_okud)

    p_lp_std  = max(sc_std,  key=sc_std.get)
    p_lp_okud = max(sc_okud, key=sc_okud.get)
    dist["lp_std"][p_lp_std]   += 1
    dist["lp_okud"][p_lp_okud] += 1
    if p_lp_std  == gold: correct["lp_std"]  += 1
    if p_lp_okud == gold: correct["lp_okud"] += 1

    raw_std  = gen(ids_s)
    raw_okud = gen(ids_o)
    p_gen_std  = extract_first_letter(raw_std)
    p_gen_okud = extract_first_letter(raw_okud)
    if p_gen_std  == gold: correct["gen_std"]  += 1
    if p_gen_okud == gold: correct["gen_okud"] += 1

    m = lambda p, g: "✓" if p==g else "✗"
    print(f"[{idx:2d}]  {gold}    "
          f"{p_lp_std}{m(p_lp_std,gold)}      {p_lp_okud}{m(p_lp_okud,gold)}      "
          f"{repr(raw_std[:20]):24s}  {repr(raw_okud[:20])}")

print()
print(f"{'Method':<16} Acc    Distribution")
for k, c in correct.items():
    d = dict(dist.get(k, {}))
    print(f"  {k:<14} {c}/{N}={c/N:.0%}  {d}")
print(f"\nExpected EduGanda: ~66% (published, generation-based)")
