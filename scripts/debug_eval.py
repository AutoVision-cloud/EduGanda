"""
Diagnostic: check benchmark splits and try pedagogy-luganda-reviewed.
The published 66% might be on the 'reviewed' dataset, not 'replaced'.
Also inspect the split column to see if there's a test subset.
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
N = 40

print(f"Loading {MODEL}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa"
)
tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model.eval()

# --- Inspect both benchmarks ---
for ds_name in ["CraneAILabs/pedagogy-luganda-replaced",
                "CraneAILabs/pedagogy-luganda-reviewed"]:
    ds = load_dataset(ds_name)["train"]
    split_dist = Counter(r.get("split", "?") for r in ds)
    gold_dist  = Counter(r["correct_answer"] for r in ds)
    status_dist = Counter(r.get("review_status", "?") for r in ds)
    proc_dist  = Counter(r.get("processing_status", "?") for r in ds)
    print(f"\n=== {ds_name} ({len(ds)} rows) ===")
    print(f"  split values:     {dict(split_dist)}")
    print(f"  gold distribution:{dict(gold_dist)}")
    print(f"  review_status:    {dict(status_dist)}")
    print(f"  processing_status:{dict(proc_dist)}")
    print(f"  first item split={ds[0].get('split')} review_status={ds[0].get('review_status')}")

SUFFIX = "<end_of_turn>\n<start_of_turn>model\n"
choice_ids = _get_choice_token_ids(tok, SUFFIX)

def eval_on(bench, label, instruction="", n=None):
    items = list(bench)[:n] if n else list(bench)
    correct_lp = correct_gen = 0
    dist_lp = Counter()
    for item in items:
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
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False, add_generation_prompt=True
        )
        ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        with torch.no_grad():
            logits = model(ids).logits[0, -1, :]
        lp = torch.log_softmax(logits, dim=-1)
        scores = {c: lp[choice_ids[c]].item() for c in ANSWER_TOKENS if choice_ids[c]}
        pred_lp = max(scores, key=scores.get)
        dist_lp[pred_lp] += 1
        if pred_lp == item["correct_answer"]: correct_lp += 1

        # Generation
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=50, do_sample=False,
                                  repetition_penalty=1.2, pad_token_id=tok.eos_token_id)
        raw = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        pred_gen = extract_first_letter(raw)
        if pred_gen == item["correct_answer"]: correct_gen += 1

    n_total = len(items)
    print(f"  {label:<50} LP={correct_lp}/{n_total}={correct_lp/n_total:.1%}  "
          f"Gen={correct_gen}/{n_total}={correct_gen/n_total:.1%}  dist={dict(dist_lp)}")

print("\n\n=== ACCURACY COMPARISON ===")
for ds_name, short in [("CraneAILabs/pedagogy-luganda-replaced", "replaced"),
                        ("CraneAILabs/pedagogy-luganda-reviewed",  "reviewed")]:
    bench = load_dataset(ds_name)["train"]
    print(f"\nDataset: {ds_name}")
    eval_on(bench, f"{short} / no_instr",    instruction="", n=N)
    eval_on(bench, f"{short} / with_instr",
            instruction="Answer with only the letter (A, B, C, or D). Do not explain.", n=N)

print("\nExpected EduGanda: ~66% PCK / ~58.8% LLK")
