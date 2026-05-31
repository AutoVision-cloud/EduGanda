"""
Sanity check 1: Does ganda-gemma-1b actually improve over Gemma base for Luganda?
Compares google/gemma-3-1b-it, ganda-gemma-1b, and EduGanda on qualitative
Luganda generation. Run BEFORE SFT.

Usage: python scripts/sanity_cpt.py
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = [
    "google/gemma-3-1b-it",
    "CraneAILabs/ganda-gemma-1b",
    "CraneAILabs/EduGanda-Gemma-3-1B",
]

PROMPTS = [
    # Fluency / vocabulary
    "Wandiika emboozi nnyimpi ku mwana agenda ku ssomero.",
    "Nnyonnyola enjawulo wakati w'ennyingo n'ekigambo.",
    # Education task
    "Kola ebibuuzo bina eby'abayizi ba P2 mu Luganda.",
    # Chat format test (matches training data format exactly)
    "Omusomesa ayagala okusomesa abayizi okumanya ennyingo. Kola ekikolwa kimu.",
]

for model_id in MODELS:
    print("\n" + "=" * 80)
    print(model_id)
    print("=" * 80)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa",
    )
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    for prompt in PROMPTS:
        text = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True,
        )
        ids = tok(text, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            out = model.generate(
                **ids, max_new_tokens=120, do_sample=False,
                repetition_penalty=1.2, pad_token_id=tok.eos_token_id,
            )
        response = tok.decode(out[0][ids["input_ids"].shape[1]:],
                               skip_special_tokens=True).strip()
        print(f"\n  PROMPT: {prompt}")
        print(f"  RESPONSE: {response[:400]}")

    del model
    torch.cuda.empty_cache()
