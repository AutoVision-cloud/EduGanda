"""
Tokenizer fertility analysis: how many subword tokens does Gemma's tokenizer
use per word of Luganda vs English text?

High fertility = model burns context budget on fragmentation.
Gemma's tokenizer was trained on English-dominated data; Luganda words are
likely split into many more subwords, compressing effective context length.

Usage:
  python scripts/diagnostics/tokenizer_fertility.py

No GPU required — tokenizer-only, runs locally.
"""
import json
import os
import numpy as np
from typing import List, Dict


def compute_fertility(tokenizer, texts: List[str]) -> Dict:
    """
    Fertility = tokens / whitespace-words for each text, then averaged.
    overall_fertility = sum(tokens) / sum(words) across all texts.
    """
    per_text = []
    total_words = total_tokens = 0

    for text in texts:
        words = text.strip().split()
        if not words:
            continue
        tokens = tokenizer.encode(text, add_special_tokens=False)
        n_tokens = len(tokens)
        n_words = len(words)
        per_text.append(n_tokens / n_words)
        total_words += n_words
        total_tokens += n_tokens

    return {
        "mean_fertility": float(np.mean(per_text)) if per_text else 0.0,
        "median_fertility": float(np.median(per_text)) if per_text else 0.0,
        "std_fertility": float(np.std(per_text)) if per_text else 0.0,
        "overall_fertility": total_tokens / total_words if total_words > 0 else 0.0,
        "total_words": total_words,
        "total_tokens": total_tokens,
        "n_texts": len(per_text),
    }


def effective_context_words(fertility: float, context_length: int = 2048) -> int:
    """How many words fit in context_length tokens at a given fertility ratio."""
    return int(context_length / fertility) if fertility > 0 else 0


def main():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("CraneAILabs/ganda-gemma-1b")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size:,}")

    # --- Try loading the dedicated tokenizer evaluation dataset ---
    luganda_texts, english_texts = [], []
    try:
        from datasets import load_dataset
        ds = load_dataset("CraneAILabs/luganda-tokenizer-evaluation")
        split = ds[list(ds.keys())[0]]
        cols = split.column_names
        print(f"Tokenizer eval dataset columns: {cols}")

        # Heuristic: pick the Luganda and English text columns
        lug_col = next((c for c in cols if "lug" in c.lower()), None)
        eng_col = next((c for c in cols if "eng" in c.lower() or "en" == c.lower()), None)
        if lug_col:
            luganda_texts = [str(r[lug_col]) for r in split if r.get(lug_col)]
        if eng_col:
            english_texts = [str(r[eng_col]) for r in split if r.get(eng_col)]
        print(f"Loaded {len(luganda_texts)} Luganda, {len(english_texts)} English samples from dataset.")
    except Exception as e:
        print(f"Could not load tokenizer eval dataset ({e}); falling back to FLN training data.")

    # --- Fallback: extract Luganda/English from FLN training data ---
    if not luganda_texts:
        from datasets import load_dataset
        fln = load_dataset("CraneAILabs/luganda-fln-training-data", "all")["train"]
        bench = load_dataset("CraneAILabs/pedagogy-luganda-replaced")["train"]

        # Extract Luganda text from FLN (strip chat template)
        for row in fln:
            text = row.get("text", "")
            if "<start_of_turn>user\n" in text and "<end_of_turn>" in text:
                user_part = text.split("<start_of_turn>user\n")[1].split("<end_of_turn>")[0]
                luganda_texts.append(user_part.strip())

        # English = benchmark English questions
        english_texts = [r["english_question"] for r in bench if r.get("english_question")]
        print(f"Fallback: {len(luganda_texts)} Luganda, {len(english_texts)} English samples from FLN/benchmark.")

    # --- Compute fertility ---
    lug_stats = compute_fertility(tokenizer, luganda_texts[:500])
    eng_stats = compute_fertility(tokenizer, english_texts[:500])

    ratio = lug_stats["overall_fertility"] / eng_stats["overall_fertility"] if eng_stats["overall_fertility"] > 0 else 0

    print("\n--- Tokenizer Fertility Report ---")
    print(f"{'Metric':<30} {'Luganda':>10} {'English':>10}")
    print("-" * 52)
    for key in ["mean_fertility", "median_fertility", "overall_fertility"]:
        print(f"{key:<30} {lug_stats[key]:>10.3f} {eng_stats[key]:>10.3f}")
    print(f"\nLuganda/English fertility ratio: {ratio:.2f}x")
    print(f"  → Luganda uses ~{ratio:.1f}x more tokens per word than English")

    lug_eff = effective_context_words(lug_stats["overall_fertility"])
    eng_eff = effective_context_words(eng_stats["overall_fertility"])
    print(f"\nEffective context (2048 tokens):")
    print(f"  English: ~{eng_eff} words")
    print(f"  Luganda: ~{lug_eff} words  ← this is the real context limit for Luganda")

    results = {
        "tokenizer": "CraneAILabs/ganda-gemma-1b",
        "luganda": lug_stats,
        "english": eng_stats,
        "luganda_english_ratio": round(ratio, 4),
        "effective_context_luganda_words": lug_eff,
        "effective_context_english_words": eng_eff,
    }

    os.makedirs("results/diagnostics", exist_ok=True)
    with open("results/diagnostics/tokenizer_fertility.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results/diagnostics/tokenizer_fertility.json")


if __name__ == "__main__":
    main()
