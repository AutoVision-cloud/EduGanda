"""
Semantic contamination check: TF-IDF cosine similarity between benchmark
questions and training examples.

The original model card claims "verified zero data contamination" with no
methodology. This script provides a documented, reproducible check using
both n-gram overlap (prefix) and TF-IDF semantic similarity.

Usage:
  python scripts/diagnostics/semantic_contamination.py

No GPU required. Runs locally once datasets are available.
"""
import json
import os
from typing import List, Dict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def find_overlaps(
    benchmark_texts: List[str],
    training_texts: List[str],
    threshold: float = 0.7,
) -> List[Dict]:
    """
    Finds (benchmark, training) pairs with TF-IDF cosine similarity >= threshold.
    Returns list sorted by similarity descending.
    """
    if not benchmark_texts or not training_texts:
        return []

    all_texts = benchmark_texts + training_texts
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
    tfidf = vectorizer.fit_transform(all_texts)

    bench_mat = tfidf[: len(benchmark_texts)]
    train_mat = tfidf[len(benchmark_texts) :]

    sim_matrix = cosine_similarity(bench_mat, train_mat)

    overlaps = []
    bench_idx, train_idx = np.where(sim_matrix >= threshold)
    for bi, ti in zip(bench_idx, train_idx):
        overlaps.append({
            "benchmark_idx": int(bi),
            "train_idx": int(ti),
            "similarity": float(sim_matrix[bi, ti]),
            "benchmark_text": benchmark_texts[bi][:150],
            "train_text": training_texts[ti][:150],
        })

    return sorted(overlaps, key=lambda x: -x["similarity"])


def prefix_overlaps(benchmark_texts: List[str], training_texts: List[str], prefix_len: int = 100) -> int:
    """Original prefix-based check from 01_explore_data.py — reproduced here for comparison."""
    bench_set = {t[:prefix_len] for t in benchmark_texts}
    train_set = {t[:prefix_len] for t in training_texts}
    return len(bench_set & train_set)


def main():
    from datasets import load_dataset

    os.makedirs("results/diagnostics", exist_ok=True)

    print("Loading benchmark and training data...")
    benchmark = load_dataset("CraneAILabs/pedagogy-luganda-replaced")["train"]
    fln = load_dataset("CraneAILabs/luganda-fln-training-data", "all")["train"]
    exercises = load_dataset("CraneAILabs/luganda-bilingual-literacy-exercises")["train"]

    # Benchmark: use both Luganda and English questions
    bench_lug = [r["luganda_question"] for r in benchmark]
    bench_eng = [r["english_question"] for r in benchmark]
    bench_all = bench_lug + bench_eng

    # Training: extract user turns from FLN + exercises
    def extract_user_turn(text: str) -> str:
        if "<start_of_turn>user\n" in text and "<end_of_turn>" in text:
            return text.split("<start_of_turn>user\n")[1].split("<end_of_turn>")[0].strip()
        return text.strip()

    train_texts = [extract_user_turn(r["text"]) for r in fln if r.get("text")]
    if "text" in exercises.column_names:
        train_texts += [extract_user_turn(r["text"]) for r in exercises if r.get("text")]

    print(f"Benchmark: {len(bench_all)} items | Training: {len(train_texts)} items")

    # --- Prefix check (original method) ---
    prefix_hits = prefix_overlaps(bench_lug, [r["text"][:100] for r in fln if r.get("text")])
    print(f"\nPrefix overlap (first 100 chars): {prefix_hits} items")

    # --- TF-IDF semantic check ---
    print("Running TF-IDF semantic similarity check (threshold=0.7)...")
    overlaps_07 = find_overlaps(bench_all, train_texts, threshold=0.7)

    print("Running TF-IDF semantic similarity check (threshold=0.5)...")
    overlaps_05 = find_overlaps(bench_all, train_texts, threshold=0.5)

    print(f"\n--- Contamination Report ---")
    print(f"Prefix overlap (100-char):         {prefix_hits} pairs")
    print(f"Semantic similarity >= 0.7:        {len(overlaps_07)} pairs")
    print(f"Semantic similarity >= 0.5:        {len(overlaps_05)} pairs")

    if overlaps_07:
        print("\nTop suspicious pairs (similarity >= 0.7):")
        for o in overlaps_07[:5]:
            print(f"  sim={o['similarity']:.3f}")
            print(f"  BENCH: {o['benchmark_text'][:80]}")
            print(f"  TRAIN: {o['train_text'][:80]}")
            print()
    else:
        print("\nNo pairs found above 0.7 threshold — contamination unlikely.")

    results = {
        "methodology": "TF-IDF cosine similarity with unigram+bigram features",
        "benchmark_size": len(bench_all),
        "training_size": len(train_texts),
        "prefix_overlap_100char": prefix_hits,
        "semantic_overlaps_threshold_0.7": len(overlaps_07),
        "semantic_overlaps_threshold_0.5": len(overlaps_05),
        "top_overlaps": overlaps_07[:10],
        "verdict": "clean" if len(overlaps_07) == 0 else "suspicious — review top_overlaps",
    }

    with open("results/diagnostics/semantic_contamination.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results/diagnostics/semantic_contamination.json")


if __name__ == "__main__":
    main()
