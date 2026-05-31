"""
Hour 1: Setup + Data Exploration
Run on your A100 instance after: pip install -r requirements.txt && huggingface-cli login
"""

from datasets import load_dataset
import collections
import json
import os

# --- Load datasets ---
fln = load_dataset("CraneAILabs/luganda-fln-training-data", "all")
exercises = load_dataset("CraneAILabs/luganda-bilingual-literacy-exercises")

print(f"FLN train: {len(fln['train'])} rows")
print(f"Exercises train: {len(exercises['train'])} rows")
print(f"\nFLN columns: {fln['train'].column_names}")
print(f"Exercise columns: {exercises['train'].column_names}")

print("\n--- FLN example ---")
print(fln['train'][0]['text'][:500])
print(f"\nFormat: {fln['train'][0]['format']}")
print(f"Correct letter: {fln['train'][0]['correct_letter']}")

# --- Position bias check (the core problem this project fixes) ---
mcq_items = [x for x in fln['train'] if x['correct_letter'] in ['A', 'B', 'C', 'D']]
position_dist = collections.Counter(x['correct_letter'] for x in mcq_items)
print(f"\n--- Position distribution (source of bias) ---")
for pos in ['A', 'B', 'C', 'D']:
    count = position_dist.get(pos, 0)
    print(f"  {pos}: {count} items ({count / len(mcq_items) * 100:.1f}%)")

# --- Exercises column inspection (schema unknown at plan-write time) ---
print("\n--- Exercises dataset structure ---")
print(f"Columns: {exercises['train'].column_names}")
print(f"First row keys/values (truncated):")
row0 = exercises['train'][0]
for k, v in row0.items():
    val = str(v)[:100]
    print(f"  {k}: {val}")
has_text_col = 'text' in exercises['train'].column_names
print(f"\nexercises dataset has 'text' column: {has_text_col}")
if not has_text_col:
    print("WARNING: exercises dataset lacks 'text' column.")
    print("You will need to add a formatting step in 03_sft_learner.py.")
    print("Inspect the columns above to write a format_exercise() function.")

# --- Benchmark exploration ---
benchmark = load_dataset("CraneAILabs/pedagogy-luganda-replaced")
print(f"\n--- Benchmark ---")
print(f"Columns: {benchmark['train'].column_names}")
print(f"Rows: {len(benchmark['train'])}")
print(f"Example question: {benchmark['train'][0]['luganda_question'][:200]}")
print(f"Correct answer: {benchmark['train'][0]['correct_answer']}")

cats = collections.Counter(r['category'] for r in benchmark['train'])
print(f"\nCategories: {dict(cats)}")

# --- Data contamination check ---
benchmark_texts = set(row['luganda_question'][:100] for row in benchmark['train'])
train_texts = set(row['text'][:100] for row in fln['train'])
overlap = benchmark_texts & train_texts
print(f"\n--- Contamination check ---")
print(f"Overlap between benchmark questions and training text prefixes: {len(overlap)} items")
if overlap:
    print("WARNING: potential contamination detected — investigate before reporting benchmark numbers.")
else:
    print("No contamination detected (prefix-level check).")

# --- Save exploration summary ---
summary = {
    "fln_rows": len(fln['train']),
    "exercises_rows": len(exercises['train']),
    "exercises_has_text_col": has_text_col,
    "exercises_columns": exercises['train'].column_names,
    "benchmark_rows": len(benchmark['train']),
    "position_distribution": dict(position_dist),
    "contamination_overlap": len(overlap),
}
os.makedirs("results", exist_ok=True)
with open("results/exploration_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nSaved exploration_summary.json")
