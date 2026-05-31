"""
Trains a logistic regression probe on model hidden states to detect
whether position bias is encoded in representations.

Usage:
  python scripts/diagnostics/probe_position_bias.py
"""
import json
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple

POSITIONS = ["A", "B", "C", "D"]


def fit_probe(hidden_states: np.ndarray, labels: List[str], cv: int = 5) -> float:
    """Fits logistic regression on hidden_states -> labels. Returns mean CV accuracy."""
    scaler = StandardScaler()
    X = scaler.fit_transform(hidden_states)
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    scores = cross_val_score(clf, X, labels, cv=cv, scoring="accuracy")
    return float(scores.mean())


def extract_hidden_states(model, tokenizer, benchmark_ds) -> Tuple[np.ndarray, List[str]]:
    """Returns (hidden_states, labels) — last-layer last-token hidden state per item."""
    import torch

    model.eval()
    all_hidden, all_labels = [], []

    for item in benchmark_ds["train"]:
        prompt = (
            f"<start_of_turn>user\n"
            f"Answer with only the letter (A, B, C, or D). Do not explain.\n\n"
            f"{item['luganda_question']}\n"
            f"(A) {item['luganda_answer_a']}\n"
            f"(B) {item['luganda_answer_b']}\n"
            f"(C) {item['luganda_answer_c']}\n"
            f"(D) {item['luganda_answer_d']}\n"
            f"<end_of_turn>\n<start_of_turn>model\n"
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1][0, -1, :].float().cpu().numpy()
        all_hidden.append(last_hidden)
        all_labels.append(item["correct_answer"])

    return np.stack(all_hidden), all_labels


def probe_model(model_path: str, benchmark_ds, label: str) -> dict:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\n[Probe] Loading {label} from {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
    tok = AutoTokenizer.from_pretrained(model_path)

    hidden_states, labels = extract_hidden_states(model, tok, benchmark_ds)
    del model
    torch.cuda.empty_cache()

    cv_acc = fit_probe(hidden_states, labels)
    print(f"  Probe CV accuracy: {cv_acc:.1%}  (chance=25%)")
    return {"model": label, "probe_cv_accuracy": cv_acc, "n_items": len(labels)}


def main():
    from datasets import load_dataset

    benchmark = load_dataset("CraneAILabs/pedagogy-luganda-replaced")

    checkpoints = {
        "ganda-gemma-1b (base)": "CraneAILabs/ganda-gemma-1b",
        "LEARNER": "./learner-full",
        "EduGanda reference": "CraneAILabs/EduGanda-Gemma-3-1B",
    }

    summary_path = "results/ablations/summary.json"
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        best = summary.get("best", {})
        axis, value = best.get("axis"), best.get("value")
        if axis and value:
            ckpt = f"./ablation-{axis}-{value}-sft"
            if os.path.isdir(ckpt):
                checkpoints[f"Best ablation ({axis}={value})"] = ckpt

    results = []
    for label, path in checkpoints.items():
        if path.startswith("./") and not os.path.isdir(path):
            print(f"Skipping {label} — checkpoint not found at {path}")
            continue
        results.append(probe_model(path, benchmark, label))

    os.makedirs("results/diagnostics", exist_ok=True)
    with open("results/diagnostics/position_bias_probe.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n--- Probe Summary ---")
    print(f"{'Model':<35} {'CV Accuracy':>12}  Interpretation")
    print("-" * 70)
    for r in results:
        acc = r["probe_cv_accuracy"]
        interp = "strong encoding" if acc > 0.40 else ("mild encoding" if acc > 0.30 else "near chance")
        print(f"{r['model']:<35} {acc:>11.1%}  {interp}")

    print("\nSaved results/diagnostics/position_bias_probe.json")


if __name__ == "__main__":
    main()
