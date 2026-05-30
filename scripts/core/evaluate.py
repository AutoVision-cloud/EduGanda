# scripts/core/evaluate.py
import numpy as np
from scipy import stats
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import DatasetDict

ANSWER_TOKENS = ["A", "B", "C", "D"]


def evaluate_on_benchmark(model, tokenizer, benchmark_ds, label: str = "") -> dict:
    """
    Evaluates model on LLPK benchmark using logit-based MCQ scoring.
    Returns accuracy, predictions, labels, confidences, per-position/category stats, spread.
    """
    import torch
    from scripts.core.data import extract_first_letter

    samples = benchmark_ds["train"]
    correct = 0
    predictions, labels_list, confidences = [], [], []
    position_stats = {pos: {"total": 0, "correct": 0} for pos in ANSWER_TOKENS}
    category_stats = {}

    answer_token_ids = [tokenizer.convert_tokens_to_ids(t) for t in ANSWER_TOKENS]

    for item in samples:
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
            out = model(**inputs)
            next_logits = out.logits[0, -1, :]
            answer_logits = next_logits[answer_token_ids]
            probs = torch.softmax(answer_logits, dim=-1)
            pred_idx = int(probs.argmax())
            predicted = ANSWER_TOKENS[pred_idx]
            confidence = float(probs[pred_idx])

        gold = item["correct_answer"]
        predictions.append(predicted)
        labels_list.append(gold)
        confidences.append(confidence)
        position_stats[gold]["total"] += 1
        if predicted == gold:
            correct += 1
            position_stats[gold]["correct"] += 1

        cat = item.get("category", "unknown")
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "correct": 0}
        category_stats[cat]["total"] += 1
        if predicted == gold:
            category_stats[cat]["correct"] += 1

    total = len(samples)
    accs = [s["correct"] / s["total"] for s in position_stats.values() if s["total"] > 0]
    spread = (max(accs) - min(accs)) * 100 if accs else 0.0

    if label:
        print(f"[{label}] accuracy={correct/total:.1%}  spread={spread:.1f}pp")

    return {
        "accuracy": correct / total,
        "predictions": predictions,
        "labels": labels_list,
        "confidences": confidences,
        "position_stats": position_stats,
        "category_stats": category_stats,
        "spread": spread,
    }


def bootstrap_ci(
    predictions: List[str],
    labels: List[str],
    n_resamples: int = 1000,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """Returns (accuracy, lower_ci, upper_ci) using percentile bootstrap."""
    correct = np.array([p == l for p, l in zip(predictions, labels)], dtype=float)
    rng = np.random.default_rng(42)
    boot_accs = [
        rng.choice(correct, size=len(correct), replace=True).mean()
        for _ in range(n_resamples)
    ]
    alpha = (1 - confidence) / 2
    lo = float(np.percentile(boot_accs, alpha * 100))
    hi = float(np.percentile(boot_accs, (1 - alpha) * 100))
    return float(correct.mean()), lo, hi


def mcnemar_test(
    preds_a: List[str],
    preds_b: List[str],
    labels: List[str],
) -> float:
    """McNemar's test (continuity correction). Returns p-value."""
    n01 = sum(pa != l and pb == l for pa, pb, l in zip(preds_a, preds_b, labels))
    n10 = sum(pa == l and pb != l for pa, pb, l in zip(preds_a, preds_b, labels))
    if n01 + n10 == 0:
        return 1.0
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    return float(stats.chi2.sf(stat, df=1))


def compute_calibration(
    confidences: List[float],
    correct: List[bool],
    n_bins: int = 10,
) -> List[dict]:
    """Bins predictions by confidence and computes accuracy per bin."""
    bins = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        # Last bin is inclusive on both ends to capture confidence == 1.0
        mask = [lo <= c <= hi if i == n_bins - 1 else lo <= c < hi for c in confidences]
        count = sum(mask)
        acc = sum(c for c, m in zip(correct, mask) if m) / count if count > 0 else 0.0
        bins.append({"bin_center": round((lo + hi) / 2, 2), "accuracy": round(acc, 4), "count": count})
    return bins
