# scripts/core/evaluate.py
import math
import numpy as np
from collections import Counter
from scipy import stats
from typing import List, Tuple, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from datasets import DatasetDict

ANSWER_TOKENS = ["A", "B", "C", "D"]


def check_tokenization(tokenizer):
    """
    Prints how A/B/C/D are tokenized with and without a leading space.
    Run once before evaluating to verify you are scoring the intended tokens.
    """
    print("Tokenization check:")
    for c in ["A", "B", "C", "D", " A", " B", " C", " D"]:
        ids = tokenizer.encode(c, add_special_tokens=False)
        print(f"  {repr(c):6s} → {ids}")


def score_choice(model, tokenizer, prompt: str, choice: str) -> float:
    """
    Computes sum of log-probabilities for `choice` tokens continuing `prompt`.
    Handles multi-token choices correctly. For single-token choices, prefer
    score_all_choices() which does this in one forward pass instead of four.
    """
    import torch

    full_text = prompt + choice
    prompt_ids = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(model.device)
    full_ids = tokenizer(
        full_text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(model.device)

    n_choice_tokens = full_ids.shape[1] - prompt_ids.shape[1]
    if n_choice_tokens <= 0:
        return float("-inf")

    with torch.no_grad():
        outputs = model(full_ids)
        logits = outputs.logits  # (1, seq_len, vocab)

    log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = full_ids[:, 1:]

    choice_start = prompt_ids.shape[1] - 1
    total = sum(
        log_probs[0, choice_start + i, target_ids[0, choice_start + i]].item()
        for i in range(n_choice_tokens)
    )
    return total


def score_all_choices(model, tokenizer, prompt: str, choices: list = ANSWER_TOKENS) -> dict:
    """
    Scores all choices in ONE forward pass on the prompt — 4x faster than
    calling score_choice() separately for each option.

    Mathematically equivalent to score_choice() for single-token choices:
    transformer attention at position t depends only on tokens 0..t, so
    the logit for the choice token is identical whether we feed prompt alone
    or prompt+choice. SentencePiece context is handled by tokenizing
    prompt+choice to get the correct in-context token ID, then looking it
    up in the prompt-only forward pass.

    Falls back to score_choice() for any choice that tokenizes to >1 token.
    """
    import torch

    # Get the in-context token ID for each choice (handles space-prefix correctly)
    prompt_len = len(tokenizer(prompt, add_special_tokens=False).input_ids)
    choice_token_ids = {}
    multi_token_choices = []
    for choice in choices:
        full_ids = tokenizer(prompt + choice, add_special_tokens=False).input_ids
        n_choice = len(full_ids) - prompt_len
        if n_choice == 1:
            choice_token_ids[choice] = full_ids[-1]
        else:
            multi_token_choices.append(choice)

    # Single forward pass on the prompt
    prompt_ids = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(model.device)

    with torch.no_grad():
        out = model(prompt_ids)
        last_logits = out.logits[0, -1, :]  # (vocab_size,)

    log_probs = torch.log_softmax(last_logits, dim=-1)

    scores = {
        choice: log_probs[tok_id].item()
        for choice, tok_id in choice_token_ids.items()
    }

    # Fall back for any multi-token choices
    for choice in multi_token_choices:
        scores[choice] = score_choice(model, tokenizer, prompt, choice)

    return scores


def _get_choice_token_ids(tokenizer, prompt_suffix: str) -> dict:
    """
    Pre-computes the in-context token ID for each answer letter.
    Uses the prompt suffix (the part after the question) to get the correct
    SentencePiece token in context. Called once before the eval loop.
    """
    ids = {}
    suffix_len = len(tokenizer(prompt_suffix, add_special_tokens=False).input_ids)
    for choice in ANSWER_TOKENS:
        full = tokenizer(prompt_suffix + choice, add_special_tokens=False).input_ids
        n_choice = len(full) - suffix_len
        if n_choice == 1:
            ids[choice] = full[-1]
        else:
            ids[choice] = None  # multi-token, will fall back to score_choice
    return ids


def evaluate_on_benchmark(model, tokenizer, benchmark_ds, label: str = "") -> dict:
    """
    Primary metric: log-prob option scoring over A/B/C/D continuations.

    Optimised: pre-computes choice token IDs once, then does exactly
    ONE tokenizer call + ONE forward pass per benchmark item.
    """
    import torch

    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pre-compute choice token IDs using the fixed prompt suffix.
    # All prompts end with this string, so the token IDs are stable.
    PROMPT_SUFFIX = "<end_of_turn>\n<start_of_turn>model\n"
    choice_token_ids = _get_choice_token_ids(tokenizer, PROMPT_SUFFIX)

    samples = benchmark_ds["train"]
    correct = 0
    predictions, labels_list, confidences = [], [], []
    position_stats = {pos: {"total": 0, "correct": 0} for pos in ANSWER_TOKENS}
    category_stats = {}
    subdomain_stats = {}
    age_group_stats = {}

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

        # ONE tokenizer call + ONE forward pass per item
        prompt_ids = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(model.device)

        with torch.no_grad():
            last_logits = model(prompt_ids).logits[0, -1, :]

        log_probs = torch.log_softmax(last_logits, dim=-1)
        log_prob_scores = {}
        for choice in ANSWER_TOKENS:
            tok_id = choice_token_ids.get(choice)
            if tok_id is not None:
                log_prob_scores[choice] = log_probs[tok_id].item()
            else:
                log_prob_scores[choice] = score_choice(model, tokenizer, prompt, choice)
        predicted = max(log_prob_scores, key=log_prob_scores.get)

        # Confidence = softmax over 4 log-prob scores
        scores_tensor = torch.tensor(list(log_prob_scores.values()))
        probs = torch.softmax(scores_tensor, dim=0)
        confidence = float(probs[ANSWER_TOKENS.index(predicted)])

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

        for field, store in [("subdomain", subdomain_stats), ("age_group", age_group_stats)]:
            key = item.get(field, "unknown") or "unknown"
            if key not in store:
                store[key] = {"total": 0, "correct": 0}
            store[key]["total"] += 1
            if predicted == gold:
                store[key]["correct"] += 1

    total = len(samples)
    accs = [s["correct"] / s["total"] for s in position_stats.values() if s["total"] > 0]
    spread = (max(accs) - min(accs)) * 100 if accs else 0.0

    pred_dist = compute_prediction_distribution(predictions)
    pred_entropy = compute_prediction_entropy(predictions)

    if label:
        print(
            f"[{label}] accuracy={correct/total:.1%}  spread={spread:.1f}pp  "
            f"pred_entropy={pred_entropy:.3f}"
        )

    return {
        "accuracy": correct / total,
        "predictions": predictions,
        "labels": labels_list,
        "confidences": confidences,
        "position_stats": position_stats,
        "category_stats": category_stats,
        "subdomain_stats": subdomain_stats,
        "age_group_stats": age_group_stats,
        "spread": spread,
        "prediction_distribution": pred_dist,
        "prediction_entropy": pred_entropy,
    }


def evaluate_on_benchmark_generation(model, tokenizer, benchmark_ds, label: str = "") -> dict:
    """
    Secondary metric: greedy generation scoring.

    Approximates deployment behaviour — how the model actually responds when
    asked a question. Uses repetition_penalty=1.2 as required by EduGanda.
    Results may differ from log-prob scoring due to output formatting and
    parser sensitivity. Report alongside primary metric for completeness.
    """
    from scripts.core.data import extract_first_letter

    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    import torch

    samples = benchmark_ds["train"]
    correct = 0
    predictions, labels_list = [], []

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
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()

        predicted = extract_first_letter(response)
        gold = item["correct_answer"]
        predictions.append(predicted or "")
        labels_list.append(gold)
        if predicted == gold:
            correct += 1

    total = len(samples)
    accuracy = correct / total
    if label:
        print(f"[{label} generation] accuracy={accuracy:.1%}")

    return {"accuracy": accuracy, "predictions": predictions, "labels": labels_list}


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


def compute_prediction_distribution(predictions: List[str]) -> Dict[str, float]:
    """
    Returns how often each letter (A/B/C/D) is predicted regardless of gold label.
    Separates model positional bias from benchmark label distribution.
    """
    total = len(predictions)
    counts = Counter(predictions)
    return {l: round(counts.get(l, 0) / total, 4) for l in ["A", "B", "C", "D"]}


def compute_prediction_entropy(predictions: List[str]) -> float:
    """
    Shannon entropy (bits) of the prediction distribution.
    Max = 2.0 bits (uniform over 4 options). Low entropy = position-biased model.
    """
    total = len(predictions)
    counts = Counter(predictions)
    probs = [counts[l] / total for l in ["A", "B", "C", "D"] if counts.get(l, 0) > 0]
    return round(-sum(p * math.log2(p) for p in probs), 4)
