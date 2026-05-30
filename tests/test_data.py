# tests/test_data.py
import pytest
from collections import Counter
from scripts.core.data import (
    balance_by_position,
    extract_first_letter,
    permute_mcq_item,
    augment_mcq_with_permutations,
)


def _make_mcq_items(counts):
    """counts = {'A': n, 'B': n, ...}"""
    items = []
    for pos, n in counts.items():
        for i in range(n):
            items.append({"correct_letter": pos, "text": f"item_{pos}_{i}"})
    return items


def test_oversample_balances_to_max():
    items = _make_mcq_items({"A": 10, "B": 30, "C": 5, "D": 20})
    balanced = balance_by_position(items, strategy="oversample")
    counts = Counter(x["correct_letter"] for x in balanced)
    assert counts["A"] == counts["B"] == counts["C"] == counts["D"]
    assert counts["B"] == 30  # max is preserved


def test_undersample_balances_to_min():
    items = _make_mcq_items({"A": 10, "B": 30, "C": 5, "D": 20})
    balanced = balance_by_position(items, strategy="undersample")
    counts = Counter(x["correct_letter"] for x in balanced)
    assert counts["A"] == counts["B"] == counts["C"] == counts["D"]
    assert counts["A"] == 5  # min is preserved


def test_none_strategy_returns_original():
    items = _make_mcq_items({"A": 10, "B": 30, "C": 5, "D": 20})
    result = balance_by_position(items, strategy="none")
    assert len(result) == len(items)


def test_extract_first_letter_finds_abcd():
    assert extract_first_letter("A is correct") == "A"
    assert extract_first_letter("The answer is B.") == "B"
    assert extract_first_letter("  c  ") == "C"
    assert extract_first_letter("xyz") is None
    assert extract_first_letter("") is None


# --- Permutation augmentation tests ---

def _make_chat_item(correct_letter="B", opts=("opt1", "opt2", "opt3", "opt4")):
    """Build a minimal Gemma-chat-formatted MCQ item."""
    a, b, c, d = opts
    text = (
        f"<start_of_turn>user\nWhat is the answer?\n"
        f"(A) {a}\n(B) {b}\n(C) {c}\n(D) {d}\n<end_of_turn>\n"
        f"<start_of_turn>model\n{correct_letter}\n<end_of_turn>"
    )
    return {"text": text, "correct_letter": correct_letter}


def test_permute_mcq_correct_letter_updated():
    # perm (1,0,2,3): new_A=old_B, new_B=old_A → correct was B (idx 1), now at idx 0 = A
    item = _make_chat_item(correct_letter="B")
    result = permute_mcq_item(item, (1, 0, 2, 3))
    assert result is not None
    assert result["correct_letter"] == "A"


def test_permute_mcq_option_content_moves_correctly():
    item = _make_chat_item(correct_letter="A", opts=("alpha", "beta", "gamma", "delta"))
    # perm (3,0,1,2): new_A=old_D, new_B=old_A, new_C=old_B, new_D=old_C
    result = permute_mcq_item(item, (3, 0, 1, 2))
    assert result is not None
    assert "(A) delta" in result["text"]
    assert "(B) alpha" in result["text"]
    assert result["correct_letter"] == "B"  # old A (idx 0) → perm.index(0) = 1 → B


def test_permute_mcq_identity_perm_unchanged():
    item = _make_chat_item(correct_letter="C", opts=("alpha", "beta", "gamma", "delta"))
    result = permute_mcq_item(item, (0, 1, 2, 3))
    assert result is not None
    assert result["correct_letter"] == "C"
    # Options stay in same positions (text may differ in whitespace after rebuild)
    assert "(A) alpha" in result["text"]
    assert "(B) beta" in result["text"]
    assert "(C) gamma" in result["text"]
    assert "(D) delta" in result["text"]


def test_permute_mcq_unparseable_returns_none():
    item = {"text": "no options here", "correct_letter": "A"}
    assert permute_mcq_item(item, (1, 0, 2, 3)) is None


def test_augment_increases_dataset_size():
    items = [_make_chat_item("A"), _make_chat_item("B")]
    augmented = augment_mcq_with_permutations(items, n_perms_per_item=3)
    assert len(augmented) == 8  # 2 originals + 3 perms each


def test_augment_all_positions_represented():
    # After augmentation, all correct letters should appear
    items = [_make_chat_item("A")] * 10
    augmented = augment_mcq_with_permutations(items, n_perms_per_item=3)
    letters = Counter(x["correct_letter"] for x in augmented)
    assert len(letters) > 1  # augmentation produces varied positions
