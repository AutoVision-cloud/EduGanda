# tests/test_data.py
import pytest
from collections import Counter
from scripts.core.data import balance_by_position, extract_first_letter


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
