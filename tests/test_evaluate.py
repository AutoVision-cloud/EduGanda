import numpy as np
import pytest
from scripts.core.evaluate import (
    bootstrap_ci,
    mcnemar_test,
    compute_calibration,
    compute_prediction_distribution,
    compute_prediction_entropy,
)


def test_bootstrap_ci_perfect_accuracy():
    preds = ["A", "B", "C", "D"] * 25
    labels = ["A", "B", "C", "D"] * 25
    acc, lo, hi = bootstrap_ci(preds, labels)
    assert acc == pytest.approx(1.0)
    assert lo == pytest.approx(1.0)
    assert hi == pytest.approx(1.0)


def test_bootstrap_ci_half_accuracy():
    preds  = ["A"] * 50 + ["B"] * 50
    labels = ["A"] * 50 + ["A"] * 50
    acc, lo, hi = bootstrap_ci(preds, labels)
    assert acc == pytest.approx(0.5)
    assert 0.39 < lo < 0.5
    assert 0.5 < hi <= 0.6


def test_bootstrap_ci_returns_ordered_bounds():
    preds  = ["A"] * 60 + ["B"] * 40
    labels = ["A"] * 100
    acc, lo, hi = bootstrap_ci(preds, labels)
    assert lo <= acc <= hi


def test_mcnemar_identical_models_returns_high_pvalue():
    preds = ["A", "B", "C", "D"] * 25
    labels = ["A", "B", "A", "D"] * 25
    p = mcnemar_test(preds, preds, labels)
    assert p == pytest.approx(1.0)


def test_mcnemar_very_different_models_returns_low_pvalue():
    n = 100
    labels = ["A"] * n
    preds_a = ["A"] * n
    preds_b = ["B"] * n
    p = mcnemar_test(preds_a, preds_b, labels)
    assert p < 0.001


def test_compute_calibration_perfect_model():
    confidences = [1.0] * 100
    correct = [True] * 100
    bins = compute_calibration(confidences, correct, n_bins=5)
    assert any(b["accuracy"] == pytest.approx(1.0) for b in bins if b["count"] > 0)


def test_prediction_distribution_sums_to_one():
    preds = ["A"] * 50 + ["B"] * 30 + ["C"] * 15 + ["D"] * 5
    dist = compute_prediction_distribution(preds)
    assert set(dist.keys()) == {"A", "B", "C", "D"}
    assert abs(sum(dist.values()) - 1.0) < 0.001


def test_prediction_distribution_biased_model():
    preds = ["B"] * 100
    dist = compute_prediction_distribution(preds)
    assert dist["B"] == pytest.approx(1.0)
    assert dist["A"] == 0.0


def test_prediction_entropy_uniform_is_max():
    preds = ["A"] * 25 + ["B"] * 25 + ["C"] * 25 + ["D"] * 25
    entropy = compute_prediction_entropy(preds)
    assert entropy == pytest.approx(2.0, abs=0.01)  # log2(4) = 2 bits


def test_prediction_entropy_biased_is_low():
    preds = ["B"] * 100
    entropy = compute_prediction_entropy(preds)
    assert entropy == pytest.approx(0.0)


def test_compute_calibration_returns_expected_structure():
    import random
    rng = random.Random(0)
    confidences = [rng.uniform(0.25, 1.0) for _ in range(200)]
    correct = [rng.random() < c for c in confidences]
    bins = compute_calibration(confidences, correct, n_bins=10)
    assert len(bins) == 10
    for b in bins:
        assert "bin_center" in b and "accuracy" in b and "count" in b
