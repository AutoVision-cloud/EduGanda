import numpy as np
import pytest
from scripts.diagnostics.probe_position_bias import fit_probe, POSITIONS


def test_probe_on_perfectly_separable_data():
    rng = np.random.default_rng(0)
    n_per_class = 50
    hidden_dim = 32
    hidden_states = np.vstack([
        rng.normal(loc=i * 10, scale=0.1, size=(n_per_class, hidden_dim))
        for i in range(4)
    ])
    labels = [p for p in POSITIONS for _ in range(n_per_class)]
    cv_acc = fit_probe(hidden_states, labels)
    assert cv_acc > 0.95, f"Expected near-perfect accuracy on separable data, got {cv_acc:.2f}"


def test_probe_on_random_data_near_chance():
    rng = np.random.default_rng(1)
    n = 200
    hidden_states = rng.normal(size=(n, 64))
    labels = [POSITIONS[i % 4] for i in range(n)]
    cv_acc = fit_probe(hidden_states, labels)
    assert cv_acc < 0.40, f"Expected near-chance accuracy on random data, got {cv_acc:.2f}"


def test_fit_probe_returns_float():
    rng = np.random.default_rng(2)
    hs = rng.normal(size=(40, 16))
    labels = [POSITIONS[i % 4] for i in range(40)]
    result = fit_probe(hs, labels)
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0
