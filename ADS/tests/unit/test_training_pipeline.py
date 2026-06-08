"""Test training pipeline core logic."""

from __future__ import annotations

import types
import sys
import unittest.mock

# Mock torchaudio before importing training module
torchaudio_mock = types.ModuleType("torchaudio")
sys.modules["torchaudio"] = torchaudio_mock
sys.modules["torchaudio.functional"] = types.ModuleType("torchaudio.functional")


import numpy as np
from ads.pipeline.training import compute_metrics


def test_compute_metrics_perfect():
    probs = np.array([0.99, 0.01, 0.99, 0.01])
    labels = np.array([1, 0, 1, 0])
    m = compute_metrics(probs, labels)
    assert m["accuracy"] == 1.0
    assert m["f1"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["auc"] >= 0.99
    print(f"  eer={m['eer']:.4f} auc={m['auc']:.4f} f1={m['f1']:.4f}")


def test_compute_metrics_all_wrong():
    probs = np.array([0.01, 0.99, 0.01, 0.99])
    labels = np.array([1, 0, 1, 0])
    m = compute_metrics(probs, labels)
    assert m["accuracy"] == 0.0
    print(f"  eer={m['eer']:.4f} auc={m['auc']:.4f} f1={m['f1']:.4f}")


def test_compute_metrics_imbalanced():
    rng = np.random.default_rng(42)
    probs = rng.uniform(0, 1, 100)
    labels = (probs > 0.7).astype(int)
    m = compute_metrics(probs, labels)
    assert 0 <= m["eer"] <= 1
    assert 0 <= m["auc"] <= 1
    assert 0 <= m["f1"] <= 1
    print(f"  eer={m['eer']:.4f} auc={m['auc']:.4f} f1={m['f1']:.4f}")


if __name__ == "__main__":
    test_compute_metrics_perfect()
    test_compute_metrics_all_wrong()
    test_compute_metrics_imbalanced()
    print("ALL training pipeline tests passed")
