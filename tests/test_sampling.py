"""Tests for Latin Hypercube parameter sampling (Phase 4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.sampling import latin_hypercube_sample, sample_parameters


@pytest.fixture
def bounds() -> dict[str, list[float]]:
    return {
        "X1": [100.0, 1200.0],
        "X2": [-5.0, 3.0],
        "X3": [20.0, 300.0],
        "X4": [1.1, 2.9],
    }


def test_lhs_deterministic_with_fixed_seed(bounds: dict[str, list[float]]) -> None:
    a = latin_hypercube_sample(bounds, n_samples=50, seed=42)
    b = latin_hypercube_sample(bounds, n_samples=50, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_lhs_samples_within_bounds(bounds: dict[str, list[float]]) -> None:
    samples = latin_hypercube_sample(bounds, n_samples=200, seed=7)
    for name, (lo, hi) in bounds.items():
        assert samples[name].min() >= lo
        assert samples[name].max() <= hi


def test_lhs_correct_number_of_samples(bounds: dict[str, list[float]]) -> None:
    samples = latin_hypercube_sample(bounds, n_samples=1000, seed=1)
    assert len(samples) == 1000


def test_sample_parameters_reads_config(basin_config: dict) -> None:
    basin_config["calibration"]["n_samples"] = 25
    samples = sample_parameters(basin_config)
    assert len(samples) == 25
    assert list(samples.columns) == ["X1", "X2", "X3", "X4"]
