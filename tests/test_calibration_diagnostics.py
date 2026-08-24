"""Tests for Phase 4B calibration diagnostics."""

from __future__ import annotations

import pandas as pd
import pytest

from src.calibration_diagnostics import (
    BOUND_PROXIMITY_FRACTION,
    delta_kge_summary,
    format_bound_proximity,
    parameter_space_summary,
    parameters_close_to_bounds,
    threshold_counts,
)


@pytest.fixture
def sample_bounds() -> dict[str, list[float]]:
    return {
        "X1": [100.0, 1200.0],
        "X2": [-5.0, 3.0],
        "X3": [20.0, 300.0],
        "X4": [1.1, 2.9],
    }


def test_parameters_close_to_lower_bound(sample_bounds: dict[str, list[float]]) -> None:
    span_x1 = sample_bounds["X1"][1] - sample_bounds["X1"][0]
    near_lower = {"x1": 100.0 + BOUND_PROXIMITY_FRACTION * span_x1, "x2": 0.0, "x3": 150.0, "x4": 2.0}
    close = parameters_close_to_bounds(near_lower, sample_bounds)
    assert close["X1"] == "lower"
    assert close["X2"] is None


def test_parameters_close_to_upper_bound(sample_bounds: dict[str, list[float]]) -> None:
    span_x4 = sample_bounds["X4"][1] - sample_bounds["X4"][0]
    near_upper = {"x1": 500.0, "x2": 0.0, "x3": 150.0, "x4": sample_bounds["X4"][1] - 0.01 * span_x4}
    close = parameters_close_to_bounds(near_upper, sample_bounds)
    assert close["X4"] == "upper"


def test_format_bound_proximity_none() -> None:
    close = {"X1": None, "X2": None, "X3": None, "X4": None}
    assert "none" in format_bound_proximity(close).lower()


def test_parameter_space_summary_subset() -> None:
    runs = pd.DataFrame(
        {
            "x1": [100.0, 200.0, 300.0, 400.0],
            "x2": [-5.0, -3.0, -1.0, 1.0],
            "x3": [20.0, 50.0, 80.0, 110.0],
            "x4": [1.1, 1.5, 1.9, 2.3],
            "kge_cal": [0.5, 0.75, 0.85, 0.65],
        }
    )
    summary = parameter_space_summary(runs, subset_label="gt_0.7", kge_threshold=0.7)
    assert int(summary.loc[summary["parameter"] == "X1", "n_runs"].iloc[0]) == 2
    assert summary.loc[summary["parameter"] == "X1", "min"].iloc[0] == pytest.approx(200.0)


def test_threshold_counts_include_075_and_085() -> None:
    runs = pd.DataFrame({"kge_cal": [0.55, 0.65, 0.72, 0.78, 0.82, 0.88]})
    counts = threshold_counts(runs)
    assert counts["kge_cal_gt_0.75"] == 3
    assert counts["kge_cal_gt_0.85"] == 1


def test_delta_kge_summary() -> None:
    runs = pd.DataFrame({"kge_cal": [0.8, 0.7, 0.9], "kge_val": [0.75, 0.72, 0.95]})
    summary = delta_kge_summary(runs)
    assert summary["worst"] == pytest.approx(-0.05)
    assert summary["best"] == pytest.approx(0.05)
    assert summary["median"] == pytest.approx(0.02)
