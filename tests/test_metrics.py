"""Tests for hydrological performance metrics (spec §4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics import (
    DEFAULT_LOG_NSE_EPSILON_MM,
    compute_metrics,
    kge,
    kge_components,
    log_nse,
    mae,
    nse,
    rmse,
    volume_bias,
)


def test_perfect_simulation_metrics() -> None:
    obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sim = obs.copy()

    assert nse(obs, sim).value == pytest.approx(1.0)
    kge_val, r_val, alpha_val, beta_val = kge(obs, sim)
    assert kge_val.value == pytest.approx(1.0)
    assert r_val.value == pytest.approx(1.0)
    assert alpha_val.value == pytest.approx(1.0)
    assert beta_val.value == pytest.approx(1.0)
    assert log_nse(obs, sim).value == pytest.approx(1.0)
    assert volume_bias(obs, sim).value == pytest.approx(0.0)
    assert mae(obs, sim).value == pytest.approx(0.0)
    assert rmse(obs, sim).value == pytest.approx(0.0)


def test_biased_simulation() -> None:
    obs = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
    sim = obs * 1.5

    kge_val, r_val, alpha_val, beta_val = kge(obs, sim)
    assert r_val.value == pytest.approx(1.0)
    assert beta_val.value == pytest.approx(1.5)
    assert alpha_val.value == pytest.approx(1.5)
    assert kge_val.value < 1.0
    assert volume_bias(obs, sim).value == pytest.approx(0.5)


def test_constant_observed_series_nse_undefined() -> None:
    obs = np.array([3.0, 3.0, 3.0, 3.0])
    sim = np.array([1.0, 2.0, 3.0, 4.0])

    result = nse(obs, sim)
    assert not result.is_defined
    assert result.undefined_reason == "zero observed variance"
    assert result.n_valid == 4


def test_kge_components_known_synthetic_example() -> None:
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    sim = np.array([2.0, 4.0, 6.0, 8.0])

    r_val, alpha_val, beta_val = kge_components(obs, sim)
    assert r_val.value == pytest.approx(1.0)
    assert alpha_val.value == pytest.approx(2.0)
    assert beta_val.value == pytest.approx(2.0)


def test_log_nse_with_zero_flows_uses_epsilon() -> None:
    obs = np.array([0.0, 1.0, 2.0])
    sim = np.array([0.0, 1.0, 2.0])
    eps = DEFAULT_LOG_NSE_EPSILON_MM

    result = log_nse(obs, sim, epsilon_mm=eps)
    assert result.is_defined
    assert result.value == pytest.approx(1.0)

    # Convention: ln(Q + epsilon); zero observed becomes ln(epsilon)
    expected_log_obs = np.log(obs + eps)
    expected_log_sim = np.log(sim + eps)
    manual = 1.0 - np.sum((expected_log_obs - expected_log_sim) ** 2) / np.sum(
        (expected_log_obs - expected_log_obs.mean()) ** 2
    )
    assert result.value == pytest.approx(manual)


def test_log_nse_near_zero_flows() -> None:
    obs = np.array([0.001, 0.5, 1.0])
    sim = np.array([0.002, 0.4, 1.1])
    result = log_nse(obs, sim, epsilon_mm=0.01)
    assert result.is_defined
    assert np.isfinite(result.value)


def test_nans_in_observations_excluded() -> None:
    obs = pd.Series([1.0, np.nan, 3.0, 4.0])
    sim = pd.Series([1.0, 2.0, 3.0, 4.0])

    result = nse(obs, sim)
    assert result.n_valid == 3
    assert result.value == pytest.approx(1.0)


def test_nans_in_simulation_excluded() -> None:
    obs = pd.Series([1.0, 2.0, 3.0, 4.0])
    sim = pd.Series([1.0, np.nan, 3.0, 4.0])

    result = nse(obs, sim)
    assert result.n_valid == 3
    assert result.value == pytest.approx(1.0)


def test_period_mask_limits_evaluation() -> None:
    obs = np.array([1.0, 100.0, 3.0, 4.0])
    sim = np.array([1.0, 2.0, 3.0, 4.0])
    mask = np.array([False, True, True, True])

    result = nse(obs, sim, period_mask=mask)
    assert result.n_valid == 3
    assert result.value < 0.0


def test_compute_metrics_exposes_n_valid() -> None:
    index = pd.date_range("2011-01-01", periods=4, freq="D")
    obs = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)
    sim = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)
    mask = pd.Series([True, True, False, False], index=index)

    result = compute_metrics(obs, sim, period_mask=mask)
    assert result.nse.n_valid == 2
    assert result.kge.n_valid == 2
