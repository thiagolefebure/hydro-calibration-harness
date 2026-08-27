"""Tests for Phase 8D multi-horizon residual forecasting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ml_horizon_forecast import (
    COMPARISON_COLUMNS,
    EXPERIMENT_NOTE,
    EXPERIMENT_TAG,
    FORECAST_FEATURE_COLUMNS,
    FORBIDDEN_FUTURE_COLUMNS,
    HORIZONS,
    MODEL_PERSISTENCE,
    MODEL_PHYSICAL,
    build_forecast_origin_dataset,
    filter_horizon_rows,
    fit_ar1_on_calibration_residuals,
    fit_ridge_for_horizon,
    predict_all_models,
    run_horizon_forecast_analysis,
    run_ml_horizon_export,
)
from src.ml_residual_baselines import split_calibration_validation
from src.ml_residual_dataset import FEATURE_COLUMNS, TARGET_COLUMN


def _synthetic_daily(n_cal: int = 100, n_val: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = n_cal + n_val
    dates = pd.date_range("2011-01-01", periods=n, freq="D")
    q_phys = rng.uniform(0.3, 2.0, n)
    residual = np.zeros(n)
    innov = rng.normal(0.0, 0.08, n)
    for t in range(1, n):
        residual[t] = 0.75 * residual[t - 1] + innov[t]
    q_obs = np.maximum(q_phys + residual, 0.02)
    frame: dict[str, object] = {
        "date": dates.strftime("%Y-%m-%d"),
        "period": ["calibration"] * n_cal + ["validation"] * n_val,
        "q_obs": q_obs,
        "q_phys": q_phys,
        TARGET_COLUMN: residual,
        "q_phys_change_1d": pd.Series(q_phys).diff().fillna(0.0).to_numpy(),
        "precipitation_1d": rng.uniform(0.0, 5.0, n),
        "precipitation_3d": rng.uniform(0.0, 12.0, n),
        "precipitation_7d": rng.uniform(0.0, 25.0, n),
        "precipitation_30d": rng.uniform(0.0, 80.0, n),
        "production_store": rng.uniform(20.0, 200.0, n),
        "routing_store": rng.uniform(10.0, 80.0, n),
        "day_of_year_sin": np.sin(2 * np.pi * dates.dayofyear / 365.25),
        "day_of_year_cos": np.cos(2 * np.pi * dates.dayofyear / 365.25),
    }
    for col in FEATURE_COLUMNS:
        if col not in frame:
            frame[col] = rng.normal(0.0, 1.0, n)
    return pd.DataFrame(frame)


def test_targets_shifted_forward_correctly() -> None:
    daily = _synthetic_daily()
    forecast = build_forecast_origin_dataset(daily)
    for h in HORIZONS:
        expected = daily[TARGET_COLUMN].to_numpy()[h:]
        actual = forecast[f"residual_h{h}"].to_numpy()[:-h]
        np.testing.assert_allclose(actual, expected)
        # Target at origin i equals residual at i+h
        for i in range(len(daily) - h):
            assert forecast.loc[i, f"residual_h{h}"] == pytest.approx(
                daily.loc[i + h, TARGET_COLUMN]
            )


def test_no_predictor_contains_future_observations() -> None:
    assert not set(FORECAST_FEATURE_COLUMNS) & FORBIDDEN_FUTURE_COLUMNS
    assert "q_obs_h1" not in FORECAST_FEATURE_COLUMNS
    assert "residual_h1" not in FORECAST_FEATURE_COLUMNS
    daily = _synthetic_daily()
    forecast = build_forecast_origin_dataset(daily)
    # Mutating future residual columns must not change predictor columns.
    poisoned = forecast.copy()
    for h in HORIZONS:
        poisoned[f"residual_h{h}"] = 999.0
        poisoned[f"q_obs_h{h}"] = 999.0
    for col in FORECAST_FEATURE_COLUMNS:
        pd.testing.assert_series_equal(forecast[col], poisoned[col], check_names=False)


def test_persistence_uses_residual_t_only() -> None:
    daily = _synthetic_daily()
    forecast = build_forecast_origin_dataset(daily)
    val = filter_horizon_rows(forecast, origin_period="validation", horizon=1)
    cal = filter_horizon_rows(forecast, origin_period="calibration", horizon=1)
    ar1 = fit_ar1_on_calibration_residuals(split_calibration_validation(daily)[0])
    ridge = fit_ridge_for_horizon(cal, 1)
    preds = predict_all_models(val, horizon=1, ar1=ar1, ridge=ridge)
    expected = np.maximum(
        val["q_phys_h1"].to_numpy() + val["residual_t"].to_numpy(),
        0.0,
    )
    np.testing.assert_allclose(preds[MODEL_PERSISTENCE], expected)


def test_ar_recursion_uses_calibration_phi_only() -> None:
    daily = _synthetic_daily()
    cal_daily, _ = split_calibration_validation(daily)
    ar1 = fit_ar1_on_calibration_residuals(cal_daily)
    assert ar1.fitted_on == "calibration"

    poisoned = daily.copy()
    poisoned.loc[poisoned["period"] == "validation", TARGET_COLUMN] = 1.0e6
    ar1_p = fit_ar1_on_calibration_residuals(split_calibration_validation(poisoned)[0])
    assert ar1.phi == pytest.approx(ar1_p.phi)
    assert ar1.intercept == pytest.approx(ar1_p.intercept)

    e0 = np.array([1.0, -0.5])
    e1 = ar1.forecast_from_origin(e0, 1)
    e2 = ar1.forecast_from_origin(e0, 2)
    np.testing.assert_allclose(e1, ar1.intercept + ar1.phi * e0)
    np.testing.assert_allclose(e2, ar1.intercept + ar1.phi * e1)


def test_one_ridge_model_per_horizon() -> None:
    daily = _synthetic_daily()
    forecast = build_forecast_origin_dataset(daily)
    ridges = {}
    for h in HORIZONS:
        cal_h = filter_horizon_rows(forecast, origin_period="calibration", horizon=h)
        ridges[h] = fit_ridge_for_horizon(cal_h, h)
        assert ridges[h].horizon == h
        assert ridges[h].feature_names == list(FORECAST_FEATURE_COLUMNS)
    # Distinct fitted objects (one per horizon).
    assert len({id(r) for r in ridges.values()}) == 3


def test_validation_never_used_for_fitting() -> None:
    daily = _synthetic_daily()
    forecast = build_forecast_origin_dataset(daily)
    cal_h = filter_horizon_rows(forecast, origin_period="calibration", horizon=2)
    ridge = fit_ridge_for_horizon(cal_h, 2)
    # Scaler mean from calibration features only.
    assert ridge.scaler.mean_[0] == pytest.approx(cal_h["residual_t"].mean())

    poisoned = forecast.copy()
    poisoned.loc[poisoned["period"] == "validation", "residual_t"] = 1.0e6
    cal_p = filter_horizon_rows(poisoned, origin_period="calibration", horizon=2)
    ridge_p = fit_ridge_for_horizon(cal_p, 2)
    assert ridge.scaler.mean_[0] == pytest.approx(ridge_p.scaler.mean_[0])


def test_physical_baseline_tagged_oracle_weather() -> None:
    assert EXPERIMENT_TAG == "ORACLE METEOROLOGICAL FORCING"
    assert "NOT operational" in EXPERIMENT_NOTE
    daily = _synthetic_daily()
    comparison, _hf, _deg, _fc, answers = run_horizon_forecast_analysis(
        daily, epsilon_mm=0.01
    )
    assert list(comparison.columns) == COMPARISON_COLUMNS
    assert MODEL_PHYSICAL in set(comparison["model"])
    assert answers["experiment_tag"] == EXPERIMENT_TAG
    assert "ORACLE" in answers["7_meteo_vs_residual"]


def test_export_writes_horizon_comparison(tmp_path: Path) -> None:
    daily = _synthetic_daily()
    path = tmp_path / "ml_residual_dataset.csv"
    daily.to_csv(path, index=False)
    result = run_ml_horizon_export(path, tmp_path, epsilon_mm=0.01)
    assert result.comparison_path.is_file()
    assert result.highflow_path.is_file()
    assert result.degradation_path.is_file()
    assert (tmp_path / "ml_horizon_forecast_dataset.csv").is_file()
    assert set(result.comparison["horizon_days"]) == {1, 2, 3}
    assert set(result.comparison["model"]) >= {
        MODEL_PHYSICAL,
        MODEL_PERSISTENCE,
        "AR1_residual",
        "ridge",
    }
    # Cross-period leakage blocked: cal origins must not use val targets.
    fc = build_forecast_origin_dataset(daily)
    cal_h3 = filter_horizon_rows(fc, origin_period="calibration", horizon=3)
    assert (cal_h3["target_period_h3"] == "calibration").all()
