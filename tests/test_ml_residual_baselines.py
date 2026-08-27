"""Tests for Phase 8B residual ML baselines."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ml_residual_baselines import (
    COMPARISON_COLUMNS,
    HGB_CONFIG,
    MODEL_DUMMY,
    MODEL_HGB,
    MODEL_PHYSICAL,
    MODEL_RIDGE,
    PREDICTION_COLUMNS,
    RIDGE_ALPHA,
    TRAIN_FEATURE_COLUMNS,
    DummyMeanResidual,
    hybrid_discharge,
    run_ml_residual_baselines_export,
    run_residual_baselines,
    split_calibration_validation,
    verdict_from_comparison,
)
from src.ml_residual_dataset import FEATURE_COLUMNS, TARGET_COLUMN


def _synthetic_dataset(n_cal: int = 80, n_val: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = n_cal + n_val
    dates = pd.date_range("2011-01-01", periods=n, freq="D")
    q_phys = rng.uniform(0.2, 2.0, n)
    # Residual partly predictable from precipitation_1d on calibration-like structure.
    precip = rng.uniform(0.0, 5.0, n)
    residual = 0.1 * precip - 0.05 + rng.normal(0.0, 0.05, n)
    q_obs = np.maximum(q_phys + residual, 0.01)
    frame = {
        "date": dates.strftime("%Y-%m-%d"),
        "period": ["calibration"] * n_cal + ["validation"] * n_val,
        "q_obs": q_obs,
        "q_phys": q_phys,
        TARGET_COLUMN: q_obs - q_phys,
    }
    for col in FEATURE_COLUMNS:
        if col == "precipitation_1d":
            frame[col] = precip
        elif col == "residual_lag_1":
            frame[col] = pd.Series(q_obs - q_phys).shift(1).fillna(0.0).to_numpy()
        elif col == "q_obs_lag_1":
            frame[col] = pd.Series(q_obs).shift(1).fillna(q_obs[0]).to_numpy()
        else:
            frame[col] = rng.normal(0.0, 1.0, n)
    return pd.DataFrame(frame)


def test_hybrid_discharge_definition() -> None:
    q_phys = np.array([1.0, 2.0, 0.1])
    residual_hat = np.array([0.5, -3.0, 0.0])
    hybrid = hybrid_discharge(q_phys, residual_hat)
    np.testing.assert_allclose(hybrid, [1.5, 0.0, 0.1])


def test_dummy_equals_calibration_mean_residual() -> None:
    dataset = _synthetic_dataset()
    cal, val = split_calibration_validation(dataset)
    model = DummyMeanResidual()
    x_cal = cal[TRAIN_FEATURE_COLUMNS].to_numpy()
    y_cal = cal[TARGET_COLUMN].to_numpy()
    model.fit(x_cal, y_cal)
    preds = model.predict(val[TRAIN_FEATURE_COLUMNS].to_numpy())
    assert preds == pytest.approx(np.full(len(val), float(y_cal.mean())))


def test_excludes_contemporaneous_q_obs_change_feature() -> None:
    assert "q_obs_change_1d" not in TRAIN_FEATURE_COLUMNS
    assert "q_obs_lag_1" in TRAIN_FEATURE_COLUMNS
    dataset = _synthetic_dataset()
    _comparison, _preds, meta = run_residual_baselines(dataset, epsilon_mm=0.01)
    assert "q_obs_change_1d" in meta["excluded_leaky_features"]
    assert "q_obs_change_1d" not in meta["feature_columns"]


def test_train_only_on_calibration_rows() -> None:
    dataset = _synthetic_dataset()
    cal, val = split_calibration_validation(dataset)
    # Mark validation features with a sentinel; models must not need them for fit.
    poisoned = dataset.copy()
    poisoned.loc[poisoned["period"] == "validation", TRAIN_FEATURE_COLUMNS[0]] = 1.0e6

    comparison_a, preds_a, meta = run_residual_baselines(dataset, epsilon_mm=0.01)
    comparison_b, _preds_b, _ = run_residual_baselines(poisoned, epsilon_mm=0.01)

    assert meta["tuning"].startswith("none")
    assert meta["ridge_alpha"] == RIDGE_ALPHA
    assert meta["hgb_config"] == HGB_CONFIG
    assert meta["n_calibration"] == len(cal)
    assert meta["n_validation"] == len(val)

    # Dummy depends only on calibration target mean — identical under feature poison.
    dummy_a = comparison_a.loc[comparison_a["model"] == MODEL_DUMMY, "kge_val"].iloc[0]
    dummy_b = comparison_b.loc[comparison_b["model"] == MODEL_DUMMY, "kge_val"].iloc[0]
    assert dummy_a == pytest.approx(dummy_b)

    assert set(preds_a["period"].unique()) == {"validation"}


def test_no_validation_hyperparameter_tuning_uses_fixed_config() -> None:
    assert RIDGE_ALPHA == 1.0
    assert HGB_CONFIG["random_state"] == 42
    assert HGB_CONFIG["max_iter"] == 200
    dataset = _synthetic_dataset()
    _comparison, _preds, meta = run_residual_baselines(dataset, epsilon_mm=0.01)
    assert "validation unused" in meta["tuning"]


def test_output_schemas_and_physical_baseline(tmp_path: Path) -> None:
    dataset = _synthetic_dataset()
    path = tmp_path / "ml_residual_dataset.csv"
    dataset.to_csv(path, index=False)
    result = run_ml_residual_baselines_export(path, tmp_path, epsilon_mm=0.01)

    assert list(result.comparison.columns) == COMPARISON_COLUMNS
    assert list(result.predictions.columns) == PREDICTION_COLUMNS
    assert set(result.comparison["model"]) == {
        MODEL_PHYSICAL,
        MODEL_DUMMY,
        MODEL_RIDGE,
        MODEL_HGB,
    }

    phys_preds = result.predictions.loc[result.predictions["model"] == MODEL_PHYSICAL]
    np.testing.assert_allclose(phys_preds["residual_pred"], 0.0)
    np.testing.assert_allclose(phys_preds["q_hybrid"], phys_preds["q_phys"])
    np.testing.assert_allclose(
        phys_preds["q_hybrid"],
        phys_preds["q_phys"] + phys_preds["residual_pred"],
        atol=1e-12,
    )

    # Hybrid definition for ML rows (after non-negative clip).
    ml_preds = result.predictions.loc[result.predictions["model"] != MODEL_PHYSICAL]
    expected = np.maximum(ml_preds["q_phys"] + ml_preds["residual_pred"], 0.0)
    np.testing.assert_allclose(ml_preds["q_hybrid"], expected)


def test_verdict_does_not_claim_improvement_when_worse() -> None:
    comparison = pd.DataFrame(
        {
            "model": [MODEL_PHYSICAL, MODEL_DUMMY, MODEL_RIDGE],
            "kge_val": [0.80, 0.50, 0.70],
            "nse_val": [0.7, 0.4, 0.6],
            "lognse_val": [0.6, 0.3, 0.5],
            "bias_val": [0.0, 0.1, 0.05],
            "mae_val": [0.1, 0.2, 0.15],
            "rmse_val": [0.2, 0.3, 0.25],
        }
    )
    improves, best, verdict = verdict_from_comparison(comparison)
    assert improves is False
    assert best == MODEL_RIDGE
    assert "does not improve" in verdict
