"""Tests for Phase 8C residual robustness and ablation analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ml_residual_ablation import (
    FORBIDDEN_PREDICTORS,
    RIDGE_ABLATION_FEATURES,
    calibration_high_flow_threshold,
    fit_ar1_calibration,
    fit_scaled_ridge,
    persistence_residual_hat,
    residual_acf_calibration,
    run_ml_ablation_analysis,
    run_ml_ablation_export,
    temporal_block_metrics,
)
from src.ml_residual_baselines import TRAIN_FEATURE_COLUMNS, split_calibration_validation
from src.ml_residual_dataset import FEATURE_COLUMNS, TARGET_COLUMN


def _synthetic_dataset(n_cal: int = 120, n_val: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = n_cal + n_val
    # Spans 2014–2015-like years for temporal blocks.
    dates = pd.date_range("2014-01-01", periods=n, freq="D")
    q_phys = rng.uniform(0.2, 2.5, n)
    innov = rng.normal(0.0, 0.05, n)
    residual = np.zeros(n)
    for t in range(1, n):
        residual[t] = 0.8 * residual[t - 1] + innov[t]
    q_obs = np.maximum(q_phys + residual, 0.01)
    period = np.array(["calibration"] * n_cal + ["validation"] * n_val)
    # Force first half of validation into 2014 and second into 2015 labels via dates:
    # dates already start 2014; extend so validation crosses into 2015.
    dates = pd.date_range("2013-06-01", periods=n, freq="D")
    frame = {
        "date": dates.strftime("%Y-%m-%d"),
        "period": period,
        "q_obs": q_obs,
        "q_phys": q_phys,
        TARGET_COLUMN: residual,
        "residual_lag_1": pd.Series(residual).shift(1).fillna(0.0).to_numpy(),
    }
    for col in FEATURE_COLUMNS:
        if col in frame:
            continue
        if col == "precipitation_1d":
            frame[col] = rng.uniform(0.0, 5.0, n)
        elif col.startswith("precipitation_"):
            frame[col] = rng.uniform(0.0, 20.0, n)
        elif col in {"production_store", "routing_store"}:
            frame[col] = rng.uniform(10.0, 100.0, n)
        elif col == "q_obs_lag_1":
            frame[col] = pd.Series(q_obs).shift(1).fillna(q_obs[0]).to_numpy()
        else:
            frame[col] = rng.normal(0.0, 1.0, n)
    return pd.DataFrame(frame)


def test_persistence_uses_t_minus_1_only() -> None:
    dataset = _synthetic_dataset()
    _, val = split_calibration_validation(dataset)
    hat = persistence_residual_hat(val)
    np.testing.assert_allclose(hat, val["residual_lag_1"].to_numpy())
    # Must not equal contemporaneous residual except by chance.
    assert not np.allclose(hat, val[TARGET_COLUMN].to_numpy())


def test_ar1_coefficients_fit_on_calibration_only() -> None:
    dataset = _synthetic_dataset()
    cal, val = split_calibration_validation(dataset)
    ar1 = fit_ar1_calibration(cal)
    assert ar1.fitted_on == "calibration"

    # Poison validation residuals; AR(1) fit must be unchanged.
    poisoned = dataset.copy()
    poisoned.loc[poisoned["period"] == "validation", TARGET_COLUMN] = 1.0e6
    poisoned.loc[poisoned["period"] == "validation", "residual_lag_1"] = 1.0e6
    cal_p, _ = split_calibration_validation(poisoned)
    ar1_p = fit_ar1_calibration(cal_p)
    assert ar1.intercept == pytest.approx(ar1_p.intercept)
    assert ar1.phi == pytest.approx(ar1_p.phi)

    # Fitting on all rows would change coefficients after poison.
    bad = fit_ar1_calibration(poisoned)
    assert abs(bad.phi - ar1.phi) > 1e-6 or abs(bad.intercept - ar1.intercept) > 1e-6


def test_feature_scaling_fit_on_calibration_only() -> None:
    dataset = _synthetic_dataset()
    cal, val = split_calibration_validation(dataset)
    feats = ["residual_lag_1", "production_store"]
    model = fit_scaled_ridge(cal, feats, name="test")

    poisoned = val.copy()
    poisoned["production_store"] = 1.0e9
    # Scaler mean/scale must match calibration-only fit, not validation.
    assert model.scaler.mean_[1] == pytest.approx(cal["production_store"].mean())
    pred_clean = model.predict(val)
    pred_poison = model.predict(poisoned)
    # Predictions change with poisoned features, but scaler stats stay calibration.
    assert model.scaler.mean_[1] == pytest.approx(cal["production_store"].mean())
    assert not np.allclose(pred_clean, pred_poison)


def test_high_flow_threshold_from_calibration_only() -> None:
    dataset = _synthetic_dataset()
    cal, val = split_calibration_validation(dataset)
    thr = calibration_high_flow_threshold(cal, quantile=0.9)
    assert thr == pytest.approx(float(np.quantile(cal["q_obs"], 0.9)))

    poisoned = dataset.copy()
    poisoned.loc[poisoned["period"] == "validation", "q_obs"] = 1.0e6
    cal_p, _ = split_calibration_validation(poisoned)
    thr_p = calibration_high_flow_threshold(cal_p, quantile=0.9)
    assert thr == pytest.approx(thr_p)


def test_validation_blocks_do_not_retrain() -> None:
    dataset = _synthetic_dataset()
    cal, val = split_calibration_validation(dataset)
    # Fixed predictions: physical only.
    preds = {"physical": val["q_phys"].to_numpy(dtype=float)}
    temporal = temporal_block_metrics(val, preds, epsilon_mm=0.01)
    assert set(temporal["block"]).issubset({"2013", "2014", "2015"})
    # Same physical series sliced — no fitting involved.
    assert (temporal["model"] == "physical").all()
    _ = cal


def test_ablation_models_use_declared_features() -> None:
    for name, feats in RIDGE_ABLATION_FEATURES.items():
        assert "residual_lag_1" in feats
        assert not set(feats) & FORBIDDEN_PREDICTORS
        if name == "ridge_A_lag1":
            assert feats == ["residual_lag_1"]
        if name == "ridge_B_lag1_qphys":
            assert feats == ["residual_lag_1", "q_phys"]
        if name == "ridge_C_lag1_states":
            assert set(feats) == {"residual_lag_1", "production_store", "routing_store"}
        if name == "ridge_D_lag1_rainfall":
            assert "precipitation_30d" in feats
        if name == "ridge_E_all_safe":
            assert feats == list(TRAIN_FEATURE_COLUMNS)
            assert "q_obs_change_1d" not in feats


def test_no_future_observed_variables_in_predictors() -> None:
    assert "q_obs" in FORBIDDEN_PREDICTORS
    assert "q_obs_change_1d" in FORBIDDEN_PREDICTORS
    dataset = _synthetic_dataset()
    ablation, _coef, _events, _temporal, acf, answers, _acf_sum = run_ml_ablation_analysis(
        dataset, epsilon_mm=0.01
    )
    assert not ablation.empty
    assert set(acf["period"]) == {"calibration"}
    assert "1_persistence_most_of_gain" in answers

    with pytest.raises(ValueError, match="forbidden"):
        fit_scaled_ridge(
            split_calibration_validation(dataset)[0],
            ["residual_lag_1", "q_obs"],
            name="leak",
        )


def test_acf_uses_calibration_only() -> None:
    dataset = _synthetic_dataset()
    cal, val = split_calibration_validation(dataset)
    acf = residual_acf_calibration(cal, max_lag=10)
    assert list(acf["lag"]) == list(range(1, 11))
    # Poisoning validation must not affect calibration ACF.
    poisoned = dataset.copy()
    poisoned.loc[poisoned["period"] == "validation", TARGET_COLUMN] = 999.0
    cal_p, _ = split_calibration_validation(poisoned)
    acf_p = residual_acf_calibration(cal_p, max_lag=10)
    pd.testing.assert_frame_equal(acf, acf_p)
    _ = val


def test_export_writes_required_artifacts(tmp_path: Path) -> None:
    dataset = _synthetic_dataset()
    path = tmp_path / "ml_residual_dataset.csv"
    dataset.to_csv(path, index=False)
    result = run_ml_ablation_export(path, tmp_path, epsilon_mm=0.01)
    assert result.ablation_path.is_file()
    assert result.event_path.is_file()
    assert result.temporal_path.is_file()
    assert result.acf_path.is_file()
    assert result.coefficients_path.is_file()
    assert "physical" in set(result.ablation["model"])
    assert "persistence" in set(result.ablation["model"])
    assert "ar1" in set(result.ablation["model"])
    assert "ridge_A_lag1" in set(result.ablation["model"])
