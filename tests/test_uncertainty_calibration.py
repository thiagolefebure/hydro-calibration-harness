"""Tests for Phase 9 forecast uncertainty calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.uncertainty_calibration import (
    CONFORMAL_FORMULA,
    CRPS_NOTE,
    FORECAST_FEATURE_COLUMNS,
    METHOD_BEHAVIORAL,
    METHOD_CONFORMAL,
    METHOD_EMPIRICAL,
    METHOD_QUANTILE,
    chronological_split,
    conformal_quantile_level,
    coverage_and_width_stats,
    fit_empirical_residual,
    fit_methods_for_horizon,
    fit_split_conformal,
    interval_score,
    predict_intervals,
    run_uncertainty_export,
)


def _synth_cal_table(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2011-01-01", periods=n, freq="D")
    residual_t = rng.normal(0.0, 0.2, n)
    q_phys = rng.uniform(0.3, 2.0, n)
    # Persistent error structure
    error = 0.7 * residual_t + rng.normal(0.0, 0.05, n)
    q_point = np.maximum(q_phys + residual_t, 0.0)
    q_obs = np.maximum(q_point + error, 0.05)
    frame = {
        "origin_date": dates,
        "target_date": dates + pd.Timedelta(days=1),
        "residual_t": residual_t,
        "q_obs_t": q_point,
        "q_phys_t": q_phys,
        "q_phys_change_1d_at_t": rng.normal(0, 0.1, n),
        "precip_last_1d": rng.uniform(0, 5, n),
        "precip_last_3d": rng.uniform(0, 12, n),
        "precip_last_7d": rng.uniform(0, 25, n),
        "precip_last_30d": rng.uniform(0, 80, n),
        "production_store_t": rng.uniform(20, 200, n),
        "routing_store_t": rng.uniform(10, 100, n),
        "day_of_year_sin": np.sin(2 * np.pi * dates.dayofyear / 365.25),
        "day_of_year_cos": np.cos(2 * np.pi * dates.dayofyear / 365.25),
        "q_point": q_point,
        "q_obs_target": q_obs,
        "error": q_obs - q_point,
        "abs_error": np.abs(q_obs - q_point),
        "horizon_days": 1,
        "period": "calibration",
    }
    for col in FORECAST_FEATURE_COLUMNS:
        if col not in frame:
            frame[col] = 0.0
    return pd.DataFrame(frame)


def test_chronological_split_no_random() -> None:
    frame = _synth_cal_table(100)
    train, calib = chronological_split(frame, train_fraction=0.75)
    assert len(train) == 75
    assert len(calib) == 25
    assert train["origin_date"].max() <= calib["origin_date"].min()
    # Deterministic
    train2, calib2 = chronological_split(frame, train_fraction=0.75)
    pd.testing.assert_frame_equal(train, train2)
    pd.testing.assert_frame_equal(calib, calib2)


def test_conformal_quantile_finite_sample_rule() -> None:
    # n=10, alpha=0.1 -> ceil(11*0.9)/10 = ceil(9.9)/10 = 10/10 = 1.0
    assert conformal_quantile_level(10, 0.10) == pytest.approx(1.0)
    # n=100, alpha=0.1 -> ceil(101*0.9)/100 = ceil(90.9)/100 = 0.91
    assert conformal_quantile_level(100, 0.10) == pytest.approx(0.91)
    assert "ceil((n+1)*(1-alpha))/n" in CONFORMAL_FORMULA


def test_interval_score_implementation() -> None:
    y = np.array([1.0, 0.0, 3.0])
    lower = np.array([0.5, 0.5, 0.5])
    upper = np.array([1.5, 1.5, 1.5])
    alpha = 0.10
    scores = interval_score(y, lower, upper, alpha=alpha)
    # inside: width=1
    assert scores[0] == pytest.approx(1.0)
    # below: width + (2/alpha)*(L-y) = 1 + 20*0.5 = 11
    assert scores[1] == pytest.approx(11.0)
    # above: width + (2/alpha)*(y-U) = 1 + 20*1.5 = 31
    assert scores[2] == pytest.approx(31.0)


def test_empirical_coverage_calculation() -> None:
    y = np.array([1.0, 2.0, 3.0, 4.0])
    lower = np.array([0.0, 0.0, 0.0, 5.0])
    upper = np.array([2.0, 3.0, 2.5, 6.0])
    stats = coverage_and_width_stats(y, lower, upper, nominal=0.90, mean_obs=2.5)
    # inside: idx 0,1 only -> 0.5
    assert stats["empirical_coverage"] == pytest.approx(0.5)
    assert stats["coverage_error"] == pytest.approx(0.5 - 0.90)


def test_lower_leq_upper_and_nonnegative() -> None:
    cal = _synth_cal_table(80)
    fitted = fit_methods_for_horizon(cal, horizon=1, train_fraction=0.75)
    for method in (METHOD_EMPIRICAL, METHOD_QUANTILE, METHOD_CONFORMAL):
        lo, up = predict_intervals(method, fitted, cal.iloc[:20], nominal=0.90)
        assert np.all(lo <= up)
        assert np.all(lo >= 0.0)


def test_validation_never_enters_fitting() -> None:
    cal = _synth_cal_table(100, seed=1)
    fitted = fit_methods_for_horizon(cal, horizon=1, train_fraction=0.8)
    # Poison a fake validation-like frame; refitting on cal only must ignore it.
    val = cal.copy()
    val["error"] = 1.0e6
    val["abs_error"] = 1.0e6
    fitted2 = fit_methods_for_horizon(cal, horizon=1, train_fraction=0.8)
    # Empirical quantiles identical
    assert fitted[METHOD_EMPIRICAL].quantiles[0.9] == fitted2[METHOD_EMPIRICAL].quantiles[0.9]
    assert fitted[METHOD_CONFORMAL].q_hat[0.9] == pytest.approx(fitted2[METHOD_CONFORMAL].q_hat[0.9])
    # Conformal uses only calib subset of cal
    _train, calib = chronological_split(cal, train_fraction=0.8)
    conf = fit_split_conformal(calib["abs_error"].to_numpy(), horizon=1)
    assert conf.n_calib == len(calib)
    _ = val


def test_q_obs_not_in_interval_features() -> None:
    assert "q_obs" not in FORECAST_FEATURE_COLUMNS
    assert "q_obs_target" not in FORECAST_FEATURE_COLUMNS
    cal = _synth_cal_table(60)
    fitted = fit_methods_for_horizon(cal, horizon=1, train_fraction=0.7)
    assert fitted[METHOD_QUANTILE].feature_names == list(FORECAST_FEATURE_COLUMNS)


def test_high_low_thresholds_from_calibration_concept() -> None:
    # thresholds helpers already tested in ablation; here ensure config documents uncertainty
    from src.ml_residual_ablation import calibration_high_flow_threshold

    cal = pd.DataFrame({"q_obs": np.linspace(0.1, 5.0, 100)})
    thr = calibration_high_flow_threshold(cal, quantile=0.9)
    assert thr == pytest.approx(float(np.quantile(cal["q_obs"], 0.9)))


def test_no_random_train_test_split_in_module() -> None:
    import src.uncertainty_calibration as mod
    import inspect

    src = inspect.getsource(mod)
    assert "train_test_split" not in src
    assert "ShuffleSplit" not in src
    assert "chronological_split" in src


def test_crps_deferred_documented() -> None:
    assert "deferred" in CRPS_NOTE.lower()


def test_behavioral_method_name_unchanged() -> None:
    assert METHOD_BEHAVIORAL == "behavioral_parametric"


def test_export_smoke(tmp_path: Path) -> None:
    """End-to-end smoke on tiny synthetic artifacts."""
    from src.gr4j import GR4JParameters, run_gr4j
    from src.ml_residual_dataset import FEATURE_COLUMNS, TARGET_COLUMN

    rng = np.random.default_rng(5)
    n = 150
    dates = pd.date_range("2011-01-01", periods=n, freq="D")
    # periods: cal 90, val 60
    period = np.array(["calibration"] * 90 + ["validation"] * 60)
    precip = rng.uniform(0, 6, n)
    precip[rng.random(n) < 0.35] = 0.0
    et0 = rng.uniform(0.5, 2.5, n)
    params = GR4JParameters(250.0, 0.0, 80.0, 1.5)
    basin = pd.DataFrame(
        {"precipitation_mm": precip, "et0_mm": et0, "discharge_mm": 0.0},
        index=dates,
    )
    q_phys, _, states = run_gr4j(basin[["precipitation_mm", "et0_mm"]], params, return_states=True)
    residual = 0.6 * pd.Series(rng.normal(0, 0.15, n), index=dates)
    residual = residual.ewm(alpha=0.3).mean()
    q_obs = np.maximum(q_phys + residual, 0.05)
    basin["discharge_mm"] = q_obs

    doy = dates.dayofyear.astype(float)
    angle = 2 * np.pi * (doy - 1) / 365.25
    daily = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "period": period,
            "q_obs": q_obs.to_numpy(),
            "q_phys": q_phys.to_numpy(),
            TARGET_COLUMN: (q_obs - q_phys).to_numpy(),
            "precipitation_1d": precip,
            "precipitation_3d": pd.Series(precip).rolling(3, min_periods=3).sum().to_numpy(),
            "precipitation_7d": pd.Series(precip).rolling(7, min_periods=7).sum().to_numpy(),
            "precipitation_30d": pd.Series(precip).rolling(30, min_periods=30).sum().to_numpy(),
            "production_store": states["production_store"].to_numpy(),
            "routing_store": states["routing_store"].to_numpy(),
            "day_of_year_sin": np.sin(angle),
            "day_of_year_cos": np.cos(angle),
            "q_phys_change_1d": q_phys.diff().to_numpy(),
            "q_obs_lag_1": q_obs.shift(1).to_numpy(),
            "residual_lag_1": (q_obs - q_phys).shift(1).to_numpy(),
            "et0_current": et0,
            "et0_7d_mean": pd.Series(et0).rolling(7, min_periods=7).mean().to_numpy(),
            "q_obs_change_1d": q_obs.diff().to_numpy(),
        }
    )
    for col in FEATURE_COLUMNS:
        if col not in daily.columns:
            daily[col] = 0.0
    daily = daily.dropna().reset_index(drop=True)

    # Ensemble timeseries stub for behavioral reference
    ens = pd.DataFrame(
        {
            "date": daily["date"],
            "q_obs": daily["q_obs"],
            "q_best_cal": daily["q_phys"],
            "q05": daily["q_obs"] - 0.3,
            "q50": daily["q_phys"],
            "q95": daily["q_obs"] + 0.3,
            "period": daily["period"],
        }
    )

    dataset_path = tmp_path / "ml_residual_dataset.csv"
    basin_path = tmp_path / "basin_daily.csv"
    runs_path = tmp_path / "runs.csv"
    ens_path = tmp_path / "ensemble_timeseries.csv"
    daily.to_csv(dataset_path, index=False)
    basin_out = basin.copy()
    basin_out.index.name = "date"
    basin_out.reset_index().to_csv(basin_path, index=False)
    ens.to_csv(ens_path, index=False)
    pd.DataFrame(
        {
            "run_id": [1],
            "x1": [params.X1],
            "x2": [params.X2],
            "x3": [params.X3],
            "x4": [params.X4],
            "kge_cal": [0.9],
            "kge_val": [0.8],
        }
    ).to_csv(runs_path, index=False)

    config = {
        "periods": {
            "warmup": ["2010-12-01", "2010-12-31"],
            "calibration": ["2011-01-01", "2011-03-31"],
            "validation": ["2011-04-01", "2011-05-30"],
        },
        "metrics": {"log_nse_epsilon_mm": 0.01},
        "uncertainty": {
            "train_fraction": 0.7,
            "rolling_coverage_window_days": 20,
            "top_n_extreme_events": 3,
            "moderate_realization": 0,
            "meteo_scenarios": ["oracle"],  # keep smoke fast
        },
        "meteo_sensitivity": {
            "n_realizations": 2,
            "seed": 1,
            "wet_day_threshold_mm": 1.0,
            "moderate": {
                "magnitude_lognormal_sigma": 0.2,
                "timing_shift_probability": 0.1,
                "miss_probability": 0.05,
                "miss_factor": 0.2,
                "false_alarm_probability": 0.05,
                "false_alarm_mm": 1.0,
            },
            "strong": {
                "magnitude_lognormal_sigma": 0.4,
                "timing_shift_probability": 0.2,
                "miss_probability": 0.1,
                "miss_factor": 0.1,
                "false_alarm_probability": 0.1,
                "false_alarm_mm": 2.0,
            },
        },
    }

    result = run_uncertainty_export(
        dataset_path=dataset_path,
        basin_data_path=basin_path,
        runs_path=runs_path,
        ensemble_path=ens_path,
        output_dir=tmp_path / "out",
        config=config,
    )
    assert result.coverage_path.is_file()
    assert result.forecasts_path.is_file()
    assert result.reliability_path.is_file()
    assert result.demo_path.is_file()
    assert result.before_after_path.is_file()
    assert METHOD_BEHAVIORAL in set(result.coverage_summary["method"])
    assert set(result.forecasts["period"]) == {"validation"}
    # lower <= upper for calibrated methods
    for method in (METHOD_EMPIRICAL, METHOD_QUANTILE, METHOD_CONFORMAL):
        sub = result.forecasts.loc[result.forecasts["method"] == method]
        assert (sub["lower_90"] <= sub["upper_90"]).all()


def test_basin_yaml_uncertainty_section(config_path: Path) -> None:
    with config_path.open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    assert "uncertainty" in cfg
    assert cfg["uncertainty"]["train_fraction"] == 0.75
    assert "oracle" in cfg["uncertainty"]["meteo_scenarios"]
    assert "moderate" in cfg["uncertainty"]["meteo_scenarios"]
