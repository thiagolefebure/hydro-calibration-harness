"""Tests for Phase 8A residual ML dataset construction (leakage + schema)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.gr4j import GR4JParameters, run_gr4j
from src.ml_residual_dataset import (
    DATASET_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    apply_imputation,
    build_ml_residual_dataset,
    build_residual_feature_frame,
    drop_unusable_rows,
    fit_calibration_imputation,
    lag_past,
    run_ml_residual_dataset_export,
    simulate_best_physical,
    trailing_mean,
    trailing_sum,
)
from src.validation import period_masks, select_best_calibration_candidate


@pytest.fixture
def ml_config() -> dict:
    return {
        "periods": {
            "warmup": ["2010-01-01", "2010-01-31"],
            "calibration": ["2010-02-01", "2010-03-02"],
            "validation": ["2010-03-03", "2010-04-01"],
        },
        "model": {
            "parameter_bounds": {
                "X1": [200.0, 400.0],
                "X2": [-1.0, 1.0],
                "X3": [50.0, 150.0],
                "X4": [1.2, 2.0],
            }
        },
        "calibration": {"behavioral_kge_threshold": 0.80},
        "metrics": {"log_nse_epsilon_mm": 0.01},
    }


@pytest.fixture
def ml_data(ml_config: dict) -> pd.DataFrame:
    index = pd.date_range("2010-01-01", "2010-04-01", freq="D")
    rng = np.random.default_rng(7)
    # Distinct precip regimes so leakage bugs are detectable.
    precip = np.zeros(len(index), dtype=float)
    precip[:] = rng.uniform(0.0, 2.0, len(index))
    precip[-5:] = 50.0  # late validation spike
    return pd.DataFrame(
        {
            "precipitation_mm": precip,
            "et0_mm": rng.uniform(0.5, 3.0, len(index)),
            "discharge_mm": rng.uniform(0.1, 3.0, len(index)),
        },
        index=index,
    )


@pytest.fixture
def ml_runs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": [10, 20],
            "x1": [250.0, 300.0],
            "x2": [0.0, -0.5],
            "x3": [80.0, 90.0],
            "x4": [1.5, 1.6],
            "kge_cal": [0.91, 0.70],
            "kge_val": [0.40, 0.99],
        }
    )


def test_lag_direction_uses_past_only() -> None:
    s = pd.Series([10.0, 20.0, 30.0, 40.0], index=pd.RangeIndex(4))
    lagged = lag_past(s, 1)
    assert np.isnan(lagged.iloc[0])
    assert lagged.iloc[1] == 10.0
    assert lagged.iloc[2] == 20.0
    assert lagged.iloc[3] == 30.0
    # Never equals a future value relative to source index.
    for t in range(1, len(s)):
        assert lagged.iloc[t] == s.iloc[t - 1]
        assert lagged.iloc[t] != s.iloc[t]


def test_rolling_window_is_trailing_not_centered() -> None:
    s = pd.Series([1.0, 2.0, 4.0, 8.0, 16.0])
    win3 = trailing_sum(s, 3)
    assert np.isnan(win3.iloc[0])
    assert np.isnan(win3.iloc[1])
    assert win3.iloc[2] == pytest.approx(1.0 + 2.0 + 4.0)
    assert win3.iloc[3] == pytest.approx(2.0 + 4.0 + 8.0)
    assert win3.iloc[4] == pytest.approx(4.0 + 8.0 + 16.0)

    # Centered window would mix future day-4 into day-2 sum; trailing must not.
    centered = s.rolling(window=3, center=True, min_periods=3).sum()
    assert win3.iloc[2] != pytest.approx(centered.iloc[2])

    mean7 = trailing_mean(pd.Series(np.arange(10.0)), 7)
    assert np.isnan(mean7.iloc[5])
    assert mean7.iloc[6] == pytest.approx(np.mean(np.arange(7.0)))


def test_no_validation_leakage_in_imputation_fit(
    ml_data: pd.DataFrame,
    ml_config: dict,
    ml_runs: pd.DataFrame,
) -> None:
    best = select_best_calibration_candidate(ml_runs)
    q_phys, states = simulate_best_physical(ml_data, ml_config, best)
    raw = build_residual_feature_frame(ml_data, ml_config, q_phys, states)

    # Poison validation precip so any val-using fit would shift medians.
    masks = period_masks(raw.index, ml_config)
    poisoned = raw.copy()
    poisoned.loc[masks["validation"], "precipitation_30d"] = 1.0e9

    fit_clean = fit_calibration_imputation(raw, ml_config, columns=["precipitation_30d"])
    fit_poison = fit_calibration_imputation(poisoned, ml_config, columns=["precipitation_30d"])
    assert fit_clean["precipitation_30d"] == pytest.approx(fit_poison["precipitation_30d"])

    # Fitting on all rows (incorrect) would differ after poison.
    bad_all = float(poisoned["precipitation_30d"].median(skipna=True))
    assert bad_all != pytest.approx(fit_clean["precipitation_30d"])

    filled = apply_imputation(raw, fit_clean)
    assert filled.loc[masks["validation"], "precipitation_1d"].notna().all()


def test_target_definition_residual_equals_obs_minus_phys(
    ml_data: pd.DataFrame,
    ml_config: dict,
    ml_runs: pd.DataFrame,
) -> None:
    dataset, summary, best_id = build_ml_residual_dataset(ml_data, ml_config, ml_runs)
    assert best_id == 10
    assert summary["target"] == "residual = q_obs - q_phys"
    np.testing.assert_allclose(
        dataset[TARGET_COLUMN].to_numpy(),
        dataset["q_obs"].to_numpy() - dataset["q_phys"].to_numpy(),
        rtol=0.0,
        atol=1e-12,
    )


def test_state_alignment_with_gr4j_step(
    ml_data: pd.DataFrame,
    ml_config: dict,
    ml_runs: pd.DataFrame,
) -> None:
    best = select_best_calibration_candidate(ml_runs)
    q_phys, states = simulate_best_physical(ml_data, ml_config, best)
    params = GR4JParameters(
        X1=float(best["x1"]),
        X2=float(best["x2"]),
        X3=float(best["x3"]),
        X4=float(best["x4"]),
    )
    inputs = ml_data[["precipitation_mm", "et0_mm"]].dropna()
    q_ref, final, states_ref = run_gr4j(inputs, params, return_states=True)

    pd.testing.assert_series_equal(q_phys, q_ref)
    pd.testing.assert_frame_equal(states, states_ref)
    assert states.index.equals(q_phys.index)
    assert states["production_store"].iloc[-1] == pytest.approx(final.production_store)
    assert states["routing_store"].iloc[-1] == pytest.approx(final.routing_store)

    frame = build_residual_feature_frame(ml_data, ml_config, q_phys, states)
    pd.testing.assert_series_equal(
        frame["production_store"],
        states["production_store"],
        check_names=False,
    )


def test_output_schema_and_causal_feature_rows(
    ml_data: pd.DataFrame,
    ml_config: dict,
    ml_runs: pd.DataFrame,
    tmp_path: Path,
) -> None:
    result = run_ml_residual_dataset_export(ml_data, ml_config, ml_runs, tmp_path)
    dataset = result.dataset

    assert list(dataset.columns) == DATASET_COLUMNS
    assert set(FEATURE_COLUMNS).issubset(dataset.columns)
    assert (tmp_path / "ml_residual_dataset.csv").is_file()
    assert (tmp_path / "ml_residual_dataset_summary.json").is_file()

    # After dropping incomplete causal windows, required columns are finite.
    assert dataset[FEATURE_COLUMNS + [TARGET_COLUMN]].notna().all().all()
    assert result.summary["total_rows"] == len(dataset)
    assert result.summary["rows_calibration"] + result.summary["rows_validation"] <= len(
        dataset
    )

    # Lag features must equal previous-day values on the raw frame before drop.
    best = select_best_calibration_candidate(ml_runs)
    q_phys, states = simulate_best_physical(ml_data, ml_config, best)
    raw = build_residual_feature_frame(ml_data, ml_config, q_phys, states)
    usable = drop_unusable_rows(raw)
    raw_by_date = raw.set_index(pd.to_datetime(raw["date"]))
    for _, row in usable.iterrows():
        day = pd.Timestamp(row["date"])
        prev = day - pd.Timedelta(days=1)
        if prev in raw_by_date.index:
            assert row["q_obs_lag_1"] == pytest.approx(raw_by_date.loc[prev, "q_obs"])
            assert row["residual_lag_1"] == pytest.approx(raw_by_date.loc[prev, "residual"])


def test_precipitation_features_do_not_use_future_days(
    ml_data: pd.DataFrame,
    ml_config: dict,
    ml_runs: pd.DataFrame,
) -> None:
    best = select_best_calibration_candidate(ml_runs)
    q_phys, states = simulate_best_physical(ml_data, ml_config, best)
    frame = build_residual_feature_frame(ml_data, ml_config, q_phys, states)

    # At the day before the validation spike, 1d precip must not equal the spike.
    spike_start = frame.index[-5]
    day_before = spike_start - pd.Timedelta(days=1)
    assert frame.loc[day_before, "precipitation_1d"] != pytest.approx(50.0)
    assert frame.loc[spike_start, "precipitation_1d"] == pytest.approx(50.0)

    # 3-day trailing sum on day_before excludes all spike days.
    expected = float(
        ml_data.loc[
            day_before - pd.Timedelta(days=2) : day_before, "precipitation_mm"
        ].sum()
    )
    assert frame.loc[day_before, "precipitation_3d"] == pytest.approx(expected)
